"""
app/main_window.py
─────────────────────
Fenêtre principale — assemble les panels et orchestre le flux de données :

    SerialReader (thread) → data_queue → MainWindow → PpgHrProcessor
                                              ↓
                                    SignalPanel (graphiques + BPM)

MainWindow reste responsable de :
  - la connexion/déconnexion série
  - la boucle de traitement (timer 50 ms) qui vide la queue et rafraîchit l'UI
  - l'AGC (auto-tuning + optimisation du BOOST), sur un timer séparé (500 ms)
  - l'enregistrement CSV brut

Les panels ne connaissent que leur propre UI ; toute la coordination est ici,
délibérément — voir la discussion sur MVC vs structure plate du projet.
"""

from __future__ import annotations

import csv
import queue
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets

from app.panels.connection_panel import ConnectionPanel
from app.panels.control_panel import ControlPanel
from app.panels.hardware_panel import HardwarePanel
from app.panels.log_panel import LogPanel
from app.panels.recording_panel import RecordingPanel
from app.panels.signal_panel import SignalPanel
from app.utils import adc_counts_to_mv, ensure_qt_plugins_visible
from app.workers.serial_worker import SerialReader
from control.agc import ADC_FULL_SCALE, AutoTuner, BoostOptimizer
from ml.hr_correction import HrCorrectionModel
from pipeline.core import PpgHrProcessor, PpgMode

ensure_qt_plugins_visible()

PLOT_SECONDS = 8
NOMINAL_FS = 250
BUFFER_LEN = PLOT_SECONDS * NOMINAL_FS
RECORDINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "recordings"

# BPM trend plot: compute() runs every 4th refresh tick (~200ms @ the 50ms
# timer below), so 5 Hz -- 600 points covers a 2-minute window.
BPM_HISTORY_LEN = 600

# Trailing-average window for the "watch-style" smoothed BPM curve, roughly
# matching how Apple Watch / Garmin's optical HR display refreshes (a short
# multi-second average, not a raw beat-to-beat number) -- separate from and
# on top of the tracker's own EMA damping, since a plain windowed mean over
# the already-computed instant readings is the simplest way to show what a
# consumer watch's reading would look like next to ours for comparison.
#
# 5.0 was too short to damp respiratory/vasomotor modulation of the PPG
# amplitude (see pipeline/core.py's adaptive_smoothing comment) -- a resting
# recording (2026-08-11) periodogrammed to a dominant ~9.3s oscillation
# (0.108 Hz, ~6.5 breaths/min) riding the true BPM, and a box average only
# nulls frequencies at multiples of 1/window, so 5s (0.2 Hz) barely touched
# it (~40% attenuation). 15s covers ~1.5-2 respiratory cycles at that rate
# and attenuates the whole typical RSA band (0.1-0.4 Hz, i.e. 6-24
# breaths/min) much harder, at the cost of more display lag on real BPM
# transitions -- acceptable here since this curve is explicitly the smooth
# comparison line, not the responsive one (that's bpm_hist_instant/raw/ml).
WATCH_AVG_SECONDS = 15.0
WATCH_AVG_SAMPLES = int(WATCH_AVG_SECONDS * NOMINAL_FS)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PPG SiPM Monitor")
        self.resize(1280, 900)

        self.reader: SerialReader | None = None
        self.hr_proc = PpgHrProcessor(
            frame_fs=NOMINAL_FS,
            mode=PpgMode.MULTILED,
            fft_window_s=8.0,
            max_buffer_s=14.0,
            correction_model=HrCorrectionModel.load_default(),
        )

        self.t: deque[int] = deque(maxlen=BUFFER_LEN)
        self.buf_ppg: deque[int] = deque(maxlen=BUFFER_LEN)
        self.buf_ax: deque[int] = deque(maxlen=BUFFER_LEN)
        self.buf_ay: deque[int] = deque(maxlen=BUFFER_LEN)
        self.buf_az: deque[int] = deque(maxlen=BUFFER_LEN)
        self.sample_index = 0

        self.current_bpm = 0.0
        self.bpm_valid = False
        self.current_snr = 0.0
        self.current_inband_snr = 0.0
        self.current_ac_amp = 0.0
        self.current_motion = 0.0
        self.beat_indices = np.array([], dtype=int)
        self._latest_result = None
        self._plot_tick = 0

        self.bpm_hist_x: deque[int] = deque(maxlen=BPM_HISTORY_LEN)
        self.bpm_hist_raw: deque[float] = deque(maxlen=BPM_HISTORY_LEN)
        self.bpm_hist_ml: deque[float] = deque(maxlen=BPM_HISTORY_LEN)
        self.bpm_hist_instant: deque[float] = deque(maxlen=BPM_HISTORY_LEN)
        self.bpm_hist_smooth: deque[float] = deque(maxlen=BPM_HISTORY_LEN)

        self.record_file = None
        self.record_writer = None
        self.record_path = None
        self.data_queue: queue.Queue = queue.Queue()

        self.auto_tuner = AutoTuner()
        self.auto_enabled = False
        self.boost_opt = BoostOptimizer()
        self.boost_opt_enabled = False

        self._build_ui()
        self._wire_signals()

        if self.hr_proc.correction_model is not None:
            self._log("# HR correction model loaded (data/models/hr_correction.json)")
            self.control_panel.set_ml_available(True)
        else:
            self._log("# No HR correction model found -- using raw FFT tracker "
                      "(run pc/ml/train_hr_correction.py to train one)")
            self.control_panel.set_ml_available(False)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh_plots)
        self.timer.start(50)

        self.agc_timer = QtCore.QTimer(self)
        self.agc_timer.timeout.connect(self._run_agc)
        self.agc_timer.start(500)

    # ── Construction de l'UI ─────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        self.connection_panel = ConnectionPanel()
        self.control_panel = ControlPanel()
        self.recording_panel = RecordingPanel()
        self.hardware_panel = HardwarePanel()
        self.signal_panel = SignalPanel()
        self.log_panel = LogPanel()

        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(self.connection_panel)
        controls.addWidget(self.control_panel)
        controls.addWidget(self.recording_panel)
        controls.addWidget(self.signal_panel.readout_widget)
        controls.addWidget(self.hardware_panel)
        controls.addWidget(self.log_panel, 1)

        right = QtWidgets.QWidget()
        right.setLayout(controls)
        right.setFixedWidth(440)

        root.addWidget(self.signal_panel.plots_widget, 1)
        root.addWidget(right)

    def _wire_signals(self):
        self.connection_panel.connectRequested.connect(self._connect)
        self.connection_panel.disconnectRequested.connect(self._disconnect)

        self.control_panel.autoToggled.connect(self._on_auto_toggled)
        self.control_panel.motionToggled.connect(
            lambda c: setattr(self.hr_proc, "use_motion_cancel", c))
        self.control_panel.boostOptToggled.connect(self._on_boost_opt_toggled)
        self.control_panel.mlToggled.connect(
            lambda c: setattr(self.hr_proc, "use_ml_correction", c))

        self.recording_panel.startRequested.connect(self._start_recording)
        self.recording_panel.stopRequested.connect(self._stop_recording)

        self.hardware_panel.commandIssued.connect(self._send)
        self.hardware_panel.hvToggled.connect(self._on_hv_toggled)

    # ── Connexion série ──────────────────────────────────────────────────

    def _connect(self, port: str):
        self.hr_proc.reset()
        for buf in (self.t, self.buf_ppg, self.buf_ax, self.buf_ay, self.buf_az):
            buf.clear()
        self.sample_index = 0
        self.beat_indices = np.array([], dtype=int)
        self.current_bpm = 0.0
        self.bpm_valid = False
        self._latest_result = None
        self._plot_tick = 0
        self.bpm_hist_x.clear()
        self.bpm_hist_raw.clear()
        self.bpm_hist_ml.clear()
        self.bpm_hist_instant.clear()
        self.bpm_hist_smooth.clear()
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        self.reader = SerialReader(port, self.data_queue)
        self.reader.message.connect(self._log)
        self.reader.error.connect(self._on_error)
        self.reader.ready.connect(self._on_reader_ready)
        self.reader.start()
        self.connection_panel.set_connected_ui(True)
        self._log(f"# Connected to {port}")

    def _on_reader_ready(self):
        # Opening the port often resets the ESP32 (DTR) and its setup()
        # (I2C init for the digital pots / ADC / accelerometer) takes real
        # time -- commands sent before this fired would be silently
        # dropped (port not open yet) or ignored (firmware not ready).
        # SerialReader only emits this once the link is actually live: the
        # first real sample arrived, or a timeout elapsed as a fallback.
        hw = self.hardware_panel
        self._send(f"ADSGAIN:{hw.sl_adsgain.slider.value()}")
        self._send(f"GAIN:{hw.sl_gain.slider.value()}")
        self._send("DCSERVO:0")
        self._send("DAC:0")
        self._send(f"BOOST:{hw.sl_boost.slider.value()}")
        self._send(f"LED1:{hw.sl_led1.slider.value()}")
        self._send(f"LED2:{hw.sl_led2.slider.value()}")
        self._send(f"HVEN:{1 if hw.is_hv_enabled() else 0}")

        # Seed AGC from the connect-time preset, then let it track ambient
        # light / skin tone drift automatically instead of requiring a
        # manual toggle each session. BOOST auto-optimize stays manual --
        # it stalls whenever bpm_valid is false (no usable quality samples),
        # which needs more bench validation before it runs unattended.
        self.control_panel.set_auto_enabled(True)

    def _disconnect(self):
        if self.record_file:
            self._stop_recording()
        if self.reader:
            self.reader.stop()
            self.reader = None
        self.connection_panel.set_connected_ui(False)
        self._log("# Disconnected.")

    def _on_error(self, msg: str):
        self._log(f"# ERROR: {msg}")
        self._disconnect()

    def _send(self, text: str):
        if self.reader:
            self.reader.send(text)

    def _on_hv_toggled(self, checked: bool):
        self._send(f"HVEN:{1 if checked else 0}")

    # ── AGC ──────────────────────────────────────────────────────────────

    def _on_auto_toggled(self, checked: bool):
        self.auto_enabled = checked
        if checked:
            hw = self.hardware_panel
            self.auto_tuner.sync(
                led=hw.sl_led1.slider.value(),
                gain_tia=hw.sl_gain.slider.value(),
                ads_gain=hw.sl_adsgain.slider.value(),
            )

    def _on_boost_opt_toggled(self, checked: bool):
        self.boost_opt_enabled = checked
        if checked:
            self.boost_opt.sync(self.hardware_panel.sl_boost.slider.value())

    def _run_agc(self):
        if self.reader is None:
            return

        ibi_cv = self._latest_result.ibi_cv if self._latest_result else float("inf")
        amp_cv = self._latest_result.peak_amp_cv if self._latest_result else float("inf")

        if self.boost_opt_enabled:
            new_boost, msg = self.boost_opt.step_quality(
                self.current_inband_snr, self.bpm_valid, self.current_motion,
                ibi_cv=ibi_cv, amp_cv=amp_cv)
            if new_boost is not None:
                self._send(f"BOOST:{new_boost}")
                self.hardware_panel.sl_boost.set_silent(new_boost)
            if msg:
                self.control_panel.set_boost_status(msg)

        if not self.auto_enabled or len(self.buf_ppg) < 32:
            return

        raw = np.fromiter(self.buf_ppg, dtype=float)[-int(1.5 * NOMINAL_FS):]
        cmds, reason = self.auto_tuner.step(
            raw, valid=self.bpm_valid, ibi_cv=ibi_cv, amp_cv=amp_cv)
        hw = self.hardware_panel
        for cmd in cmds:
            self._send(cmd)
            name, _, val = cmd.partition(":")
            val = int(val)
            if name == "GAIN":
                hw.sl_gain.set_silent(val)
            elif name == "ADSGAIN":
                hw.sl_adsgain.set_silent(val)
            elif name == "LED1":
                hw.sl_led1.set_silent(val)
                hw.sl_led2.set_silent(val)
        if reason:
            self.control_panel.set_auto_status(reason)

    # ── Enregistrement CSV ───────────────────────────────────────────────

    def _start_recording(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_path = RECORDINGS_DIR / f"ppg_raw_{stamp}.csv"
        self.record_file = self.record_path.open("w", newline="")
        self.record_writer = csv.writer(self.record_file)
        self.record_writer.writerow([
            "unix_time_s", "sample_index", "led1_adc", "led2_adc",
            "accel_x", "accel_y", "accel_z",
            "gain_tia", "bias_ht", "dac_offset", "adsgain",
            "led1_pwm", "led2_pwm", "hv_enabled",
            "bpm", "fft_quality", "motion_level",
        ])
        self.recording_panel.set_recording_ui(str(self.record_path))
        self._log(f"# Recording: {self.record_path}")

    def _stop_recording(self):
        if self.record_file:
            path = self.record_path
            self.record_file.close()
            self.record_file = None
            self.record_writer = None
            self.record_path = None
            self._log(f"# Recording saved: {path}")
            self.recording_panel.set_saved_label(str(path))
        self.recording_panel.set_recording_ui(None)

    # ── Traitement des échantillons ──────────────────────────────────────

    def _on_sample(self, ppg: int, _ppg2: int, ax: int, ay: int, az: int, ts: float):
        self.t.append(self.sample_index)
        self.sample_index += 1
        self.buf_ppg.append(ppg)
        self.buf_ax.append(ax)
        self.buf_ay.append(ay)
        self.buf_az.append(az)
        self.hr_proc.push(ppg, ppg, ax, ay, az, ts=ts)

        if self.record_writer:
            hw = self.hardware_panel
            self.record_writer.writerow([
                f"{ts:.6f}", self.sample_index - 1, ppg, ppg,
                ax, ay, az,
                hw.sl_gain.slider.value(),
                hw.sl_boost.slider.value(),
                hw.sl_dac.slider.value(),
                hw.sl_adsgain.slider.value(),
                hw.sl_led1.slider.value(),
                hw.sl_led2.slider.value(),
                int(hw.is_hv_enabled()),
                f"{self.current_bpm:.2f}" if self.bpm_valid else "",
                f"{self.current_snr:.3f}" if self.current_snr > 0 else "",
                f"{self.current_motion:.3f}",
            ])

    def _refresh_plots(self):
        while not self.data_queue.empty():
            try:
                self._on_sample(*self.data_queue.get_nowait())
            except queue.Empty:
                break

        if not self.t:
            return

        self._plot_tick += 1
        ppg = np.fromiter(self.buf_ppg, dtype=float)
        x = np.fromiter(self.t, dtype=float)

        self.signal_panel.update_raw_plot(
            x, ppg, self.control_panel.is_center_display_enabled())

        if self._plot_tick % 4 == 0 or self._latest_result is None:
            self._latest_result = self.hr_proc.compute()
        result = self._latest_result

        if result is not None:
            self.current_bpm = result.bpm
            self.bpm_valid = result.bpm_valid
            self.current_snr = result.snr
            self.current_ac_amp = result.ac_amplitude
            self.current_motion = result.motion_level
            self.current_inband_snr = result.inband_snr
            self.beat_indices = result.beat_indices

            self.signal_panel.update_filtered_plot(x, result.filtered, self.beat_indices)

            if result.bpm_valid:
                self.bpm_hist_x.append(self.sample_index)
                self.bpm_hist_raw.append(result.bpm_raw)
                self.bpm_hist_ml.append(
                    result.bpm_ml if result.bpm_ml is not None else np.nan)
                self.bpm_hist_instant.append(result.bpm)

                # Watch-style smoothed line: trailing mean of the instant
                # BPM (result.bpm -- raw or ML-corrected, whichever is
                # actually driving the display right now) over the last
                # WATCH_AVG_SECONDS, using the sample-index x-values already
                # kept in bpm_hist_x so gaps from invalid stretches don't
                # skew the window.
                x_hist = np.fromiter(self.bpm_hist_x, dtype=float)
                instant_hist = np.fromiter(self.bpm_hist_instant, dtype=float)
                window = x_hist >= (self.sample_index - WATCH_AVG_SAMPLES)
                self.bpm_hist_smooth.append(float(np.mean(instant_hist[window])))

                self.signal_panel.update_bpm_trend(
                    x_hist,
                    np.fromiter(self.bpm_hist_raw, dtype=float),
                    np.fromiter(self.bpm_hist_ml, dtype=float),
                    self.control_panel.is_ml_compare_enabled(),
                    np.fromiter(self.bpm_hist_smooth, dtype=float),
                )

        self.signal_panel.update_accel_plot(
            x,
            np.fromiter(self.buf_ax, dtype=float),
            np.fromiter(self.buf_ay, dtype=float),
            np.fromiter(self.buf_az, dtype=float),
        )

        self.signal_panel.update_bpm(self.current_bpm, self.bpm_valid)
        self.signal_panel.update_snr(self.current_snr, self.current_inband_snr)

        gain_idx = self.hardware_panel.sl_adsgain.slider.value()
        ppg_counts = float(ppg[-1])
        pin_counts = ADC_FULL_SCALE - ppg_counts
        pin_mv = adc_counts_to_mv(pin_counts, gain_idx)
        self.signal_panel.update_adc_voltage(pin_mv, ppg_counts)

        if self.current_ac_amp > 0:
            ac_mv = adc_counts_to_mv(self.current_ac_amp, gain_idx)
            self.signal_panel.update_ac_amplitude(self.current_ac_amp, ac_mv)
        else:
            self.signal_panel.update_ac_amplitude(0.0, 0.0)

        self.signal_panel.update_motion(self.current_motion)

    # ── Divers ───────────────────────────────────────────────────────────

    def _log(self, text: str):
        self.log_panel.append(text)

    def closeEvent(self, event):
        if self.record_file:
            self._stop_recording()
        if self.reader:
            self.reader.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
