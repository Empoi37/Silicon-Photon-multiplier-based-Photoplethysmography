#!/usr/bin/env python3
"""Real-time PPG GUI for the SiPM sensor prototype.

Run:  python gui_ppg.py
"""

from __future__ import annotations

import csv
import queue
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PySide6 import QtCore, QtGui, QtWidgets

from ppg_agc import AutoTuner, BoostOptimizer
from ppg_pipeline import PpgHrProcessor, PpgMode

BAUDRATE = 115200
PLOT_SECONDS = 8
NOMINAL_FS = 250
BUFFER_LEN = PLOT_SECONDS * NOMINAL_FS
RECORDINGS_DIR = Path(__file__).parent / "recordings"


def _ensure_qt_plugins_visible() -> None:
    """macOS: unhide pip-installed Qt plugins so PySide6 can load them."""
    if sys.platform != "darwin":
        return
    spec = find_spec("PySide6")
    if spec and spec.origin:
        plugins = Path(spec.origin).parent / "Qt" / "plugins"
        if plugins.is_dir():
            subprocess.run(["chflags", "-R", "nohidden", str(plugins)], check=False)


_ensure_qt_plugins_visible()


class SerialReader(QtCore.QThread):
    message = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, port: str, data_queue: queue.Queue, baudrate: int = BAUDRATE):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
        self._running = False
        self._ser = None

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=1)
        except serial.SerialException as exc:
            self.error.emit(f"Cannot open {self.port}: {exc}")
            return

        self._running = True
        while self._running:
            try:
                raw = self._ser.readline()
            except serial.SerialException as exc:
                self.error.emit(f"Serial read error: {exc}")
                break
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                self.message.emit(line)
                continue

            parts = line.split(",")
            if len(parts) != 5:
                continue
            try:
                led1, led2, ax, ay, az = (int(p) for p in parts)
            except ValueError:
                continue
            self.data_queue.put((led1, led2, ax, ay, az, time.time()))

        if self._ser and self._ser.is_open:
            self._ser.close()

    def send(self, text: str):
        if self._ser and self._ser.is_open:
            try:
                self._ser.write((text + "\n").encode("utf-8"))
            except serial.SerialException as exc:
                self.error.emit(f"Serial write error: {exc}")

    def stop(self):
        self._running = False
        self.wait(2000)


class CommandSlider(QtWidgets.QWidget):
    def __init__(self, label: str, cmd: str, vmin: int, vmax: int,
                 default: int, on_command):
        super().__init__()
        self.cmd = cmd
        self.on_command = on_command

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel(label)
        title.setMinimumWidth(200)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(vmin, vmax)
        self.slider.setValue(default)

        self.value_lbl = QtWidgets.QLabel(str(default))
        self.value_lbl.setMinimumWidth(40)
        self.value_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        layout.addWidget(title)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_lbl)
        self.slider.valueChanged.connect(self._changed)

    def _changed(self, value: int):
        self.value_lbl.setText(str(value))
        self.on_command(f"{self.cmd}:{value}")


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

        self.record_file = None
        self.record_writer = None
        self.record_path = None
        self.data_queue: queue.Queue = queue.Queue()

        self.auto_tuner = AutoTuner()
        self.auto_enabled = False
        self.boost_opt = BoostOptimizer()
        self.boost_opt_enabled = False

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh_plots)
        self.timer.start(50)

        self.agc_timer = QtCore.QTimer(self)
        self.agc_timer.timeout.connect(self._run_agc)
        self.agc_timer.start(500)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        pg.setConfigOptions(antialias=True)
        graphs = QtWidgets.QVBoxLayout()

        self.plot_raw = pg.PlotWidget(title="Raw PPG (ADC counts)")
        self.plot_raw.setLabel("bottom", "Sample")
        self.plot_raw.setLabel("left", "ADC")
        self.plot_raw.showGrid(x=True, y=True, alpha=0.3)
        self.curve_raw = self.plot_raw.plot(pen=pg.mkPen("#27ae60", width=2))

        self.plot_proc = pg.PlotWidget(title="Filtered PPG + heartbeats")
        self.plot_proc.setLabel("bottom", "Sample")
        self.plot_proc.setLabel("left", "Normalized")
        self.plot_proc.showGrid(x=True, y=True, alpha=0.3)
        self.curve_filtered = self.plot_proc.plot(pen=pg.mkPen("#e67e22", width=2))
        self.scatter_beats = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen(None), brush=pg.mkBrush("#e74c3c"), symbol="o")
        self.plot_proc.addItem(self.scatter_beats)

        self.plot_acc = pg.PlotWidget(title="Accelerometer (LIS2DH12)")
        self.plot_acc.addLegend()
        self.plot_acc.setLabel("bottom", "Sample")
        self.plot_acc.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ax = self.plot_acc.plot(pen=pg.mkPen("#3498db", width=1), name="X")
        self.curve_ay = self.plot_acc.plot(pen=pg.mkPen("#f1c40f", width=1), name="Y")
        self.curve_az = self.plot_acc.plot(pen=pg.mkPen("#9b59b6", width=1), name="Z")

        graphs.addWidget(self.plot_raw, 2)
        graphs.addWidget(self.plot_proc, 2)
        graphs.addWidget(self.plot_acc, 1)

        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(self._build_connection_box())
        controls.addWidget(self._build_auto_box())
        controls.addWidget(self._build_recording_box())
        controls.addWidget(self._build_hr_box())
        controls.addWidget(self._build_hardware_box())
        controls.addWidget(self._build_hv_box())
        controls.addWidget(self._build_log_box(), 1)

        right = QtWidgets.QWidget()
        right.setLayout(controls)
        right.setFixedWidth(440)
        root.addLayout(graphs, 1)
        root.addWidget(right)

    def _build_connection_box(self):
        box = QtWidgets.QGroupBox("Connection")
        lay = QtWidgets.QHBoxLayout(box)
        self.port_combo = QtWidgets.QComboBox()
        self._refresh_ports()
        self.refresh_btn = QtWidgets.QPushButton("Scan")
        self.refresh_btn.clicked.connect(self._refresh_ports)
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setCheckable(True)
        self.connect_btn.clicked.connect(self._toggle_connection)
        lay.addWidget(self.port_combo, 1)
        lay.addWidget(self.refresh_btn)
        lay.addWidget(self.connect_btn)
        return box

    def _build_auto_box(self):
        box = QtWidgets.QGroupBox("Automatic control")
        lay = QtWidgets.QVBoxLayout(box)

        self.auto_btn = QtWidgets.QPushButton("Enable auto-tuning (AGC)")
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(self._toggle_auto)

        self.motion_btn = QtWidgets.QCheckBox("Motion cancellation (accelerometer)")
        self.motion_btn.setChecked(True)
        self.motion_btn.toggled.connect(
            lambda c: setattr(self.hr_proc, "use_motion_cancel", c))

        self.dc_servo_btn = QtWidgets.QCheckBox("Firmware DC servo (limited DAC authority)")
        self.dc_servo_btn.setChecked(False)
        self.dc_servo_btn.toggled.connect(self._toggle_dc_servo)

        self.center_display_btn = QtWidgets.QCheckBox("Center raw plot (remove DC for display)")
        self.center_display_btn.setChecked(True)

        self.boost_opt_btn = QtWidgets.QCheckBox("Auto-optimize BOOST (SiPM overvoltage)")
        self.boost_opt_btn.setChecked(False)
        self.boost_opt_btn.toggled.connect(self._toggle_boost_opt)

        self.auto_status = QtWidgets.QLabel("Auto-tuning: off")
        self.auto_status.setWordWrap(True)
        self.auto_status.setStyleSheet("color:#777;font-size:11px;")
        self.boost_status = QtWidgets.QLabel("BOOST optimizer: off")
        self.boost_status.setWordWrap(True)
        self.boost_status.setStyleSheet("color:#777;font-size:11px;")

        for w in (self.auto_btn, self.motion_btn, self.dc_servo_btn,
                  self.center_display_btn, self.boost_opt_btn,
                  self.auto_status, self.boost_status):
            lay.addWidget(w)
        return box

    def _build_recording_box(self):
        box = QtWidgets.QGroupBox("Raw recording")
        lay = QtWidgets.QVBoxLayout(box)
        self.record_btn = QtWidgets.QPushButton("Start CSV recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._toggle_recording)
        self.record_label = QtWidgets.QLabel("No recording")
        self.record_label.setWordWrap(True)
        self.record_label.setStyleSheet("color:#777;font-size:11px;")
        lay.addWidget(self.record_btn)
        lay.addWidget(self.record_label)
        return box

    def _build_hr_box(self):
        box = QtWidgets.QGroupBox("Heart rate")
        lay = QtWidgets.QVBoxLayout(box)

        self.bpm_label = QtWidgets.QLabel("--")
        self.bpm_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(36)
        font.setBold(True)
        self.bpm_label.setFont(font)
        self.bpm_label.setStyleSheet("color:#c0392b;")

        self.bpm_unit = QtWidgets.QLabel("BPM")
        self.bpm_unit.setAlignment(QtCore.Qt.AlignCenter)
        self.snr_label = QtWidgets.QLabel("SNR: --")
        self.snr_label.setAlignment(QtCore.Qt.AlignCenter)
        self.snr_label.setStyleSheet("color:#666;font-size:12px;")
        self.ac_label = QtWidgets.QLabel("AC amplitude: --")
        self.ac_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ac_label.setStyleSheet("color:#666;font-size:12px;")
        self.motion_label = QtWidgets.QLabel("Motion: --")
        self.motion_label.setAlignment(QtCore.Qt.AlignCenter)
        self.motion_label.setStyleSheet("color:#666;font-size:12px;")

        for w in (self.bpm_label, self.bpm_unit, self.snr_label,
                  self.ac_label, self.motion_label):
            lay.addWidget(w)
        return box

    def _build_hardware_box(self):
        box = QtWidgets.QGroupBox("Hardware controls")
        lay = QtWidgets.QVBoxLayout(box)

        self.sl_gain = CommandSlider("TIA gain (MCP4531)", "GAIN", 0, 255, 210, self._send)
        self.sl_boost = CommandSlider("SiPM bias (MCP4018)", "BOOST", 0, 127, 64, self._send)
        self.sl_dac = CommandSlider("DC offset (DAC)", "DAC", 0, 255, 128, self._send)
        self.sl_adsgain = CommandSlider("ADC gain (ADSGAIN 0-5)", "ADSGAIN", 0, 5, 5, self._send)
        self.sl_led1 = CommandSlider("LED 1 brightness", "LED1", 0, 255, 48, self._send)
        self.sl_led2 = CommandSlider("LED 2 brightness", "LED2", 0, 255, 48, self._send)

        for w in (self.sl_gain, self.sl_boost, self.sl_dac, self.sl_adsgain,
                  self.sl_led1, self.sl_led2):
            lay.addWidget(w)
        return box

    def _build_hv_box(self):
        box = QtWidgets.QGroupBox("High voltage (SiPM ~27 V)")
        lay = QtWidgets.QVBoxLayout(box)
        self.hv_btn = QtWidgets.QPushButton("ENABLE HIGH VOLTAGE")
        self.hv_btn.setCheckable(True)
        self.hv_btn.setMinimumHeight(48)
        self._style_hv_button(False)
        self.hv_btn.clicked.connect(self._toggle_hv)
        lay.addWidget(self.hv_btn)
        return box

    def _build_log_box(self):
        box = QtWidgets.QGroupBox("Log")
        lay = QtWidgets.QVBoxLayout(box)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        lay.addWidget(self.log)
        return box

    def _refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        if current in ports:
            self.port_combo.setCurrentText(current)

    def _toggle_connection(self, checked: bool):
        if checked:
            port = self.port_combo.currentText()
            if not port:
                self._log("# No port selected.")
                self.connect_btn.setChecked(False)
                return

            self.hr_proc.reset()
            for buf in (self.t, self.buf_ppg, self.buf_ax, self.buf_ay, self.buf_az):
                buf.clear()
            self.sample_index = 0
            self.beat_indices = np.array([], dtype=int)
            self.current_bpm = 0.0
            self.bpm_valid = False
            self._latest_result = None
            self._plot_tick = 0
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break

            self.reader = SerialReader(port, self.data_queue)
            self.reader.message.connect(self._log)
            self.reader.error.connect(self._on_error)
            self.reader.start()
            self.connect_btn.setText("Disconnect")
            self.port_combo.setEnabled(False)
            self._log(f"# Connected to {port}")

            self._send(f"ADSGAIN:{self.sl_adsgain.slider.value()}")
            self._send(f"GAIN:{self.sl_gain.slider.value()}")
            self._send(f"DCSERVO:{1 if self.dc_servo_btn.isChecked() else 0}")
            self._send(f"LED1:{self.sl_led1.slider.value()}")
            self._send(f"LED2:{self.sl_led2.slider.value()}")
        else:
            self._disconnect()

    def _disconnect(self):
        if self.record_file:
            self._stop_recording()
        if self.reader:
            self.reader.stop()
            self.reader = None
        self.connect_btn.setChecked(False)
        self.connect_btn.setText("Connect")
        self.port_combo.setEnabled(True)
        self._log("# Disconnected.")

    def _on_error(self, msg: str):
        self._log(f"# ERROR: {msg}")
        self._disconnect()

    def _send(self, text: str):
        if self.reader:
            self.reader.send(text)

    def _toggle_auto(self, checked: bool):
        self.auto_enabled = checked
        if checked:
            self.auto_tuner.sync(
                led=self.sl_led1.slider.value(),
                gain_tia=self.sl_gain.slider.value(),
                ads_gain=self.sl_adsgain.slider.value(),
            )
            self.auto_btn.setText("Disable auto-tuning")
            self.auto_status.setText("Auto-tuning: active")
        else:
            self.auto_btn.setText("Enable auto-tuning (AGC)")
            self.auto_status.setText("Auto-tuning: off")

    def _toggle_dc_servo(self, checked: bool):
        if self.reader:
            self._send(f"DCSERVO:{1 if checked else 0}")
        self._log(f"# DC servo: {'ON' if checked else 'OFF'}")

    def _toggle_boost_opt(self, checked: bool):
        self.boost_opt_enabled = checked
        if checked:
            self.boost_opt.sync(self.sl_boost.slider.value())
            self.boost_status.setText("BOOST optimizer: exploring...")
        else:
            self.boost_status.setText("BOOST optimizer: off")

    def _set_slider_silent(self, slider: CommandSlider, value: int):
        slider.slider.blockSignals(True)
        slider.slider.setValue(int(value))
        slider.value_lbl.setText(str(int(value)))
        slider.slider.blockSignals(False)

    def _run_agc(self):
        if self.reader is None:
            return

        if self.boost_opt_enabled:
            new_boost, msg = self.boost_opt.step_quality(
                self.current_inband_snr, self.bpm_valid, self.current_motion)
            if new_boost is not None:
                self._send(f"BOOST:{new_boost}")
                self._set_slider_silent(self.sl_boost, new_boost)
            if msg:
                self.boost_status.setText(f"BOOST optimizer: {msg}")

        if not self.auto_enabled or len(self.buf_ppg) < 32:
            return

        raw = np.fromiter(self.buf_ppg, dtype=float)[-int(1.5 * NOMINAL_FS):]
        cmds, reason = self.auto_tuner.step(raw)
        for cmd in cmds:
            self._send(cmd)
            name, _, val = cmd.partition(":")
            val = int(val)
            if name == "GAIN":
                self._set_slider_silent(self.sl_gain, val)
            elif name == "ADSGAIN":
                self._set_slider_silent(self.sl_adsgain, val)
            elif name == "LED1":
                self._set_slider_silent(self.sl_led1, val)
                self._set_slider_silent(self.sl_led2, val)
        if reason:
            self.auto_status.setText(f"Auto-tuning: {reason}")

    def _toggle_recording(self, checked: bool):
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

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
        self.record_btn.setText("Stop CSV recording")
        self.record_label.setText(str(self.record_path))
        self._log(f"# Recording: {self.record_path}")

    def _stop_recording(self):
        if self.record_file:
            path = self.record_path
            self.record_file.close()
            self.record_file = None
            self.record_writer = None
            self.record_path = None
            self._log(f"# Recording saved: {path}")
            self.record_label.setText(f"Saved: {path}")
        self.record_btn.setChecked(False)
        self.record_btn.setText("Start CSV recording")

    def _toggle_hv(self, checked: bool):
        self._style_hv_button(checked)
        self._send(f"HVEN:{1 if checked else 0}")

    def _style_hv_button(self, on: bool):
        if on:
            self.hv_btn.setText("DISABLE HIGH VOLTAGE (ON)")
            self.hv_btn.setStyleSheet("background-color:#c0392b;color:white;font-weight:bold;")
        else:
            self.hv_btn.setText("ENABLE HIGH VOLTAGE (OFF)")
            self.hv_btn.setStyleSheet("background-color:#27ae60;color:white;font-weight:bold;")

    def _on_sample(self, ppg: int, _ppg2: int, ax: int, ay: int, az: int, ts: float):
        self.t.append(self.sample_index)
        self.sample_index += 1
        self.buf_ppg.append(ppg)
        self.buf_ax.append(ax)
        self.buf_ay.append(ay)
        self.buf_az.append(az)
        self.hr_proc.push(ppg, ppg, ax, ay, az, ts=ts)

        if self.record_writer:
            self.record_writer.writerow([
                f"{ts:.6f}", self.sample_index - 1, ppg, ppg,
                ax, ay, az,
                self.sl_gain.slider.value(),
                self.sl_boost.slider.value(),
                self.sl_dac.slider.value(),
                self.sl_adsgain.slider.value(),
                self.sl_led1.slider.value(),
                self.sl_led2.slider.value(),
                int(self.hv_btn.isChecked()),
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

        raw_view = ppg - ppg.mean() if self.center_display_btn.isChecked() else ppg
        self.curve_raw.setData(x, raw_view)

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

            filt = result.filtered
            n_show = min(len(x), len(filt))
            x_f = x[-n_show:]
            filt_view = filt[-n_show:]
            self.curve_filtered.setData(x_f, filt_view)

            if len(self.beat_indices) > 0 and n_show > 0:
                offset = len(filt) - n_show
                visible = self.beat_indices[
                    (self.beat_indices >= offset) & (self.beat_indices < len(filt))]
                if len(visible) > 0:
                    local = visible - offset
                    self.scatter_beats.setData(x=x_f[local], y=filt_view[local])
                else:
                    self.scatter_beats.setData([], [])
            else:
                self.scatter_beats.setData([], [])

        self.curve_ax.setData(x, np.fromiter(self.buf_ax, dtype=float))
        self.curve_ay.setData(x, np.fromiter(self.buf_ay, dtype=float))
        self.curve_az.setData(x, np.fromiter(self.buf_az, dtype=float))

        if self.bpm_valid and self.current_bpm > 0:
            self.bpm_label.setText(f"{self.current_bpm:.0f}")
            self.bpm_label.setStyleSheet("color:#c0392b;")
        else:
            self.bpm_label.setText("--")
            self.bpm_label.setStyleSheet("color:#95a5a6;")

        if self.current_snr > 0:
            self.snr_label.setText(
                f"FFT quality: {self.current_snr:.2f}  |  "
                f"In-band SNR: {self.current_inband_snr:.0f}")
        else:
            self.snr_label.setText("SNR: --")

        if self.current_ac_amp > 0:
            self.ac_label.setText(f"AC amplitude: {self.current_ac_amp:.1f} counts")
        else:
            self.ac_label.setText("AC amplitude: --")

        m = self.current_motion
        if m > 0.6:
            self.motion_label.setText(f"Motion: HIGH ({m:.2f})")
            self.motion_label.setStyleSheet("color:#c0392b;font-size:12px;")
        elif m > 0.2:
            self.motion_label.setText(f"Motion: moderate ({m:.2f})")
            self.motion_label.setStyleSheet("color:#e67e22;font-size:12px;")
        else:
            self.motion_label.setText(f"Motion: low ({m:.2f})")
            self.motion_label.setStyleSheet("color:#27ae60;font-size:12px;")

    def _log(self, text: str):
        self.log.appendPlainText(text)

    def closeEvent(self, event):
        if self.record_file:
            self._stop_recording()
        if self.reader:
            self._send("HVEN:0")
            self.reader.stop()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
