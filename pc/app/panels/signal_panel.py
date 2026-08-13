"""
app/panels/signal_panel.py
─────────────────────────────
Affichage temps réel : graphiques (brut, filtré + battements, accéléromètre)
et les lectures numériques (BPM, tension AIN0, SNR, amplitude AC, mouvement).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets


class SignalPanel(QtWidgets.QWidget):
    """
    Regroupe les 3 graphiques temps réel (colonne gauche de l'app) et le
    bloc de lecture "Heart rate" (BPM, SNR, etc.) affiché dans la colonne
    de droite. Exposé comme deux widgets séparés via `.plots_widget` et
    `.readout_widget` pour que MainWindow les place où il veut.
    """

    def __init__(self):
        super().__init__()
        pg.setConfigOptions(antialias=True)

        self.plots_widget = self._build_plots()
        self.readout_widget = self._build_readout()

    # ── Graphiques ────────────────────────────────────────────────────────

    def _build_plots(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        graphs = QtWidgets.QVBoxLayout(container)
        graphs.setContentsMargins(0, 0, 0, 0)

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

        self.plot_bpm = pg.PlotWidget(title="Heart rate trend (BPM)")
        self.plot_bpm.addLegend()
        self.plot_bpm.setLabel("bottom", "Sample")
        self.plot_bpm.setLabel("left", "BPM")
        self.plot_bpm.showGrid(x=True, y=True, alpha=0.3)
        self.curve_bpm_raw = self.plot_bpm.plot(
            pen=pg.mkPen("#7f8c8d", width=2), name="Raw (no ML)")
        self.curve_bpm_ml = self.plot_bpm.plot(
            pen=pg.mkPen("#27ae60", width=2), name="ML corrected")
        self.curve_bpm_smooth = self.plot_bpm.plot(
            pen=pg.mkPen("#2980b9", width=2, style=QtCore.Qt.DashLine),
            name="Smoothed (~5s, watch-style)")

        graphs.addWidget(self.plot_raw, 2)
        graphs.addWidget(self.plot_proc, 2)
        graphs.addWidget(self.plot_bpm, 2)
        graphs.addWidget(self.plot_acc, 1)
        return container

    # ── Lecture BPM / qualité ────────────────────────────────────────────

    def _build_readout(self) -> QtWidgets.QWidget:
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

        self.adc_v_label = QtWidgets.QLabel("AIN0 voltage: --")
        self.adc_v_label.setAlignment(QtCore.Qt.AlignCenter)
        self.adc_v_label.setStyleSheet("color:#2c3e50;font-size:13px;font-weight:bold;")

        self.snr_label = QtWidgets.QLabel("SNR: --")
        self.snr_label.setAlignment(QtCore.Qt.AlignCenter)
        self.snr_label.setStyleSheet("color:#666;font-size:12px;")

        self.ac_label = QtWidgets.QLabel("AC amplitude: --")
        self.ac_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ac_label.setStyleSheet("color:#666;font-size:12px;")

        self.motion_label = QtWidgets.QLabel("Motion: --")
        self.motion_label.setAlignment(QtCore.Qt.AlignCenter)
        self.motion_label.setStyleSheet("color:#666;font-size:12px;")

        for w in (self.bpm_label, self.bpm_unit, self.adc_v_label, self.snr_label,
                  self.ac_label, self.motion_label):
            lay.addWidget(w)
        return box

    # ── Mises à jour ─────────────────────────────────────────────────────

    def update_raw_plot(self, x: np.ndarray, ppg: np.ndarray, center: bool):
        raw_view = ppg - ppg.mean() if center else ppg
        self.curve_raw.setData(x, raw_view)

    def update_accel_plot(self, x: np.ndarray, ax: np.ndarray, ay: np.ndarray, az: np.ndarray):
        self.curve_ax.setData(x, ax)
        self.curve_ay.setData(x, ay)
        self.curve_az.setData(x, az)

    def update_filtered_plot(self, x: np.ndarray, filt: np.ndarray, beat_indices: np.ndarray):
        n_show = min(len(x), len(filt))
        x_f = x[-n_show:]
        filt_view = filt[-n_show:]
        self.curve_filtered.setData(x_f, filt_view)

        if len(beat_indices) > 0 and n_show > 0:
            offset = len(filt) - n_show
            visible = beat_indices[(beat_indices >= offset) & (beat_indices < len(filt))]
            if len(visible) > 0:
                local = visible - offset
                self.scatter_beats.setData(x=x_f[local], y=filt_view[local])
                return
        self.scatter_beats.setData([], [])

    def update_bpm_trend(self, x: np.ndarray, raw: np.ndarray, ml: np.ndarray, show_ml: bool,
                         smooth: np.ndarray | None = None):
        self.curve_bpm_raw.setData(x, raw)
        if smooth is not None:
            self.curve_bpm_smooth.setData(x, smooth)
        if show_ml:
            self.curve_bpm_ml.setData(x, ml)
        else:
            self.curve_bpm_ml.setData([], [])

    def update_bpm(self, bpm: float, valid: bool):
        if valid and bpm > 0:
            self.bpm_label.setText(f"{bpm:.0f}")
            self.bpm_label.setStyleSheet("color:#c0392b;")
        else:
            self.bpm_label.setText("--")
            self.bpm_label.setStyleSheet("color:#95a5a6;")

    def update_snr(self, fft_quality: float, inband_snr: float):
        if fft_quality > 0:
            self.snr_label.setText(
                f"FFT quality: {fft_quality:.2f}  |  In-band SNR: {inband_snr:.0f}")
        else:
            self.snr_label.setText("SNR: --")

    def update_adc_voltage(self, pin_mv: float, ppg_counts: float):
        self.adc_v_label.setText(f"AIN0 pin: {pin_mv:.3f} mV  |  PPG: {ppg_counts:.0f} counts")

    def update_ac_amplitude(self, ac_amp: float, ac_mv: float):
        if ac_amp > 0:
            self.ac_label.setText(f"AC amplitude: {ac_amp:.1f} counts ({ac_mv:.3f} mV pk)")
        else:
            self.ac_label.setText("AC amplitude: --")

    def update_motion(self, motion: float):
        if motion > 0.6:
            self.motion_label.setText(f"Motion: HIGH ({motion:.2f})")
            self.motion_label.setStyleSheet("color:#c0392b;font-size:12px;")
        elif motion > 0.2:
            self.motion_label.setText(f"Motion: moderate ({motion:.2f})")
            self.motion_label.setStyleSheet("color:#e67e22;font-size:12px;")
        else:
            self.motion_label.setText(f"Motion: low ({motion:.2f})")
            self.motion_label.setStyleSheet("color:#27ae60;font-size:12px;")
