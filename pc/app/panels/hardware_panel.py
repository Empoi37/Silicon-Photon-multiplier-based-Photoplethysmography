"""
app/panels/hardware_panel.py
──────────────────────────────
Contrôles matériels bas niveau : gain TIA, bias SiPM (BOOST), LEDs, gain ADC,
et le bouton haute tension (HV_EN).

Ce panel n'envoie rien lui-même à l'ESP32 — il émet des signaux Qt que
MainWindow relaie vers le SerialReader. Ça garde le panel testable et
indépendant de la connexion série.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from app.widgets import CommandSlider


class HardwarePanel(QtWidgets.QWidget):
    """
    Signaux
    ───────
    commandIssued(str)   Émis avec une commande brute ("GAIN:64") à envoyer
    hvToggled(bool)       Émis quand l'utilisateur bascule la haute tension
    """

    commandIssued = QtCore.Signal(str)
    hvToggled = QtCore.Signal(bool)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_hardware_box())
        layout.addWidget(self._build_hv_box())

    def _build_hardware_box(self):
        box = QtWidgets.QGroupBox("Hardware controls")
        lay = QtWidgets.QVBoxLayout(box)

        # ── Preset "vie quotidienne" verrouillé le 2026-07-27 ──────────────
        # Trouvé par caractérisation systématique (tools/sweep_preset.py)
        # puis validé visuellement : signal propre, FFT quality=12.18,
        # in-band SNR=699, rythme régulier ~100 BPM au repos.
        self.sl_gain = CommandSlider(
            "TIA gain (MCP4531)", "GAIN", 0, 128, 96, self.commandIssued.emit)
        self.sl_boost = CommandSlider(
            "SiPM bias (MCP4018, ~31.5V cathode)", "BOOST", 0, 127, 18,
            self.commandIssued.emit)
        self.sl_dac = CommandSlider(
            "DC offset (DAC, disabled)", "DAC", 0, 255, 0, self.commandIssued.emit)
        self.sl_dac.setEnabled(False)
        self.sl_adsgain = CommandSlider(
            "ADC gain (ADSGAIN 0-5)", "ADSGAIN", 0, 5, 0, self.commandIssued.emit)
        self.sl_led1 = CommandSlider(
            "LED 1 brightness", "LED1", 0, 255, 144, self.commandIssued.emit)
        self.sl_led2 = CommandSlider(
            "LED 2 brightness", "LED2", 0, 255, 144, self.commandIssued.emit)

        for w in (self.sl_gain, self.sl_boost, self.sl_dac, self.sl_adsgain,
                  self.sl_led1, self.sl_led2):
            lay.addWidget(w)
        return box

    def _build_hv_box(self):
        box = QtWidgets.QGroupBox("High voltage (SiPM ~27 V)")
        lay = QtWidgets.QVBoxLayout(box)
        self.hv_btn = QtWidgets.QPushButton("ENABLE HIGH VOLTAGE")
        self.hv_btn.setCheckable(True)
        self.hv_btn.setChecked(True)
        self.hv_btn.setMinimumHeight(48)
        self._style_hv_button(True)
        self.hv_btn.clicked.connect(self._on_hv_toggle)
        lay.addWidget(self.hv_btn)
        return box

    def _on_hv_toggle(self, checked: bool):
        self._style_hv_button(checked)
        self.hvToggled.emit(checked)

    def _style_hv_button(self, on: bool):
        if on:
            self.hv_btn.setText("DISABLE HIGH VOLTAGE (ON)")
            self.hv_btn.setStyleSheet(
                "background-color:#c0392b;color:white;font-weight:bold;")
        else:
            self.hv_btn.setText("ENABLE HIGH VOLTAGE (OFF)")
            self.hv_btn.setStyleSheet(
                "background-color:#27ae60;color:white;font-weight:bold;")

    def is_hv_enabled(self) -> bool:
        return self.hv_btn.isChecked()