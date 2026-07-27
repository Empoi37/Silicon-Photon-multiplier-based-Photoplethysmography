"""
app/panels/control_panel.py
─────────────────────────────
Panel de contrôle automatique : AGC (auto-tuning), annulation de mouvement,
et optimiseur de BOOST (rétroaction logicielle décrite dans le rapport).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class ControlPanel(QtWidgets.QGroupBox):
    """
    Signaux
    ───────
    autoToggled(bool)           Active/désactive l'AGC (gain/LED automatiques)
    motionToggled(bool)          Active/désactive l'annulation de mouvement
    centerDisplayToggled(bool)   Centre l'affichage brut (retire la DC visuellement)
    boostOptToggled(bool)        Active/désactive l'optimiseur de BOOST
    """

    autoToggled = QtCore.Signal(bool)
    motionToggled = QtCore.Signal(bool)
    centerDisplayToggled = QtCore.Signal(bool)
    boostOptToggled = QtCore.Signal(bool)

    def __init__(self):
        super().__init__("Automatic control")
        lay = QtWidgets.QVBoxLayout(self)

        self.auto_btn = QtWidgets.QPushButton("Enable auto-tuning (AGC)")
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(self._on_auto_toggle)

        self.motion_btn = QtWidgets.QCheckBox("Motion cancellation (accelerometer)")
        self.motion_btn.setChecked(True)
        self.motion_btn.toggled.connect(self.motionToggled.emit)

        self.dc_servo_btn = QtWidgets.QCheckBox(
            "Firmware DC servo (disabled: 3.3V TIA ref)")
        self.dc_servo_btn.setChecked(False)
        self.dc_servo_btn.setEnabled(False)

        self.center_display_btn = QtWidgets.QCheckBox(
            "Center raw plot (remove DC for display)")
        self.center_display_btn.setChecked(True)
        self.center_display_btn.toggled.connect(self.centerDisplayToggled.emit)

        self.boost_opt_btn = QtWidgets.QCheckBox(
            "Auto-optimize BOOST (SiPM overvoltage)")
        self.boost_opt_btn.setChecked(False)
        self.boost_opt_btn.toggled.connect(self._on_boost_opt_toggle)

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

    def _on_auto_toggle(self, checked: bool):
        self.auto_btn.setText(
            "Disable auto-tuning" if checked else "Enable auto-tuning (AGC)")
        self.auto_status.setText("Auto-tuning: active" if checked else "Auto-tuning: off")
        self.autoToggled.emit(checked)

    def _on_boost_opt_toggle(self, checked: bool):
        self.boost_status.setText(
            "BOOST optimizer: exploring..." if checked else "BOOST optimizer: off")
        self.boostOptToggled.emit(checked)

    def is_motion_cancel_enabled(self) -> bool:
        return self.motion_btn.isChecked()

    def is_center_display_enabled(self) -> bool:
        return self.center_display_btn.isChecked()

    def set_auto_status(self, text: str):
        self.auto_status.setText(f"Auto-tuning: {text}")

    def set_boost_status(self, text: str):
        self.boost_status.setText(f"BOOST optimizer: {text}")
