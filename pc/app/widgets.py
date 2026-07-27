"""
app/widgets.py
──────────────
Petits widgets Qt réutilisables à travers l'application.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class CommandSlider(QtWidgets.QWidget):
    """
    Slider avec étiquette et valeur affichée, qui envoie une commande
    "CMD:valeur" à l'ESP32 chaque fois qu'il change.

    Utilisation :
        slider = CommandSlider("TIA gain", "GAIN", 0, 128, 34, on_command=send_fn)
    """

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

    def set_silent(self, value: int):
        """Met à jour le slider sans déclencher on_command (pour l'AGC)."""
        self.slider.blockSignals(True)
        self.slider.setValue(int(value))
        self.value_lbl.setText(str(int(value)))
        self.slider.blockSignals(False)
