"""
app/panels/recording_panel.py
────────────────────────────────
Panel d'enregistrement CSV brut. N'écrit rien lui-même — MainWindow gère le
fichier — ce panel se contente de l'UI et des signaux start/stop.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class RecordingPanel(QtWidgets.QGroupBox):
    """
    Signaux
    ───────
    startRequested()   L'utilisateur veut démarrer l'enregistrement
    stopRequested()    L'utilisateur veut arrêter l'enregistrement
    """

    startRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()

    def __init__(self):
        super().__init__("Raw recording")
        lay = QtWidgets.QVBoxLayout(self)

        self.record_btn = QtWidgets.QPushButton("Start CSV recording")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._on_toggle)

        self.record_label = QtWidgets.QLabel("No recording")
        self.record_label.setWordWrap(True)
        self.record_label.setStyleSheet("color:#777;font-size:11px;")

        lay.addWidget(self.record_btn)
        lay.addWidget(self.record_label)

    def _on_toggle(self, checked: bool):
        if checked:
            self.startRequested.emit()
        else:
            self.stopRequested.emit()

    def set_recording_ui(self, path_text: str | None):
        """path_text=None → pas d'enregistrement en cours."""
        if path_text is None:
            self.record_btn.setChecked(False)
            self.record_btn.setText("Start CSV recording")
        else:
            self.record_btn.setText("Stop CSV recording")
            self.record_label.setText(path_text)

    def set_saved_label(self, path_text: str):
        self.record_label.setText(f"Saved: {path_text}")
