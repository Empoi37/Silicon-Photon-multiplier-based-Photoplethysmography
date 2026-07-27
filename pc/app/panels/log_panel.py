"""
app/panels/log_panel.py
──────────────────────────
Zone de texte affichant les messages de statut du firmware et les
événements de l'application (connexion, erreurs, enregistrement...).
"""

from __future__ import annotations

from PySide6 import QtWidgets


class LogPanel(QtWidgets.QGroupBox):
    def __init__(self):
        super().__init__("Log")
        lay = QtWidgets.QVBoxLayout(self)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        lay.addWidget(self.log)

    def append(self, text: str):
        self.log.appendPlainText(text)
