"""
app/panels/connection_panel.py
────────────────────────────────
Panel de connexion : sélection du port série, scan, connexion/déconnexion.
"""

from __future__ import annotations

import serial.tools.list_ports
from PySide6 import QtCore, QtWidgets


class ConnectionPanel(QtWidgets.QGroupBox):
    """
    Signaux
    ───────
    connectRequested(str)   Émis avec le port choisi quand l'utilisateur clique "Connect"
    disconnectRequested()   Émis quand l'utilisateur clique "Disconnect"
    """

    connectRequested = QtCore.Signal(str)
    disconnectRequested = QtCore.Signal()

    def __init__(self):
        super().__init__("Connection")
        lay = QtWidgets.QHBoxLayout(self)

        self.port_combo = QtWidgets.QComboBox()
        self.refresh_btn = QtWidgets.QPushButton("Scan")
        self.connect_btn = QtWidgets.QPushButton("Connect")
        self.connect_btn.setCheckable(True)

        lay.addWidget(self.port_combo, 1)
        lay.addWidget(self.refresh_btn)
        lay.addWidget(self.connect_btn)

        self.refresh_btn.clicked.connect(self.refresh_ports)
        self.connect_btn.clicked.connect(self._on_toggle)

        self.refresh_ports()

    def refresh_ports(self):
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
        if current in ports:
            self.port_combo.setCurrentText(current)

    def _on_toggle(self, checked: bool):
        if checked:
            port = self.port_combo.currentText()
            if not port:
                self.connect_btn.setChecked(False)
                return
            self.connectRequested.emit(port)
        else:
            self.disconnectRequested.emit()

    def set_connected_ui(self, connected: bool):
        """Met à jour l'apparence du panel selon l'état de connexion."""
        self.connect_btn.setChecked(connected)
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.port_combo.setEnabled(not connected)
