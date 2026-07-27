"""
app/workers/serial_worker.py
─────────────────────────────
Thread Qt dédié à la lecture du port série. Tourne séparément du thread
principal (UI) pour ne jamais geler l'interface pendant la lecture bloquante.

Format attendu par ligne : "led1,led2,ax,ay,az"
Les lignes commençant par '#' sont des messages de statut du firmware
(ex: réponse à PING) et sont émises séparément via le signal `message`.
"""

from __future__ import annotations

import queue
import time

import serial
from PySide6 import QtCore

BAUDRATE = 115200


class SerialReader(QtCore.QThread):
    """
    Lit le port série en continu dans un thread séparé.

    Signaux
    ───────
    message(str)   Émis pour chaque ligne de statut du firmware ("# ...")
    error(str)     Émis en cas d'erreur d'ouverture ou de lecture du port

    Les échantillons valides sont poussés dans `data_queue` sous forme de
    tuple (led1, led2, ax, ay, az, timestamp_unix).
    """

    message = QtCore.Signal(str)
    error = QtCore.Signal(str)

    def __init__(self, port: str, data_queue: "queue.Queue", baudrate: int = BAUDRATE):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.data_queue = data_queue
        self._running = False
        self._ser: serial.Serial | None = None

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
        """Envoie une commande texte à l'ESP32 (ex: 'GAIN:64')."""
        if self._ser and self._ser.is_open:
            try:
                self._ser.write((text + "\n").encode("utf-8"))
            except serial.SerialException as exc:
                self.error.emit(f"Serial write error: {exc}")

    def stop(self):
        self._running = False
        self.wait(2000)
