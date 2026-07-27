#!/usr/bin/env python3
"""Serial diagnostic tool for the ESP32 PPG firmware (runs without the GUI).

Measures the raw firmware sample rate and reads device status via PING.

Usage:
  python serial_diag.py
  python serial_diag.py /dev/cu.usbserial-XXXX
  python serial_diag.py --seconds 10
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import serial
import serial.tools.list_ports


def autodetect_port() -> str | None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        dev = p.device.lower()
        if any(k in dev for k in ("usbserial", "slab", "wchusb", "cu.usb", "ttyusb", "ttyacm")):
            return p.device
    return ports[0].device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    port = args.port or autodetect_port()
    if not port:
        print("No serial port found. Connect the ESP32 or specify a port.")
        for p in serial.tools.list_ports.comports():
            print(" ", p.device, "-", p.description)
        return 1

    print(f"# Port: {port} @ {args.baud} baud")
    try:
        ser = serial.Serial(port, args.baud, timeout=0.2)
    except serial.SerialException as exc:
        print(f"Cannot open port: {exc}")
        return 1

    time.sleep(0.3)
    ser.reset_input_buffer()

    ser.write(b"PING\n")
    time.sleep(0.4)
    t_end = time.time() + 1.0
    banners = []
    buf = b""
    while time.time() < t_end:
        buf += ser.read(ser.in_waiting or 1)
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.decode("utf-8", "ignore").strip()
            if s.startswith("#"):
                banners.append(s)

    print("\n# --- Firmware status (PING) ---")
    for b in banners:
        print(" ", b)
    if not banners:
        print("  (no status lines received)")

    print(f"\n# --- Sample rate measurement ({args.seconds:.0f} s) ---")
    ser.reset_input_buffer()
    stamps: list[float] = []
    values: list[int] = []
    buf = b""
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            continue
        now = time.time()
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.decode("utf-8", "ignore").strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split(",")
            if len(parts) != 5:
                continue
            try:
                ppg = int(parts[0])
            except ValueError:
                continue
            stamps.append(now)
            values.append(ppg)

    ser.close()

    n = len(stamps)
    if n < 5:
        print(f"  Only {n} data lines received.")
        return 2

    dur = stamps[-1] - stamps[0]
    rate = (n - 1) / dur if dur > 0 else 0.0
    dt = np.diff(np.asarray(stamps))
    dt = dt[dt > 0]
    v = np.asarray(values, dtype=float)

    print(f"  Lines received : {n}")
    print(f"  Duration       : {dur:.2f} s")
    print(f"  Sample rate    : {rate:.1f} Hz")
    if len(dt):
        print(f"  Interval (med) : {np.median(dt) * 1000:.1f} ms")
    print(f"  DC mean        : {v.mean():.0f} counts ({100 * v.mean() / 32767:.1f} % of full scale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
