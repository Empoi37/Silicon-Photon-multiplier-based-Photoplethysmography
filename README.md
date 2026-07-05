# Silicon-Photon-multiplier-based Photoplethysmography

Real-time PPG acquisition and heart-rate monitoring for a SiPM-based wearable sensor prototype.

## Repository layout

- `python/` — Host-side GUI and signal-processing tools
- `firmware_esp32/` — ESP32 firmware for sampling and front-end control

## Python GUI

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python gui_ppg.py
```

Supporting scripts:

- `serial_diag.py` — Serial link diagnostics (sample rate, PING)
- `analyze_ppg_recording.py` — Offline analysis of saved CSV recordings

## ESP32 firmware

Open `firmware_esp32/firmware_esp32.ino` in the Arduino IDE.

Required libraries:

- Adafruit ADS1X15
- SparkFun LIS2DH12 Arduino Library

Target: ESP32 @ 115200 baud. The firmware streams PPG + accelerometer samples as CSV and accepts front-end tuning commands over serial.
