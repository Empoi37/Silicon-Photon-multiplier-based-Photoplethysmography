"""
app/utils.py
────────────
Utilitaires partagés par l'interface : conversion ADC → mV et
correctif macOS pour les plugins Qt.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np

# ADS1115 single-ended full-scale (mV) selon l'index ADSGAIN 0-5.
ADS_FSR_MV = (6144.0, 4096.0, 2048.0, 1024.0, 512.0, 256.0)


def adc_counts_to_mv(counts: float, adsgain: int) -> float:
    """Convertit un compte brut ADS1115 (AIN0 vs GND) en millivolts."""
    idx = int(np.clip(adsgain, 0, 5))
    return float(counts) * ADS_FSR_MV[idx] / 32768.0


def ensure_qt_plugins_visible() -> None:
    """macOS : démasque les plugins Qt installés par pip pour que PySide6 les charge."""
    if sys.platform != "darwin":
        return
    spec = find_spec("PySide6")
    if spec and spec.origin:
        plugins = Path(spec.origin).parent / "Qt" / "plugins"
        if plugins.is_dir():
            subprocess.run(["chflags", "-R", "nohidden", str(plugins)], check=False)
