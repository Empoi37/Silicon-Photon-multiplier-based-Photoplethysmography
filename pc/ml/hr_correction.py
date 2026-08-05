"""pc/ml/hr_correction.py
────────────────────────
Lightweight numpy-only inference for the ridge-regression HR correction
model trained by pc/ml/train_hr_correction.py on PPG-DaLiA.

Intentionally dependency-free (no scikit-learn/torch) so the desktop app
never needs anything beyond numpy to load and apply the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "hr_correction.json"

# Must match PpgHrResult field names — pipeline/core.py builds this dict
# straight from the result object.
FEATURE_ORDER = ["fft_bpm", "quality", "inband_snr", "motion_level", "ac_amplitude"]


@dataclass
class HrCorrectionModel:
    feature_order: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def load(cls, path: Path) -> HrCorrectionModel:
        data = json.loads(path.read_text())
        return cls(
            feature_order=list(data["feature_order"]),
            mean=np.asarray(data["mean"], dtype=float),
            scale=np.asarray(data["scale"], dtype=float),
            weights=np.asarray(data["weights"], dtype=float),
            bias=float(data["bias"]),
        )

    @classmethod
    def load_default(cls) -> HrCorrectionModel | None:
        """Returns None if no model has been trained yet — the app must run
        identically with plain FFT tracking in that case."""
        if not MODEL_PATH.exists():
            return None
        try:
            return cls.load(MODEL_PATH)
        except Exception:
            return None

    def predict(self, features: dict) -> float:
        x = np.array([features[k] for k in self.feature_order], dtype=float)
        x_norm = (x - self.mean) / self.scale
        return float(x_norm @ self.weights + self.bias)
