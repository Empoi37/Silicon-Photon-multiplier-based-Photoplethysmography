#!/usr/bin/env python3
"""pc/ml/train_hr_correction.py
────────────────────────────────
Trains a small ridge-regression correction on top of the deployed FFT-based
BPM tracker (pipeline/core.py), using PPG-DaLiA (open, CC BY 4.0) as ground
truth. See pc/ml/dalia.py for how subjects are replayed through the real
pipeline and pc/ml/hr_correction.py for how the saved model is applied live.

Evaluated with leave-one-subject-out (LOSO) cross-validation — the standard
protocol for this dataset — so the printed MAE reflects generalization to a
person the model has never seen, not just curve-fitting.

Usage:
  python pc/ml/train_hr_correction.py --download   # first run: fetch dataset
  python pc/ml/train_hr_correction.py              # subsequent runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from dalia import DATASET_DIR, download_dalia, iter_subjects
from hr_correction import FEATURE_ORDER, MODEL_PATH


def _build_xy(samples: list[tuple[dict, float]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[f[k] for k in FEATURE_ORDER] for f, _ in samples], dtype=float)
    y = np.array([t for _, t in samples], dtype=float)
    return x, y


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    """Closed-form ridge regression via augmented least squares — no
    scikit-learn dependency. Standardizes features and centers the target
    so `alpha` regularizes toward the mean rather than toward zero."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    x_norm = (x - mean) / scale

    n, d = x_norm.shape
    y_mean = float(y.mean())
    A = np.vstack([x_norm, np.sqrt(alpha) * np.eye(d)])
    b = np.concatenate([y - y_mean, np.zeros(d)])
    weights, *_ = np.linalg.lstsq(A, b, rcond=None)
    return mean, scale, weights, y_mean


def _predict(mean, scale, weights, bias, x: np.ndarray) -> np.ndarray:
    return (x - mean) / scale @ weights + bias


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="Fetch PPG-DaLiA first (2.7 GB, one-time)")
    ap.add_argument("--alpha", type=float, default=5.0, help="Ridge regularization strength")
    ap.add_argument("--subjects", default=None,
                    help="Comma-separated subject IDs to use (default: all found)")
    args = ap.parse_args()

    if args.download:
        download_dalia()

    ids = [int(s) for s in args.subjects.split(",")] if args.subjects else None

    print("Replaying subjects through PpgHrProcessor (reuses the live pipeline code)...")
    per_subject: dict[int, list[tuple[dict, float]]] = {}
    for sid, samples in iter_subjects(DATASET_DIR, ids):
        per_subject[sid] = samples
        print(f"  S{sid}: {len(samples)} usable labeled windows")

    all_ids = sorted(per_subject)
    if len(all_ids) < 2:
        print("Need at least 2 subjects for leave-one-subject-out evaluation.")
        return 1

    print(f"\n{'Subject':>8} {'N':>6} {'MAE raw':>10} {'MAE corrected':>14}")
    print("-" * 42)
    raw_errs_all, corr_errs_all = [], []
    fft_bpm_idx = FEATURE_ORDER.index("fft_bpm")
    for held_out in all_ids:
        train_samples = [s for sid in all_ids if sid != held_out for s in per_subject[sid]]
        test_samples = per_subject[held_out]
        if not train_samples or not test_samples:
            continue

        x_train, y_train = _build_xy(train_samples)
        x_test, y_test = _build_xy(test_samples)
        mean, scale, weights, bias = _fit_ridge(x_train, y_train, args.alpha)

        raw_pred = x_test[:, fft_bpm_idx]
        corr_pred = _predict(mean, scale, weights, bias, x_test)

        raw_err = np.abs(raw_pred - y_test)
        corr_err = np.abs(corr_pred - y_test)
        raw_errs_all.append(raw_err)
        corr_errs_all.append(corr_err)

        print(f"{'S' + str(held_out):>8} {len(test_samples):>6} "
              f"{raw_err.mean():>10.2f} {corr_err.mean():>14.2f}")

    raw_mae = float(np.concatenate(raw_errs_all).mean())
    corr_mae = float(np.concatenate(corr_errs_all).mean())
    print("-" * 42)
    print(f"{'ALL (LOSO)':>8} {'':>6} {raw_mae:>10.2f} {corr_mae:>14.2f}")
    print(f"\nRaw FFT tracker MAE: {raw_mae:.2f} BPM")
    print(f"Corrected MAE:       {corr_mae:.2f} BPM "
          f"({'better' if corr_mae < raw_mae else 'WORSE -- check features/alpha before shipping'})")

    all_samples = [s for sid in all_ids for s in per_subject[sid]]
    x_all, y_all = _build_xy(all_samples)
    mean, scale, weights, bias = _fit_ridge(x_all, y_all, args.alpha)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps({
        "feature_order": FEATURE_ORDER,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias,
        "trained_on": f"PPG-DaLiA subjects {all_ids}",
        "loso_mae_raw_bpm": raw_mae,
        "loso_mae_corrected_bpm": corr_mae,
    }, indent=2))
    print(f"\nSaved model -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
