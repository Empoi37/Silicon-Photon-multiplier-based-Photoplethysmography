#!/usr/bin/env python3
"""pc/ml/sweep_motion_cancel.py
────────────────────────────────
Sweeps `cancel_motion()`'s parameters (n_lags, ridge_alpha) in
pipeline/core.py against PPG-DaLiA ground truth, reusing the same dataset
loading as pc/ml/train_hr_correction.py (pc/ml/dalia.py) -- but *not*
dalia.replay_subject's filtered yield: that only yields windows where
result.bpm_valid is True, which excludes exactly the high-motion windows
this sweep needs to measure (motion tanks spectral quality/regularity,
which is what bpm_valid gates on). This replays subjects itself and
collects `bpm_raw` (the tracker's smoothed output, what a user actually
sees -- it holds the last committed value through invalid windows rather
than going blank) for *every* labeled instant, unconditionally.

Reports both overall MAE and MAE restricted to high-motion windows. The
high-motion cutoff is picked as a percentile of the pooled motion_level
distribution rather than a fixed absolute number: pipeline/core.py's own
motion_norm = motion_level/512 threshold is calibrated to the deployed
hardware's accelerometer counts, and PPG-DaLiA's rescaled-to-milli-g Empatica
E4 accel (see dalia.py's load_subject) lands on a very different absolute
scale -- confirmed empirically: that threshold produced zero high-motion
windows across an entire subject's session here. A percentile split is
dataset-agnostic and still isolates "the windows where this subject was
moving the most," which is what actually matters for evaluating cancellation.

No train/test split is needed here (unlike train_hr_correction.py's LOSO
CV): the filter only ever looks at the current window, there's no
cross-subject fitting to leak.

Usage:
  python pc/ml/sweep_motion_cancel.py --download          # first run: fetch dataset
  python pc/ml/sweep_motion_cancel.py --subjects 2,3,4     # quick coarse pass
  python pc/ml/sweep_motion_cancel.py                      # full grid, all subjects
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

from dalia import BVP_FS, DATASET_DIR, download_dalia, load_subject, subject_ids
from pipeline.core import PpgHrProcessor, PpgMode

HIGH_MOTION_PERCENTILE = 75.0


def _replay_all(subject, **processor_kwargs):
    """Like dalia.replay_subject, but yields (bpm_raw, true_hr, motion_level)
    for every labeled instant regardless of bpm_valid."""
    proc = PpgHrProcessor(frame_fs=BVP_FS, mode=PpgMode.ADC_ONLY,
                          fft_window_s=8.0, max_buffer_s=14.0, **processor_kwargs)
    t = np.arange(len(subject.bvp)) / BVP_FS
    label_idx = 0
    n_labels = len(subject.label_t)
    for i in range(len(subject.bvp)):
        ppg = float(subject.bvp[i])
        ax, ay, az = subject.accel[i]
        proc.push(ppg, ppg, ax, ay, az, ts=float(t[i]))
        while label_idx < n_labels and subject.label_t[label_idx] <= t[i]:
            result = proc.compute()
            true_hr = float(subject.label_hr[label_idx])
            label_idx += 1
            if result is None or true_hr <= 0:
                continue
            yield result.bpm_raw, true_hr, result.motion_level


def _collect(ids, dataset_dir, **processor_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Returns (motion_level, abs_err) arrays pooled across all requested subjects."""
    motion, err = [], []
    for sid in (ids if ids is not None else subject_ids(dataset_dir)):
        subject = load_subject(sid, dataset_dir)
        for bpm_raw, true_hr, ml in _replay_all(subject, **processor_kwargs):
            motion.append(ml)
            err.append(abs(bpm_raw - true_hr))
    return np.array(motion), np.array(err)


def _summarize(label: str, motion: np.ndarray, err: np.ndarray, threshold: float) -> tuple[float, float]:
    moving = motion > threshold
    overall = float(err.mean()) if len(err) else float("nan")
    high = float(err[moving].mean()) if moving.any() else float("nan")
    print(f"{label:<42} overall={overall:6.2f}  high-motion(n={int(moving.sum()):>5})={high:6.2f}")
    return overall, high


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true",
                    help="Fetch PPG-DaLiA first (2.7 GB, one-time)")
    ap.add_argument("--subjects", default=None,
                    help="Comma-separated subject IDs (default: all found) -- "
                         "use a small subset for a fast coarse pass")
    ap.add_argument("--n-lags", default="40,80,120",
                    help="Comma-separated n_lags grid")
    ap.add_argument("--ridge-alpha", default="0,1,5,20,50,100",
                    help="Comma-separated ridge_alpha grid (0 = unregularized, the old default)")
    ap.add_argument("--gate", default="var,snr",
                    help="Comma-separated accept/reject gate(s) to compare: "
                         "'var' (old: revert if cleaned variance grew >5%%) or "
                         "'snr' (revert if in-band SNR didn't improve)")
    args = ap.parse_args()

    if args.download:
        download_dalia()

    ids = [int(s) for s in args.subjects.split(",")] if args.subjects else None
    n_lags_grid = [int(s) for s in args.n_lags.split(",")]
    alpha_grid = [float(s) for s in args.ridge_alpha.split(",")]
    gate_grid = args.gate.split(",")

    print("Baselines")
    print("-" * 90)
    # motion_level is only populated when use_motion_cancel=True (it's computed
    # inside cancel_motion), so the percentile threshold has to come from a
    # cancellation-enabled run, not the no-cancellation one.
    motion_ref, err_ref = _collect(ids, DATASET_DIR, motion_n_lags=80, motion_ridge_alpha=0.0, motion_gate="var")
    threshold = float(np.percentile(motion_ref, HIGH_MOTION_PERCENTILE))
    print(f"High-motion cutoff: motion_level > {threshold:.3f} "
          f"(P{HIGH_MOTION_PERCENTILE:.0f} of pooled distribution)\n")
    _summarize("old default (n_lags=80, ridge_alpha=0, var gate)", motion_ref, err_ref, threshold)
    # use_motion_cancel=False never computes motion_level (it's only set inside
    # cancel_motion), but the (subject, label) sequence is identical across
    # configs, so motion_ref's per-window motion values still apply here.
    _, err_nc = _collect(ids, DATASET_DIR, use_motion_cancel=False)
    _summarize("no cancellation", motion_ref, err_nc, threshold)

    print("\nRidge-regularized grid")
    print("-" * 90)
    results = []
    for n_lags, alpha, gate in itertools.product(n_lags_grid, alpha_grid, gate_grid):
        motion, err = _collect(ids, DATASET_DIR, motion_n_lags=n_lags, motion_ridge_alpha=alpha, motion_gate=gate)
        label = f"n_lags={n_lags} ridge_alpha={alpha} gate={gate}"
        overall, high = _summarize(label, motion, err, threshold)
        results.append((high, overall, n_lags, alpha, gate))

    results.sort(key=lambda r: (r[0] if r[0] == r[0] else float("inf")))
    print("\nBest by high-motion MAE:")
    for high, overall, n_lags, alpha, gate in results[:5]:
        print(f"  n_lags={n_lags:>4} ridge_alpha={alpha:<6} gate={gate:<4} "
              f"high-motion={high:6.2f}  overall={overall:6.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
