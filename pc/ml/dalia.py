"""pc/ml/dalia.py
────────────────
Loads the PPG-DaLiA dataset (UCI ML Repository, DOI 10.24432/C53890, CC BY 4.0)
and replays each subject's wrist PPG + accelerometer through the *deployed*
PpgHrProcessor (pipeline/core.py) to produce (features, true_hr) training
pairs for pc/ml/train_hr_correction.py.

Replaying through the real processor — rather than re-deriving features here —
guarantees training features match exactly what's available live, so there's
no train/serve skew.
"""

from __future__ import annotations

import pickle
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import PpgHrProcessor, PpgMode

DALIA_URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "datasets" / "ppg_dalia"

BVP_FS = 64.0
ACC_FS = 32.0
# Standard PPG-DaLiA evaluation protocol: one HR label every 2s, each
# averaged over an 8s ECG window.
LABEL_STRIDE_S = 2.0
LABEL_WINDOW_S = 8.0


def download_dalia(dest_dir: Path = DATASET_DIR, force: bool = False) -> Path:
    """Download and fully unzip the (nested) PPG-DaLiA archive.

    Idempotent: skips the download if S*.pkl files are already present.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not force and list(dest_dir.rglob("S*.pkl")):
        return dest_dir

    zip_path = dest_dir / "_download.zip"
    print(f"Downloading PPG-DaLiA (2.7 GB) from {DALIA_URL} ...")
    with urlopen(DALIA_URL) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        with zip_path.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if total:
                    print(f"\r  {read / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()

    _unzip_recursive(zip_path, dest_dir)
    if not list(dest_dir.rglob("S*.pkl")):
        raise RuntimeError(f"No S*.pkl files found under {dest_dir} after extraction")
    return dest_dir


def _unzip_recursive(zip_path: Path, dest_dir: Path, depth: int = 0):
    """The UCI archive wraps an inner data.zip inside the outer download."""
    if depth > 4 or not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    for nested in list(dest_dir.rglob("*.zip")):
        _unzip_recursive(nested, dest_dir, depth + 1)


def subject_ids(dataset_dir: Path = DATASET_DIR) -> list[int]:
    ids = set()
    for p in dataset_dir.rglob("S*.pkl"):
        m = re.fullmatch(r"S(\d+)\.pkl", p.name)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)


def _find_subject_pkl(dataset_dir: Path, subject_id: int) -> Path:
    name = f"S{subject_id}.pkl"
    for p in dataset_dir.rglob(name):
        return p
    raise FileNotFoundError(f"{name} not found under {dataset_dir} — run download_dalia() first")


@dataclass
class SubjectData:
    subject_id: int
    bvp: np.ndarray          # PPG @ BVP_FS
    accel: np.ndarray        # (N, 3), resampled to BVP_FS
    label_hr: np.ndarray     # ground-truth BPM, one per LABEL_STRIDE_S
    label_t: np.ndarray      # seconds — window CENTER for each label


def load_subject(subject_id: int, dataset_dir: Path = DATASET_DIR) -> SubjectData:
    path = _find_subject_pkl(dataset_dir, subject_id)
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    bvp = np.asarray(data["signal"]["wrist"]["BVP"], dtype=float).reshape(-1)
    # PPG-DaLiA's Empatica E4 ACC is in g (range ~[-2, 2]); the deployed
    # SparkFun_LIS2DH12 firmware (esp32/firmware_esp32_full.ino, lis.getX/Y/Z)
    # reports milli-g. Rescale here so motion_level -- computed downstream by
    # the same PpgHrProcessor code used live -- lands on the same numeric
    # scale in training as in deployment. Without this, the live app's
    # motion_level is ~1000x outside the training distribution.
    acc_raw = np.asarray(data["signal"]["wrist"]["ACC"], dtype=float) * 1000.0
    label = np.asarray(data["label"], dtype=float).reshape(-1)

    t_bvp = np.arange(len(bvp)) / BVP_FS
    t_acc = np.arange(len(acc_raw)) / ACC_FS
    accel = np.column_stack([
        np.interp(t_bvp, t_acc, acc_raw[:, i]) for i in range(acc_raw.shape[1])
    ])

    label_t = np.arange(len(label)) * LABEL_STRIDE_S + LABEL_WINDOW_S / 2.0
    return SubjectData(subject_id, bvp, accel, label, label_t)


def replay_subject(subject: SubjectData, **processor_kwargs):
    """Push one subject's BVP+accel through a fresh PpgHrProcessor sample by
    sample, calling compute() at each ground-truth label timestamp.

    `processor_kwargs` is forwarded straight to PpgHrProcessor(...), so any
    tunable field there (motion_n_lags, motion_ridge_alpha, adaptive_smoothing,
    ...) can be swept without editing this function.

    Yields (features: dict, true_hr: float). Feature keys mirror
    PpgHrResult fields exactly, so pc/ml/hr_correction.py can apply the
    trained model to the same result object at inference time.
    """
    proc = PpgHrProcessor(frame_fs=BVP_FS, mode=PpgMode.ADC_ONLY,
                          fft_window_s=8.0, max_buffer_s=14.0,
                          **processor_kwargs)
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
            if result is None or not result.bpm_valid or true_hr <= 0:
                continue
            yield ({
                "fft_bpm": result.bpm,
                "quality": result.snr,
                "inband_snr": result.inband_snr,
                "motion_level": result.motion_level,
                "ac_amplitude": result.ac_amplitude,
            }, true_hr)


def iter_subjects(dataset_dir: Path = DATASET_DIR, ids: list[int] | None = None,
                  **processor_kwargs):
    """Yields (subject_id, [(features, true_hr), ...]) for each subject.

    `processor_kwargs` is forwarded to replay_subject() -> PpgHrProcessor(...).
    """
    for sid in (ids if ids is not None else subject_ids(dataset_dir)):
        subject = load_subject(sid, dataset_dir)
        yield sid, list(replay_subject(subject, **processor_kwargs))
