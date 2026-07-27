#!/usr/bin/env python3
"""Analyze a raw PPG CSV recording and compare to reference databases.

Usage:
  python analyze_ppg_recording.py recordings/ppg_raw_YYYYMMDD_HHMMSS.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Permet de lancer ce script directement (python tools/analyze_recording.py)
# sans avoir à configurer PYTHONPATH ou marquer pc/ comme Sources Root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import PpgHrProcessor, PpgMode, robust_ac_amplitude


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"Empty CSV: {path}")
    return np.atleast_1d(data)


def estimate_fs(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(1.0 / np.median(dt)) if len(dt) else 0.0


def summarize_local(data: np.ndarray) -> dict:
    t = data["unix_time_s"].astype(float)
    fs = estimate_fs(t)
    led1 = data["led1_adc"].astype(float)
    led2 = data["led2_adc"].astype(float)
    ax = data["accel_x"].astype(float)
    ay = data["accel_y"].astype(float)
    az = data["accel_z"].astype(float)

    proc = PpgHrProcessor(frame_fs=fs or 250.0, mode=PpgMode.MULTILED)
    for i in range(len(t)):
        proc.push(int(led1[i]), int(led2[i]), int(ax[i]), int(ay[i]), int(az[i]), ts=float(t[i]))
    result = proc.compute()

    mean_signal = 0.5 * (led1 + led2)
    dc = float(np.median(mean_signal))
    raw_ac = robust_ac_amplitude(mean_signal - dc, fs or 250.0)
    raw_acdc = 100.0 * raw_ac / abs(dc) if abs(dc) > 1e-9 else 0.0

    accel_mag = np.sqrt(ax * ax + ay * ay + az * az)
    accel_ac = robust_ac_amplitude(accel_mag - np.median(accel_mag), fs or 250.0)

    return {
        "fs": fs,
        "duration": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "dc": dc,
        "raw_ac": raw_ac,
        "raw_acdc": raw_acdc,
        "processed_ac": result.ac_amplitude if result else 0.0,
        "bpm": result.bpm if result else 0.0,
        "bpm_valid": result.bpm_valid if result else False,
        "fft_quality": result.snr if result else 0.0,
        "accel_ac": accel_ac,
    }


def bidmc_reference_metrics(max_records: int = 5) -> dict | None:
    try:
        import wfdb
    except Exception:
        return None

    acdc_values = []
    fs_values = []
    for i in range(1, max_records + 1):
        try:
            rec = wfdb.rdrecord(f"bidmc{i:02d}", pn_dir="bidmc/1.0.0")
        except Exception:
            continue
        ppg_idx = next(
            (j for j, name in enumerate(rec.sig_name) if "ppg" in name.lower()), None)
        if ppg_idx is None:
            continue
        ppg = rec.p_signal[:, ppg_idx].astype(float)
        fs = float(rec.fs)
        dc = float(np.median(ppg))
        ac = robust_ac_amplitude(ppg - dc, fs)
        acdc = 100.0 * ac / abs(dc) if abs(dc) > 1e-9 else np.nan
        if np.isfinite(acdc):
            acdc_values.append(acdc)
            fs_values.append(fs)

    if not acdc_values:
        return None
    return {
        "records": len(acdc_values),
        "fs_median": float(np.median(fs_values)),
        "acdc_median": float(np.median(acdc_values)),
        "acdc_p10": float(np.percentile(acdc_values, 10)),
        "acdc_p90": float(np.percentile(acdc_values, 90)),
    }


def print_report(local: dict, ref: dict | None):
    print("=== Local recording ===")
    print(f"Duration: {local['duration']:.1f} s")
    print(f"Effective sample rate: {local['fs']:.1f} Hz")
    print(f"DC median: {local['dc']:.1f} counts")
    print(f"Raw AC (P95-P5): {local['raw_ac']:.2f} counts")
    print(f"Raw AC/DC: {local['raw_acdc']:.3f} %")
    print(f"Accelerometer AC: {local['accel_ac']:.2f}")
    print(f"BPM: {local['bpm']:.1f} ({'valid' if local['bpm_valid'] else 'invalid'})")
    print(f"FFT quality: {local['fft_quality']:.2f}")

    print("\n=== Reference databases ===")
    print("Typical reflective PPG AC/DC: 0.1 % to 5 % depending on site and contact.")
    if ref:
        print(
            f"BIDMC ({ref['records']} records): AC/DC median {ref['acdc_median']:.3f} %, "
            f"P10-P90 {ref['acdc_p10']:.3f}-{ref['acdc_p90']:.3f} %, "
            f"Fs median {ref['fs_median']:.1f} Hz")
    else:
        print("BIDMC unavailable (install wfdb and ensure PhysioNet access).")

    print("\n=== Quick assessment ===")
    if local["raw_acdc"] < 0.05:
        print("- AC/DC very low: increase LED/TIA gain or improve mechanical contact.")
    elif local["raw_acdc"] > 5.0:
        print("- AC/DC very high: likely motion artifact or saturation.")
    else:
        print("- AC/DC in a plausible PPG range; check mechanical stability.")
    if local["fft_quality"] < 1.5:
        print("- Low FFT quality: heart rate peak not clearly above noise.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--bidmc-records", type=int, default=5)
    args = parser.parse_args()

    data = load_csv(args.csv)
    local = summarize_local(data)
    ref = bidmc_reference_metrics(args.bidmc_records)
    print_report(local, ref)


if __name__ == "__main__":
    main()
