#!/usr/bin/env python3
"""Visualize PPG + accelerometer logs (nRF Connect BLE export or GUI CSV).

Usage:
  python visualize_ppg_log.py /path/to/PPG-SiPM.csv
  python visualize_ppg_log.py recording.csv --save plot.png
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Permet de lancer ce script directement (python tools/visualize_log.py)
# sans avoir à configurer PYTHONPATH ou marquer pc/ comme Sources Root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core import PpgHrProcessor, PpgMode

NRF_VALUE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}\.\d{3}),Connected Device,Application,"
    r'""(\d+,\d+,-?\d+,-?\d+,-?\d+)\s*\n"\s*value received',
    re.MULTILINE,
)
NRF_PLAIN_RE = re.compile(
    r"^(\d+),(\d+),(-?\d+),(-?\d+),(-?\d+)\s*$"
)


def _parse_clock(ts: str) -> datetime:
    return datetime.strptime(ts, "%H:%M:%S.%f")


def load_nrf_connect_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(NRF_VALUE_RE.finditer(text))
    if not matches:
        raise ValueError(f"No BLE samples found in nRF Connect log: {path}")

    clocks = [_parse_clock(m.group(1)) for m in matches]
    samples = [tuple(map(int, m.group(2).split(","))) for m in matches]

    t_sec = np.zeros(len(samples), dtype=float)
    prev = clocks[0]
    t_sec[0] = 0.0
    for i in range(1, len(clocks)):
        dt = (clocks[i] - prev).total_seconds()
        if dt < 0:
            dt += 24 * 3600
        if dt == 0:
            dt = 1e-6
        t_sec[i] = t_sec[i - 1] + dt
        prev = clocks[i]

    duration = float(t_sec[-1])
    if duration <= 0:
        duration = max(len(samples) / 250.0, 1.0)
        t_sec = np.arange(len(samples), dtype=float) / 250.0

    arr = np.asarray(samples, dtype=float)
    fs_est = (len(samples) - 1) / duration if duration > 0 else 250.0

    return {
        "source": "nrf_connect",
        "t": t_sec,
        "ppg1": arr[:, 0],
        "ppg2": arr[:, 1],
        "ax": arr[:, 2],
        "ay": arr[:, 3],
        "az": arr[:, 4],
        "fs_est": float(np.clip(fs_est, 5.0, 250.0)),
        "duration": duration,
    }


def load_gui_recording(path: Path) -> dict:
    try:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Could not read GUI CSV: {path}") from exc

    if data.size == 0:
        raise ValueError(f"Empty CSV: {path}")

    data = np.atleast_1d(data)
    names = [n.lower() for n in data.dtype.names]

    def col(*candidates: str) -> np.ndarray:
        for c in candidates:
            if c in names:
                return data[c].astype(float)
        raise KeyError(f"Missing column (tried {candidates}) in {names}")

    t = col("unix_time_s", "time_s", "t")
    t = t - t[0]
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    fs_est = float(1.0 / np.median(dt)) if len(dt) else 250.0

    return {
        "source": "gui_csv",
        "t": t.astype(float),
        "ppg1": col("led1_adc", "ppg", "led1"),
        "ppg2": col("led2_adc", "ppg2", "led2"),
        "ax": col("accel_x", "ax"),
        "ay": col("accel_y", "ay"),
        "az": col("accel_z", "az"),
        "fs_est": fs_est,
        "duration": float(t[-1]) if len(t) > 1 else 0.0,
    }


def load_plain_csv(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = NRF_PLAIN_RE.match(line)
        if m:
            rows.append(tuple(map(int, m.groups())))
    if not rows:
        raise ValueError(f"No ppg,ppg,ax,ay,az rows in: {path}")

    arr = np.asarray(rows, dtype=float)
    fs_est = 250.0
    t = np.arange(len(arr), dtype=float) / fs_est
    return {
        "source": "plain_csv",
        "t": t,
        "ppg1": arr[:, 0],
        "ppg2": arr[:, 1],
        "ax": arr[:, 2],
        "ay": arr[:, 3],
        "az": arr[:, 4],
        "fs_est": fs_est,
        "duration": float(t[-1]),
    }


def load_log(path: Path) -> dict:
    text_head = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    if "Timestamp,Source,Level,Line" in text_head:
        return load_nrf_connect_log(path)
    try:
        return load_gui_recording(path)
    except Exception:
        return load_plain_csv(path)


def compute_hr_timeline(
    data: dict,
    *,
    step_s: float = 0.2,
    use_motion_cancel: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PpgHrProcessor]:
    """Replay samples through PpgHrProcessor like the live GUI."""
    proc = PpgHrProcessor(
        frame_fs=data["fs_est"],
        mode=PpgMode.MULTILED,
        fft_window_s=8.0,
        max_buffer_s=14.0,
        use_motion_cancel=use_motion_cancel,
    )

    t = data["t"]
    hr_t: list[float] = []
    hr_bpm: list[float] = []
    hr_valid: list[bool] = []
    next_compute = 0.0

    for i in range(len(t)):
        proc.push(
            int(data["ppg1"][i]),
            int(data["ppg2"][i]),
            int(data["ax"][i]),
            int(data["ay"][i]),
            int(data["az"][i]),
            ts=float(t[i]),
        )
        if t[i] < next_compute:
            continue
        next_compute = float(t[i]) + step_s
        result = proc.compute()
        if result is None:
            continue
        hr_t.append(float(t[i]))
        hr_bpm.append(float(result.bpm) if result.bpm_valid else np.nan)
        hr_valid.append(bool(result.bpm_valid))

    return (
        np.asarray(hr_t, dtype=float),
        np.asarray(hr_bpm, dtype=float),
        np.asarray(hr_valid, dtype=bool),
        proc,
    )


def plot_log(
    data: dict,
    hr_t: np.ndarray,
    hr_bpm: np.ndarray,
    hr_valid: np.ndarray,
    final_result,
    *,
    title: str,
    save_path: Path | None = None,
) -> None:
    t = data["t"]
    ppg = 0.5 * (data["ppg1"] + data["ppg2"])

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=13)

    ax0 = axes[0]
    ax0.plot(t, data["ppg1"], color="#27ae60", linewidth=0.8, alpha=0.85, label="PPG ch1")
    if not np.allclose(data["ppg1"], data["ppg2"]):
        ax0.plot(t, data["ppg2"], color="#2ecc71", linewidth=0.6, alpha=0.6, label="PPG ch2")
    ax0.set_ylabel("ADC counts")
    ax0.set_title("Raw PPG")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    if final_result is not None and len(final_result.filtered) > 0:
        n = len(final_result.filtered)
        fs = final_result.fs_effective or data["fs_est"]
        t_f = t[-1] - (np.arange(n)[::-1] / fs)
        filt = final_result.filtered
        ax1.plot(t_f, filt, color="#e67e22", linewidth=1.0, label="Filtered (GUI algo)")
        if len(final_result.beat_indices) > 0:
            beats = final_result.beat_indices
            ax1.scatter(t_f[beats], filt[beats], s=12, c="#e74c3c", zorder=3, label="beats")
    else:
        ppg_ac = ppg - np.median(ppg)
        ax1.plot(t, ppg_ac, color="#e67e22", linewidth=0.8, label="PPG - median")
    ax1.set_ylabel("a.u.")
    ax1.set_title("Processed PPG")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = axes[2]
    ax2.plot(t, data["ax"], linewidth=0.7, label="X", color="#3498db")
    ax2.plot(t, data["ay"], linewidth=0.7, label="Y", color="#f1c40f")
    ax2.plot(t, data["az"], linewidth=0.7, label="Z", color="#9b59b6")
    ax2.set_ylabel("counts")
    ax2.set_title("Accelerometer (LIS2DH12)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8, ncol=3)

    ax3 = axes[3]
    if len(hr_t) > 0:
        ax3.plot(hr_t, hr_bpm, color="#c0392b", linewidth=1.5, drawstyle="steps-post")
        valid_mask = hr_valid & np.isfinite(hr_bpm)
        if np.any(valid_mask):
            ax3.scatter(hr_t[valid_mask], hr_bpm[valid_mask], s=8, c="#c0392b", alpha=0.5)
    ax3.set_ylim(40, 200)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("BPM")
    ax3.set_title("Heart rate over time (GUI FFT + motion cancel)")
    ax3.grid(True, alpha=0.3)

    if final_result is not None and final_result.bpm_valid:
        ax3.axhline(final_result.bpm, color="#95a5a6", linestyle="--", linewidth=0.8,
                    label=f"final {final_result.bpm:.0f} BPM")
        ax3.legend(loc="upper right", fontsize=8)

    meta = (
        f"{data['source']} | {data['duration']:.1f} s | "
        f"~{data['fs_est']:.1f} Hz | {len(t)} samples"
    )
    fig.text(0.01, 0.01, meta, fontsize=9, color="#555")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize PPG BLE/GUI logs")
    parser.add_argument("log", type=Path, help="nRF Connect CSV, GUI recording, or plain CSV")
    parser.add_argument("--save", type=Path, default=None, help="Save figure to PNG instead of showing")
    parser.add_argument("--no-motion-cancel", action="store_true")
    parser.add_argument("--hr-step", type=float, default=0.2, help="HR compute interval (s), GUI ~0.2")
    args = parser.parse_args(argv)

    if not args.log.is_file():
        print(f"File not found: {args.log}", file=sys.stderr)
        return 1

    data = load_log(args.log)
    hr_t, hr_bpm, hr_valid, proc = compute_hr_timeline(
        data,
        step_s=args.hr_step,
        use_motion_cancel=not args.no_motion_cancel,
    )
    final_result = proc.compute()

    valid_frac = float(np.mean(hr_valid)) if len(hr_valid) else 0.0
    med_bpm = float(np.nanmedian(hr_bpm)) if np.any(np.isfinite(hr_bpm)) else 0.0
    print(f"Loaded {args.log.name}: {len(data['t'])} samples, {data['duration']:.1f} s, "
          f"fs~{data['fs_est']:.1f} Hz")
    print(f"HR timeline: {len(hr_t)} points, valid {100*valid_frac:.0f}%, median {med_bpm:.0f} BPM")
    if final_result is not None:
        print(f"Final BPM: {final_result.bpm:.1f} ({'valid' if final_result.bpm_valid else 'invalid'}), "
              f"quality {final_result.snr:.2f}")

    plot_log(
        data,
        hr_t,
        hr_bpm,
        hr_valid,
        final_result,
        title=args.log.stem,
        save_path=args.save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
