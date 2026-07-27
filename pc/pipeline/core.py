"""PPG heart-rate processing: filtering, motion cancellation, spectral BPM tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt


class PpgMode(str, Enum):
    MULTILED = "multiled"
    ADC_ONLY = "adc_only"


@dataclass
class PpgHrResult:
    raw_signal: np.ndarray
    raw_led1: np.ndarray
    raw_led2: np.ndarray
    preprocessed: np.ndarray
    filtered: np.ndarray
    beat_indices: np.ndarray
    bpm: float
    bpm_valid: bool
    fs_effective: float
    snr: float = 0.0
    ac_amplitude: float = 0.0
    bpm_method: str = "fft"
    motion_level: float = 0.0
    inband_snr: float = 0.0


def robust_ac_amplitude(x: np.ndarray, fs: float) -> float:
    """Peak-to-peak amplitude (P95 - P5) after cardiac bandpass."""
    if len(x) < 4:
        return 0.0
    y = _bandpass(x.astype(float), fs, 0.7, 4.0)
    return float(np.percentile(y, 95) - np.percentile(y, 5))


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    if len(x) < int(1.5 * fs):
        return x - np.mean(x)
    nyq = 0.5 * fs
    sos = butter(4, [max(lo / nyq, 1e-4), min(hi / nyq, 0.99)], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        return x - np.mean(x)


def _detrend_channel(x: np.ndarray, fs: float) -> np.ndarray:
    x = x.astype(float) - np.mean(x)
    if len(x) >= 8:
        x = detrend(x, type="linear")
    return _bandpass(x, fs, 0.7, 4.0)


def _despike(x: np.ndarray, sigma_limit: float = 5.0) -> np.ndarray:
    if len(x) < 8:
        return x.copy()
    y = x.astype(float).copy()
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)))
    if mad < 1e-12:
        return y
    thr = sigma_limit * 1.4826 * mad
    bad = np.abs(y - med) > thr
    if not np.any(bad):
        return y
    bad[1:] |= bad[:-1]
    bad[:-1] |= bad[1:]
    good = np.flatnonzero(~bad)
    if len(good) < 2:
        return np.clip(y, med - thr, med + thr)
    y[np.flatnonzero(bad)] = np.interp(np.flatnonzero(bad), good, y[good])
    return y


def _robust_pp(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def inband_snr(x: np.ndarray, fs: float, lo: float = 0.7, hi: float = 3.5,
               nlo: float = 5.0, nhi: float = 15.0) -> float:
    """Cardiac-band power divided by out-of-band noise power."""
    if len(x) < 16:
        return 0.0
    w = (x - np.mean(x)) * np.hanning(len(x))
    n = 1
    while n < 4 * len(w):
        n <<= 1
    p = np.abs(np.fft.rfft(w, n=n)) ** 2
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    nyq = fs / 2.0
    card = p[(f >= lo) & (f <= min(hi, nyq - 0.1))].sum()
    noise = p[(f >= nlo) & (f <= min(nhi, nyq - 0.1))].sum() + 1e-12
    return float(card / noise)


def cancel_motion(ppg: np.ndarray, accel: np.ndarray, fs: float,
                  n_lags: int = 6) -> tuple[np.ndarray, float]:
    """Subtract accelerometer-correlated component via Wiener regression."""
    n = len(ppg)
    if n < int(2.0 * fs) or accel.shape[0] != n:
        return ppg, 0.0

    acc_bp = np.column_stack([_bandpass(accel[:, i], fs, 0.7, 4.0) for i in range(accel.shape[1])])
    motion_level = float(np.mean([_robust_pp(acc_bp[:, i]) for i in range(acc_bp.shape[1])]))

    cols = [np.ones(n)]
    for i in range(acc_bp.shape[1]):
        a = acc_bp[:, i]
        for lag in range(n_lags):
            if lag == 0:
                cols.append(a)
            else:
                shifted = np.zeros(n)
                shifted[lag:] = a[:-lag]
                cols.append(shifted)
    A = np.column_stack(cols)
    scales = np.std(A, axis=0)
    scales[scales < 1e-9] = 1.0
    A_norm = A / scales

    try:
        w, *_ = np.linalg.lstsq(A_norm, ppg, rcond=None)
        cleaned = ppg - A_norm @ w
    except np.linalg.LinAlgError:
        cleaned = ppg

    if np.std(cleaned) >= np.std(ppg) * 1.05:
        cleaned = ppg
    return cleaned, motion_level


@dataclass
class PpgHrProcessor:
    frame_fs: float = 250.0
    mode: PpgMode = PpgMode.MULTILED
    bpm_min: float = 45.0
    bpm_max: float = 200.0
    fft_window_s: float = 8.0
    bpm_smooth: float = 0.75
    min_fft_quality: float = 1.2
    continuity_bpm: float = 20.0
    max_buffer_s: float = 14.0
    use_motion_cancel: bool = True

    _led1: list = field(default_factory=list)
    _led2: list = field(default_factory=list)
    _ax: list = field(default_factory=list)
    _ay: list = field(default_factory=list)
    _az: list = field(default_factory=list)
    _t: list = field(default_factory=list)
    _fs_dynamic: float = 0.0
    _bpm_ema: float = 0.0
    _bpm_initialized: bool = False

    @property
    def fs(self) -> float:
        return self._fs_dynamic if self._fs_dynamic > 0 else self.frame_fs

    def reset(self):
        for buf in (self._led1, self._led2, self._ax, self._ay, self._az, self._t):
            buf.clear()
        self._fs_dynamic = 0.0
        self._bpm_ema = 0.0
        self._bpm_initialized = False

    def push(self, led1, led2, ax=0, ay=0, az=0, ts=None):
        self._led1.append(float(led1))
        self._led2.append(float(led2))
        self._ax.append(float(ax))
        self._ay.append(float(ay))
        self._az.append(float(az))
        self._t.append(time.time() if ts is None else float(ts))
        max_len = int(self.max_buffer_s * max(self.frame_fs, 200.0))
        if len(self._led1) > max_len:
            for buf in (self._led1, self._led2, self._ax, self._ay, self._az, self._t):
                del buf[:-max_len]

    def _resample_uniform(self):
        if len(self._t) < 16:
            return None
        t = np.asarray(self._t, dtype=float)
        t -= t[0]
        span = t[-1]
        if span <= 0:
            return None
        dt = np.diff(t)
        dt = dt[dt > 0]
        if len(dt) < 8:
            return None
        fs = float(np.clip(1.0 / np.median(dt), 5.0, 250.0))
        keep = min(span, self.max_buffer_s)
        grid = np.arange(span - keep, span, 1.0 / fs)
        if len(grid) < 16:
            return None

        def interp(buf):
            return np.interp(grid, t, np.asarray(buf, dtype=float))

        self._fs_dynamic = fs
        return (interp(self._led1), interp(self._led2),
                interp(self._ax), interp(self._ay), interp(self._az), fs)

    def _spectrum(self, x: np.ndarray):
        w = x - np.mean(x)
        n_fft = 1
        while n_fft < 4 * len(w):
            n_fft <<= 1
        spec = np.abs(np.fft.rfft(w * np.hanning(len(w)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.fs)
        return freqs, spec

    def _track_bpm(self, channels, accel_mag: np.ndarray,
                   motion_norm: float = 0.0) -> tuple[float, bool, float]:
        if not isinstance(channels, (list, tuple)):
            channels = [channels]
        win_len = min(min(len(c) for c in channels), int(self.fft_window_s * self.fs))
        if win_len < int(3.0 * self.fs):
            return (self._bpm_ema if self._bpm_initialized else 0.0, False, 0.0)

        freqs = None
        spec = None
        for c in channels:
            w = _despike(c[-win_len:])
            if np.std(w) < 1e-9:
                continue
            fr, sp = self._spectrum(w)
            p = sp ** 2
            if spec is None:
                freqs, spec = fr, p
            else:
                spec += p
        if spec is None:
            return (self._bpm_ema if self._bpm_initialized else 0.0, False, 0.0)
        spec = np.sqrt(spec)

        if accel_mag is not None and len(accel_mag) >= win_len:
            am = _bandpass(accel_mag[-win_len:], self.fs, 0.7, 4.0)
            if np.std(am) > 1e-9:
                _, aspec = self._spectrum(am)
                aspec_norm = aspec / (np.max(aspec) + 1e-12)
                peaks, _ = find_peaks(aspec_norm, height=0.3, distance=2)
                suppress = np.ones_like(spec)
                for p in peaks:
                    suppress[max(0, p - 1):min(len(suppress), p + 2)] = 0.01
                suppress *= (1.0 - 0.5 * np.clip(aspec_norm, 0.0, 1.0))
                spec *= suppress

        mask = (freqs >= self.bpm_min / 60.0) & (freqs <= self.bpm_max / 60.0)
        if not np.any(mask):
            return (self._bpm_ema if self._bpm_initialized else 0.0, False, 0.0)

        band = spec[mask]
        band_freqs = freqs[mask]
        floor = float(np.median(band) + 1e-12)
        score = band.copy()
        if self._bpm_initialized and self._bpm_ema > 0:
            bpm_axis = band_freqs * 60.0
            penalty = np.exp(-0.5 * ((bpm_axis - self._bpm_ema) / self.continuity_bpm) ** 2)
            score = band * (0.5 + 0.5 * penalty)

        idx = int(np.argmax(score))
        quality = float(band[idx] / floor)
        bpm = float(band_freqs[idx] * 60.0)
        if motion_norm > 1.5:
            quality *= 1.5 / motion_norm

        valid = quality >= self.min_fft_quality
        if valid:
            if self._bpm_initialized:
                bpm = self.bpm_smooth * self._bpm_ema + (1.0 - self.bpm_smooth) * bpm
            self._bpm_ema = bpm
            self._bpm_initialized = True
        else:
            bpm = self._bpm_ema if self._bpm_initialized else 0.0
        return bpm, valid, quality

    def _find_beats(self, x: np.ndarray, bpm: float) -> np.ndarray:
        if len(x) < int(1.5 * self.fs) or bpm <= 0:
            return np.array([], dtype=int)
        min_dist = max(int(0.28 * self.fs), int(60.0 / bpm * 0.5 * self.fs))
        prominence = max(_robust_pp(x) * 0.15, 0.03)
        peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence)
        return peaks

    def compute(self) -> PpgHrResult | None:
        resampled = self._resample_uniform()
        if resampled is None:
            return None
        l1, l2, rax, ray, raz, fs = resampled
        if len(l1) < int(1.5 * fs):
            return None

        accel = np.column_stack([rax, ray, raz])
        accel_mag = np.sqrt(np.sum(accel ** 2, axis=1))

        channels_raw = [_detrend_channel(l1, self.fs)]
        if self.mode != PpgMode.ADC_ONLY:
            channels_raw.append(_detrend_channel(l2, self.fs))

        motion_level = 0.0
        channels = []
        for c in channels_raw:
            if self.use_motion_cancel:
                cc, ml = cancel_motion(c, accel, self.fs)
                motion_level = max(motion_level, ml)
                channels.append(cc)
            else:
                channels.append(c)

        motion_norm = motion_level / 512.0
        bpm, valid, quality = self._track_bpm(channels, accel_mag, motion_norm)

        snr_list = [inband_snr(c, self.fs) for c in channels]
        best = int(np.argmax(snr_list)) if snr_list else 0
        best_ch = channels[best]
        inband = float(snr_list[best]) if snr_list else 0.0
        peaks = self._find_beats(best_ch, bpm if valid else self._bpm_ema)

        ac_amp = _robust_pp(best_ch)
        display = best_ch / (ac_amp / 2.0) if ac_amp > 1e-9 else best_ch.copy()
        combined = 0.5 * sum(channels) if len(channels) > 1 else channels[0]

        return PpgHrResult(
            raw_signal=combined,
            raw_led1=l1,
            raw_led2=l2,
            preprocessed=combined,
            filtered=display,
            beat_indices=peaks,
            bpm=bpm,
            bpm_valid=valid,
            fs_effective=self.fs,
            snr=quality,
            ac_amplitude=ac_amp,
            bpm_method="fft+accel" if self.use_motion_cancel else "fft",
            motion_level=motion_norm,
            inband_snr=inband,
        )
