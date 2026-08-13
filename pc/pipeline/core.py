"""PPG heart-rate processing: filtering, motion cancellation, spectral BPM tracking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

from ml.hr_correction import HrCorrectionModel


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
    ibi_cv: float = float("inf")
    peak_amp_cv: float = float("inf")
    bpm_raw: float = 0.0
    bpm_ml: float | None = None


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
        return (x.copy())
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


def _lagged_design_matrix(acc_bp: np.ndarray, n_lags: int) -> np.ndarray:
    """Intercept column + `n_lags` samples of history per accelerometer axis."""
    n = acc_bp.shape[0]
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
    return np.column_stack(cols)


def cancel_motion(ppg: np.ndarray, acc_bp: np.ndarray, fs: float, n_lags: int = 120,
                  ridge_alpha: float = 0.0, gate: str = "var") -> tuple[np.ndarray, float]:
    """Subtract accelerometer-correlated component via per-window regression.

    n_lags and ridge_alpha were both swept against PPG-DaLiA ground truth
    (pc/ml/sweep_motion_cancel.py), measuring MAE on high-motion windows
    specifically (not just the dataset-wide blend, which hides exactly the
    windows this exists to fix): n_lags=80->120 improved high-motion MAE
    10.62->9.97, n_lags=160 barely helped further (9.81) at a real cost to
    calm-window accuracy (overall MAE 7.19->8.46), and n_lags>=200 collapses
    outright (MAE >12 and climbing fast) as the unregularized fit overfits
    the `3 axes * n_lags` autocorrelated lag columns. Ridge regularization
    was tried as a fix for that overfitting -- it didn't help at any n_lags
    or alpha tested; alpha=0 won every comparison. n_lags=120 is the
    practical ceiling for this approach on real data; getting closer to a
    wearable's true worst-case-while-moving target needs a different
    technique, not just pushing this knob further.

    A from-scratch persistent adaptive filter (NLMS) was also tried, on the
    theory that carrying weights across windows would help -- it didn't:
    both an NLMS variant and a fully causal per-sample version consistently
    *underperformed* no cancellation at all. Root cause in hindsight: accel-
    to-PPG motion coupling isn't stationary across a session (walking vs.
    sitting couple differently), so a filter that blends old coupling
    coefficients with new ones fights against what each window actually
    needs -- a fresh best fit to its own motion characteristics. The batch
    approach's "weakness" (no memory, refit every window) was actually the
    right behavior.

    `gate` picks the accept/reject check for a window's cleaned output:
    "var" (revert if cleaned variance grew >5%) beat "snr" (revert if
    in-band SNR didn't improve) empirically -- the SNR gate was too
    conservative and rejected most genuinely-helpful cleanings.
    """
    n = len(ppg)
    if n < int(2.0 * fs) or acc_bp.shape[0] != n:
        return ppg, 0.0

    motion_level = float(np.mean([_robust_pp(acc_bp[:, i]) for i in range(acc_bp.shape[1])]))

    A = _lagged_design_matrix(acc_bp, n_lags)
    scales = np.std(A, axis=0)
    scales[scales < 1e-9] = 1.0
    A_norm = A / scales

    d = A_norm.shape[1]
    try:
        if ridge_alpha > 0:
            A_fit = np.vstack([A_norm, np.sqrt(ridge_alpha) * np.eye(d)])
            b_fit = np.concatenate([ppg, np.zeros(d)])
        else:
            A_fit, b_fit = A_norm, ppg
        w, *_ = np.linalg.lstsq(A_fit, b_fit, rcond=None)
        cleaned = ppg - A_norm @ w
    except np.linalg.LinAlgError:
        return ppg, motion_level

    if not np.all(np.isfinite(cleaned)):
        return ppg, motion_level
    if gate == "snr" and inband_snr(cleaned, fs) < inband_snr(ppg, fs):
        return ppg, motion_level
    if gate == "var" and np.std(cleaned) >= np.std(ppg) * 1.05:
        return ppg, motion_level
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
    # Higher bar than min_fft_quality specifically for the *first* lock (or
    # re-lock after losing one) -- a weak initial lock gets defended by the
    # continuity penalty even when wrong, causing a slow multi-second drift
    # toward the true value instead of just waiting for a confident one.
    # Swept against PPG-DaLiA ground truth alongside time-to-first-lock
    # (pc/ml/train_hr_correction.py infra): early-window MAE 27.4 -> 10.1
    # BPM at 8.0, for a worst-case lock delay of 26s (vs 5s at 1.2). Higher
    # values (14, 20) improve accuracy further but the delay explodes --
    # 20.0 hit a 701s worst case in the sweep, unacceptable for a wearable.
    lock_fft_quality: float = 8.0
    continuity_bpm: float = 20.0
    max_buffer_s: float = 14.0
    use_motion_cancel: bool = True
    motion_n_lags: int = 120
    motion_ridge_alpha: float = 0.0
    motion_gate: str = "var"
    # Confidence-weighted EMA blending, validated against PPG-DaLiA ground
    # truth: MAE 9.90 -> 9.15 BPM, high-motion-window MAE 13.27 -> 12.43
    # (pc/ml/train_hr_correction.py sweep infra). A barely-valid estimate
    # should barely move the tracked BPM; fixed-weight smoothing let
    # low-confidence motion-corrupted readings drag it around just as much
    # as a confident one.
    adaptive_smoothing: bool = True
    quality_ref: float = 8.0
    # Tried sharpening this (confidence**power + a relative-to-recent-
    # baseline penalty) to fight a periodic ~10s signal-quality dip
    # (respiratory/vasomotor modulation of PPG amplitude, confirmed via
    # periodogram against a real recording's raw ADC trace) that was
    # compounding into a 24 BPM reported swing. It worked for that resting
    # scenario, but swept against PPG-DaLiA ground truth (activities with
    # real HR transitions, not just resting) it made overall MAE strictly
    # worse the harder it damped (10.59 -> 12.45 BPM from power 1 -> 6) --
    # extra damping in the core tracker trades real responsiveness for
    # resting-state smoothness everywhere, not just during a dip. Handling
    # this in the UI instead (MainWindow's watch-style trailing-average BPM
    # curve, separate from the instant reading) gets the smoothing without
    # that tradeoff.
    # Require detected peaks to actually be periodic/consistent (not just a
    # confident-looking FFT peak) before trusting a window enough to update
    # the tracked BPM -- same lesson as BoostOptimizer's SNR-can-be-fooled
    # fix, applied to the tracker itself. Catches e.g. a motion/contact-loss
    # wobble that has strong in-band spectral power but isn't a real pulse.
    # Validated against PPG-DaLiA ground truth (pc/ml/train_hr_correction.py
    # sweep infra): MAE 9.146 -> 9.138, median 5.239 -> 5.106, early-window
    # (cold-start) MAE 27.4 -> 22.5 -- no regression on any metric.
    regularity_gate: bool = True
    max_ibi_cv: float = 0.35
    max_amp_cv: float = 1.0
    correction_model: HrCorrectionModel | None = None
    use_ml_correction: bool = True

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

        required_quality = self.min_fft_quality if self._bpm_initialized else self.lock_fft_quality
        spectrally_valid = quality >= required_quality
        return bpm, spectrally_valid, quality

    def _commit_bpm(self, bpm: float, quality: float) -> float:
        """Blend a validated candidate into the tracked EMA. Only called for
        windows that passed every gate (spectral quality, and -- when
        regularity_gate is on -- peak regularity); a rejected window must
        never reach here, or a fake-but-confident reading could still
        corrupt the tracked value."""
        if self._bpm_initialized:
            if self.adaptive_smoothing:
                # A barely-valid (low-quality) estimate should barely move
                # the tracked value; let confident estimates move it more.
                # Fixed smoothing blends both the same way, which lets
                # low-confidence motion-corrupted readings drag the EMA
                # around while still reporting "valid".
                confidence = float(np.clip(quality / self.quality_ref, 0.0, 1.0))
                alpha = confidence * (1.0 - self.bpm_smooth)
                bpm = (1.0 - alpha) * self._bpm_ema + alpha * bpm
            else:
                bpm = self.bpm_smooth * self._bpm_ema + (1.0 - self.bpm_smooth) * bpm
        self._bpm_ema = bpm
        self._bpm_initialized = True
        return bpm

    def _find_beats(self, x: np.ndarray, bpm: float) -> np.ndarray:
        if len(x) < int(1.5 * self.fs) or bpm <= 0:
            return np.array([], dtype=int)
        min_dist = max(int(0.28 * self.fs), int(60.0 / bpm * 0.5 * self.fs))
        prominence = max(_robust_pp(x) * 0.15, 0.03)
        peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence)
        return peaks

    def _beat_regularity(self, peaks: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        """Coefficient of variation of inter-beat intervals and peak
        amplitudes. A real, cleanly-coupled pulse is periodic and
        consistent; a strong-but-fake in-band SNR (e.g. a distorted or
        motion-driven signal) usually isn't -- this is what lets callers
        (BoostOptimizer) tell the two apart instead of trusting SNR alone."""
        if len(peaks) < 4:
            return float("inf"), float("inf")
        ibi = np.diff(peaks) / self.fs
        ibi_cv = float(np.std(ibi) / np.mean(ibi)) if np.mean(ibi) > 0 else float("inf")
        heights = x[peaks]
        amp_cv = float(np.std(heights) / abs(np.mean(heights))) if np.mean(heights) != 0 else float("inf")
        return ibi_cv, amp_cv

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
        if self.use_motion_cancel:
            acc_bp = np.column_stack(
                [_bandpass(accel[:, i], self.fs, 0.7, 4.0) for i in range(accel.shape[1])])
            for c in channels_raw:
                cc, ml = cancel_motion(c, acc_bp, self.fs, n_lags=self.motion_n_lags,
                                       ridge_alpha=self.motion_ridge_alpha, gate=self.motion_gate)
                motion_level = max(motion_level, ml)
                channels.append(cc)
        else:
            channels = channels_raw

        motion_norm = motion_level / 512.0
        cand_bpm, spectrally_valid, quality = self._track_bpm(channels, accel_mag, motion_norm)

        snr_list = [inband_snr(c, self.fs) for c in channels]
        best = int(np.argmax(snr_list)) if snr_list else 0
        best_ch = channels[best]
        inband = float(snr_list[best]) if snr_list else 0.0

        probe_bpm = cand_bpm if spectrally_valid else (self._bpm_ema if self._bpm_initialized else 0.0)
        peaks = self._find_beats(best_ch, probe_bpm)
        ibi_cv, peak_amp_cv = self._beat_regularity(peaks, best_ch)

        if self.regularity_gate:
            regular = len(peaks) >= 4 and ibi_cv <= self.max_ibi_cv and peak_amp_cv <= self.max_amp_cv
            valid = spectrally_valid and regular
        else:
            valid = spectrally_valid

        bpm_raw = self._commit_bpm(cand_bpm, quality) if valid else (
            self._bpm_ema if self._bpm_initialized else 0.0)

        ac_amp = _robust_pp(best_ch)
        display = best_ch / (ac_amp / 2.0) if ac_amp > 1e-9 else best_ch.copy()
        combined = 0.5 * sum(channels) if len(channels) > 1 else channels[0]

        method = "fft+accel" if self.use_motion_cancel else "fft"
        bpm_ml = None
        if self.correction_model is not None and valid:
            corrected = self.correction_model.predict({
                "fft_bpm": bpm_raw,
                "quality": quality,
                "inband_snr": inband,
                "motion_level": motion_norm,
                "ac_amplitude": ac_amp,
            })
            # A feature landing outside the training distribution (e.g. an
            # accelerometer/gain scale the model never saw) can otherwise
            # send a linear model's output arbitrarily far off -- never let
            # the correction override the FFT estimate by more than a
            # plausible nudge, or leave the physiological BPM range.
            if abs(corrected - bpm_raw) <= 25.0 and self.bpm_min <= corrected <= self.bpm_max:
                bpm_ml = corrected

        # bpm_ml is always computed (when available) so the UI can plot raw
        # vs. ML side by side even when the correction isn't the one driving
        # the tracker/display -- use_ml_correction only picks which feeds `bpm`.
        if bpm_ml is not None and self.use_ml_correction:
            bpm = bpm_ml
            method += "+ml"
        else:
            bpm = bpm_raw

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
            bpm_method=method,
            motion_level=motion_norm,
            inband_snr=inband,
            bpm_raw=bpm_raw,
            bpm_ml=bpm_ml,
            ibi_cv=ibi_cv,
            peak_amp_cv=peak_amp_cv,
        )
