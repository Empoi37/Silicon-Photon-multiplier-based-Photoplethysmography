"""Automatic gain control and SiPM overvoltage (BOOST) optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ADC_FULL_SCALE = 32767.0
ADC_SAT_HIGH = 0.90 * ADC_FULL_SCALE
ADC_SAT_LOW = 0.03 * ADC_FULL_SCALE
AC_TARGET_MIN = 40.0


@dataclass
class AutoTuner:
    """Adjusts LED brightness, TIA gain, and ADC gain to keep the signal in range."""

    led: int = 48
    gain_tia: int = 128
    ads_gain: int = 0

    led_min: int = 4
    led_max: int = 255
    gain_min: int = 8
    # MCP4531 is a 7-bit digital pot (0-128, matches hardware_panel's GAIN
    # slider) -- was 255, which let this drive GAIN commands the firmware
    # would silently clamp/wrap past the real ceiling. Once TIA gain stalled
    # there, low-AC kept escalating the LED to its max too.
    gain_max: int = 128
    # ADSGAIN's UI/firmware-valid range is 0-5 (default 0) -- was 1, which
    # made _clamp() silently bump ADSGAIN 0->1 on the first tick after every
    # connect even when nothing was actually out of range.
    ads_min: int = 0
    ads_max: int = 5

    led_step: int = 8
    # Was 12, then 6 -- a real bench session showed even 6 taps was enough
    # to overshoot the useful TIA gain window in one hop and briefly drop
    # BPM to NaN while the tracker lost lock on the jump. AutoTuner is meant
    # to track small drift from an already-good preset, not sweep the full
    # range (that's what the connect-time manual preset is for), so a
    # 1-tap nudge per tick is plenty -- it just takes a couple more 4s
    # verify cycles to get there, and it never overshoots past the target.
    gain_step: int = 1

    _cooldown: int = 0
    # Tracks a pending "raised LED/gain because AC looked low" step so the
    # *next* call can check whether it actually helped. On this SiPM, more
    # light doesn't always mean more AC -- past a point it saturates/piles
    # up and modulation depth collapses instead of growing (confirmed on a
    # real recording: AC shrank from ~470 to ~3 counts as LED climbed to
    # max). Without this check the tuner can't tell escalation is
    # backfiring and just keeps pushing to the ceiling.
    _pending_state: tuple | None = field(default=None, repr=False)
    _pending_ac: float = field(default=-1.0, repr=False)
    _pending_snr: float = field(default=-1.0, repr=False)
    # "low_ac" (verified against _pending_ac) or "relax" (verified against
    # AC_TARGET_MIN/near_sat directly) -- the two paths that leave a pending
    # change need different success criteria, see step().
    _pending_kind: str = field(default="", repr=False)
    _verify_samples: list = field(default_factory=list, repr=False)
    _low_ac_backoff: int = field(default=0, repr=False)
    # A step whose true effect is smaller than tick-to-tick AC noise (from
    # breathing, motion, contact pressure) would otherwise revert forever in
    # a tight retry loop -- exactly the "trying -- not better -- back to
    # value -- trying" flutter reported from the bench. After two failed
    # attempts in a row, treat the gain as converged and back off for much
    # longer before reprobing (same philosophy as BoostOptimizer's
    # reversal-counted convergence).
    _low_ac_fail_streak: int = field(default=0, repr=False)
    retry_backoff_ticks: int = 20        # ~10s @ 500ms: retry after one miss
    converged_backoff_ticks: int = 240   # ~2min: stop hunting after two misses
    # Require the AC gain to clear noise by a margin before counting a step
    # as "improved" -- a strict `ac > previous_ac` comparison treats every
    # sub-percent jitter as a failed step and reverts it, which is itself a
    # source of the oscillation (the step often *did* help; the noise on
    # the very next window just briefly masked it).
    ac_improve_margin: float = 0.05
    # A bigger AC swing (p99-p1) that comes with meaningfully *worse*
    # in-band SNR is the same SiPM-saturation trap documented above under
    # led_step (more light -> more raw amplitude without the pulse actually
    # getting cleaner) -- allow some slack since SNR is noisier window to
    # window than AC, but don't let a step through if it tanked SNR by more
    # than this fraction.
    snr_regress_margin: float = 0.10
    # tools/sweep_preset.py's own settling analysis warns TIA RC settling
    # time grows with gain, and even its 1.0s default --settle isn't always
    # enough at high gain -- judging "did this help" right after the normal
    # 3-tick (1.5s) cooldown risks measuring mid-transient. settle_ticks lets
    # the hardware physically settle first; measure_ticks then collects that
    # many samples and compares their *median* against the pre-change
    # baseline instead of trusting a single before/after snapshot -- a
    # single window is noisy (breathing, contact micro-shifts), and now that
    # "keep" requires AC *and* SNR *and* regularity to all agree, a single
    # bad sample on any one of them would revert a change that was actually
    # fine. Same measure-then-median approach as BoostOptimizer below.
    # Total wait (8 ticks = 4s) matches the old single-sample verify_ticks.
    settle_ticks: int = 3
    measure_ticks: int = 5
    # Gate the "low AC -> raise gain" optimization on the tracker NOT
    # already having a clean lock -- a bench recording with the sensor
    # pressed hard showed AC drop well under AC_TARGET_MIN (occlusion
    # reducing pulsatile amplitude) while bpm_valid still held ~90% of the
    # time, yet the tuner spent 50s creeping gain 96->100 chasing a target
    # that didn't need chasing. Raising gain can't undo an occluded pulse
    # anyway -- it just risks amplifying noise into a working signal.
    # Saturation/floor correction below is NOT gated by this: real clipping
    # is worth fixing even if the tracker currently looks valid (a clipped
    # window can still spuriously line up into a "valid"-looking beat).
    max_ibi_cv: float = 0.35
    max_amp_cv: float = 1.0

    # Connect-time preset, captured in sync(). A defensive move away from
    # this (LED cut for saturation, gain raised for low AC) has no built-in
    # way back -- confirmed on a bench test that loosened then re-tightened
    # the watch: LED correctly dropped for the saturation spike while loose,
    # but once tightened again nothing ever raised it back or lowered the
    # gain that got raised alongside it, because the low-AC branch always
    # preferred gain (which kept "working") over LED, and there was no path
    # at all for backing off once the signal was already healthy again.
    _baseline_led: int = field(default=-1, repr=False)
    _baseline_gain_tia: int = field(default=-1, repr=False)
    _relax_backoff: int = field(default=0, repr=False)
    relax_backoff_ticks: int = 40        # ~20s @ 500ms between relax nudges
    # Require AC comfortably above target (not just barely over) before
    # trying to relax gain/LED back -- avoids immediately re-triggering the
    # low-AC branch right at the boundary.
    ac_relax_margin: float = 1.5

    def sync(self, led: int, gain_tia: int, ads_gain: int):
        self.led = int(led)
        self.gain_tia = int(gain_tia)
        self.ads_gain = int(ads_gain)
        self._baseline_led = self.led
        self._baseline_gain_tia = self.gain_tia
        self._pending_state = None
        self._pending_ac = -1.0
        self._pending_snr = -1.0
        self._pending_kind = ""
        self._verify_samples = []
        self._low_ac_backoff = 0
        self._low_ac_fail_streak = 0
        self._relax_backoff = self.relax_backoff_ticks

    def _clamp(self):
        self.led = int(np.clip(self.led, self.led_min, self.led_max))
        self.gain_tia = int(np.clip(self.gain_tia, self.gain_min, self.gain_max))
        self.ads_gain = int(np.clip(self.ads_gain, self.ads_min, self.ads_max))

    def _regularity_ok(self, ibi_cv: float, amp_cv: float) -> bool:
        """True if peak regularity is within bounds, or not measurable yet
        (ibi_cv/amp_cv come back as inf when there aren't enough detected
        peaks -- e.g. right after a change, before the tracker has settled).
        Inconclusive shouldn't block a verification that otherwise looks
        fine; only a *confirmed* irregular signal should."""
        if not np.isfinite(ibi_cv) or not np.isfinite(amp_cv):
            return True
        return ibi_cv <= self.max_ibi_cv and amp_cv <= self.max_amp_cv

    def step(self, raw_window: np.ndarray, valid: bool = True,
             ibi_cv: float = 0.0, amp_cv: float = 0.0,
             inband_snr: float = 0.0) -> tuple[list[str], str]:
        cmds: list[str] = []
        if raw_window is None or len(raw_window) < 16:
            return cmds, ""
        if self._cooldown > 0:
            self._cooldown -= 1
            return cmds, ""

        dc = float(np.median(raw_window))
        p_hi = float(np.percentile(raw_window, 99))
        p_lo = float(np.percentile(raw_window, 1))
        ac = p_hi - p_lo
        near_sat = float(np.mean(raw_window > ADC_SAT_HIGH))
        near_floor = float(np.mean(raw_window < ADC_SAT_LOW))

        # Check whether the last pending step actually helped before doing
        # anything else this cycle. "low_ac" and "relax" have different
        # success criteria: a low-AC raise must show a real AC gain over
        # what it was before, while a relax step just needs to confirm it
        # didn't quietly break anything (AC still usable, no new clipping).
        # Both also now require SNR/regularity to not have gotten worse --
        # AC amplitude alone can't tell a genuinely cleaner pulse from a
        # louder-but-noisier one (a visually "thin", clean trace is what
        # SNR + low ibi_cv/amp_cv actually measure; raw p99-p1 doesn't).
        #
        # Collect measure_ticks samples and compare their *median* rather
        # than a single before/after snapshot -- with three metrics now
        # required to agree, a single noisy tick on any one of them would
        # revert a change that was actually fine.
        if self._pending_state is not None:
            self._verify_samples.append((ac, inband_snr, ibi_cv, amp_cv, near_sat))
            if len(self._verify_samples) < self.measure_ticks:
                return cmds, ""
            med_ac, med_snr, med_ibi, med_amp, med_near_sat = np.median(
                np.array(self._verify_samples, dtype=float), axis=0)
            self._verify_samples = []

            if self._pending_kind == "relax":
                ok = (med_ac >= AC_TARGET_MIN and med_near_sat <= 0.02
                     and self._regularity_ok(med_ibi, med_amp))
            else:
                ac_ok = med_ac > self._pending_ac * (1.0 + self.ac_improve_margin)
                snr_ok = (self._pending_snr <= 0
                         or med_snr >= self._pending_snr * (1.0 - self.snr_regress_margin))
                ok = ac_ok and snr_ok and self._regularity_ok(med_ibi, med_amp)

            if not ok:
                kind = self._pending_kind
                self.led, self.gain_tia, self.ads_gain = self._pending_state
                self._pending_state = None
                self._pending_ac = -1.0
                self._pending_snr = -1.0
                self._pending_kind = ""
                self._cooldown = 3
                if kind == "relax":
                    self._relax_backoff = self.converged_backoff_ticks
                    return ([f"LED1:{self.led}", f"LED2:{self.led}", f"GAIN:{self.gain_tia}"],
                           "relax step needed after all -- reverting")
                self._low_ac_fail_streak += 1
                converged = self._low_ac_fail_streak >= 2
                self._low_ac_backoff = (
                    self.converged_backoff_ticks if converged else self.retry_backoff_ticks)
                msg = ("low AC: converged, giving up for now"
                       if converged else "low AC didn't improve -- reverting LED/gain")
                return ([f"LED1:{self.led}", f"LED2:{self.led}", f"GAIN:{self.gain_tia}"], msg)

            # A relax step that verified fine keeps climbing/lowering right
            # away (just the normal settle+measure wait below, same pace
            # as low-AC) -- only a step that turned out unsafe should wait
            # the long converged_backoff before trying again. Otherwise
            # restoring LED/gain across several steps back to baseline
            # takes minutes instead of seconds once it's clearly safe to.
            if self._pending_kind != "relax":
                self._low_ac_fail_streak = 0
            self._pending_state = None
            self._pending_ac = -1.0
            self._pending_snr = -1.0
            self._pending_kind = ""

        if self._low_ac_backoff > 0:
            self._low_ac_backoff -= 1
        if self._relax_backoff > 0:
            self._relax_backoff -= 1

        before = (self.led, self.gain_tia, self.ads_gain)
        reason = ""

        if near_sat > 0.02 or dc > ADC_SAT_HIGH:
            if self.led > self.led_min:
                self.led -= self.led_step
                reason = "saturation: lower LED"
            elif self.ads_gain > self.ads_min:
                self.ads_gain -= 1
                reason = "saturation: lower ADC gain"
            elif self.gain_tia > self.gain_min:
                self.gain_tia -= self.gain_step
                reason = "saturation: lower TIA gain"

        elif near_floor > 0.02 or dc < ADC_SAT_LOW:
            if self.led < self.led_max:
                self.led += self.led_step
                reason = "weak signal: raise LED"
            elif self.gain_tia < self.gain_max:
                self.gain_tia += self.gain_step
                reason = "weak signal: raise TIA gain"
            elif self.ads_gain < self.ads_max:
                self.ads_gain += 1
                reason = "weak signal: raise ADC gain"

        elif ac < AC_TARGET_MIN and self._low_ac_backoff == 0 and not (
                valid and ibi_cv <= self.max_ibi_cv and amp_cv <= self.max_amp_cv):
            headroom = min(ADC_FULL_SCALE - p_hi, p_lo)
            # LED first, mirroring the saturation/floor branches above (and
            # unlike the old gain-first order): low AC is very often the
            # mirror image of an earlier saturation-driven LED cut -- loose
            # contact let ambient light in, LED got dropped, then tightening
            # again removes that ambient light but nothing raised LED back,
            # only gain, which kept "working" (genuinely helps once contact
            # is good) and so this branch never fell through to LED at all.
            if self.led < self.led_max and headroom > AC_TARGET_MIN:
                self.led += self.led_step
                reason = "low AC: raise LED"
            elif self.gain_tia < self.gain_max:
                self.gain_tia += self.gain_step
                reason = "low AC: raise TIA gain"
            if reason:
                self._pending_state = before
                self._pending_ac = ac
                self._pending_snr = inband_snr
                self._pending_kind = "low_ac"

        elif (self._relax_backoff == 0 and near_sat <= 0.02 and near_floor <= 0.02
              and ac >= AC_TARGET_MIN * self.ac_relax_margin
              and valid and ibi_cv <= self.max_ibi_cv and amp_cv <= self.max_amp_cv
              and (self.led < self._baseline_led or self.gain_tia > self._baseline_gain_tia)):
            # Conditions are comfortably good and settings are still away
            # from the connect-time preset -- ease one step back toward it.
            # Without this, a temporary excursion (loosen then re-tighten)
            # leaves the tuner stuck whatever it last settled on forever,
            # long after the original trigger is gone.
            if self.led < self._baseline_led:
                self.led = min(self.led + self.led_step, self._baseline_led)
                reason = "conditions improved: restore LED toward preset"
            elif self.gain_tia > self._baseline_gain_tia:
                self.gain_tia = max(self.gain_tia - self.gain_step, self._baseline_gain_tia)
                reason = "conditions improved: relax TIA gain toward preset"
            if reason:
                self._pending_state = before
                self._pending_kind = "relax"

        self._clamp()
        after = (self.led, self.gain_tia, self.ads_gain)
        if after == before:
            return cmds, ""

        if after[0] != before[0]:
            cmds.extend((f"LED1:{self.led}", f"LED2:{self.led}"))
        if after[1] != before[1]:
            cmds.append(f"GAIN:{self.gain_tia}")
        if after[2] != before[2]:
            cmds.append(f"ADSGAIN:{self.ads_gain}")

        self._cooldown = self.settle_ticks if self._pending_state is not None else 3
        return cmds, reason


@dataclass
class BoostOptimizer:
    """Hill-climbing search for the SiPM overvoltage (BOOST) that maximizes in-band SNR."""

    boost: int = 63
    boost_min: int = 16
    boost_max: int = 127
    step: int = 6
    quality_margin: float = 0.08
    settle_ticks: int = 6
    measure_ticks: int = 6
    reprobe_ticks: int = 60
    # Hard gate on peak regularity, same philosophy as tools/sweep_preset.py's
    # "trustworthy" check: a distorted/motion-driven signal can post a huge
    # in-band SNR despite being unusable (confirmed on a real boost sweep --
    # one point hit SNR=160, the best in the sweep, off just 14 irregular,
    # inconsistent-amplitude peaks). Without this, the optimizer chases SNR
    # spikes that don't correspond to a clean pulse.
    max_ibi_cv: float = 0.35
    max_amp_cv: float = 1.0

    _phase: str = "measure"
    _direction: int = 1
    _ticks: int = 0
    _samples: list = field(default_factory=list)
    _best_quality: float = -1.0
    _best_boost: int = 63
    _reversals: int = 0
    _converged_wait: int = 0

    def sync(self, boost: int):
        self.boost = int(boost)
        self._best_boost = int(boost)
        self._best_quality = -1.0
        self._phase = "measure"
        self._direction = 1
        self._ticks = 0
        self._reversals = 0
        self._samples.clear()

    def _clamp(self):
        self.boost = int(np.clip(self.boost, self.boost_min, self.boost_max))

    def _probe(self) -> int:
        self.boost = self._best_boost + self._direction * self.step
        self._clamp()
        self._phase = "settle"
        self._ticks = 0
        return self.boost

    def step_quality(self, quality: float, valid: bool, motion: float,
                     ibi_cv: float = 0.0, amp_cv: float = 0.0) -> tuple[int | None, str]:
        usable = (valid and quality > 0 and motion < 0.35
                 and ibi_cv <= self.max_ibi_cv and amp_cv <= self.max_amp_cv)

        if self._phase == "settle":
            self._ticks += 1
            if self._ticks >= self.settle_ticks:
                self._phase = "measure"
                self._ticks = 0
                self._samples.clear()
            return None, ""

        if self._phase == "converged":
            self._converged_wait += 1
            if self._converged_wait >= self.reprobe_ticks:
                self._converged_wait = 0
                self._phase = "measure"
                self._best_quality = -1.0
                self._reversals = 0
                self._samples.clear()
            return None, ""

        if usable:
            self._samples.append(float(quality))
        if len(self._samples) < self.measure_ticks:
            return None, ""

        q = float(np.median(self._samples))
        self._samples.clear()

        if self._best_quality < 0:
            self._best_quality = q
            self._best_boost = self.boost
            return self._probe(), f"exploring BOOST -> {self.boost}"

        if q > self._best_quality * (1.0 + self.quality_margin):
            self._best_quality = q
            self._best_boost = self.boost
            self._reversals = 0
            return self._probe(), f"SNR improved ({q:.1f}) -> BOOST {self.boost}"

        self._reversals += 1
        self._direction *= -1
        if self._reversals >= 2:
            self.boost = self._best_boost
            self._clamp()
            self._phase = "converged"
            self._converged_wait = 0
            return self.boost, f"optimal BOOST ~{self.boost} (SNR {self._best_quality:.1f})"

        return self._probe(), f"no SNR gain -> try BOOST {self.boost}"
