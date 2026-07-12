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
    gain_max: int = 255
    ads_min: int = 1
    ads_max: int = 5

    led_step: int = 8
    gain_step: int = 12

    _cooldown: int = 0

    def sync(self, led: int, gain_tia: int, ads_gain: int):
        self.led = int(led)
        self.gain_tia = int(gain_tia)
        self.ads_gain = int(ads_gain)

    def _clamp(self):
        self.led = int(np.clip(self.led, self.led_min, self.led_max))
        self.gain_tia = int(np.clip(self.gain_tia, self.gain_min, self.gain_max))
        self.ads_gain = int(np.clip(self.ads_gain, self.ads_min, self.ads_max))

    def step(self, raw_window: np.ndarray) -> tuple[list[str], str]:
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

        elif ac < AC_TARGET_MIN:
            headroom = min(ADC_FULL_SCALE - p_hi, p_lo)
            if headroom > AC_TARGET_MIN and self.gain_tia < self.gain_max:
                self.gain_tia += self.gain_step
                reason = "low AC: raise TIA gain"
            elif self.led < self.led_max:
                self.led += self.led_step
                reason = "low AC: raise LED"

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

        self._cooldown = 3
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

    def step_quality(self, quality: float, valid: bool, motion: float) -> tuple[int | None, str]:
        usable = valid and quality > 0 and motion < 0.35

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
