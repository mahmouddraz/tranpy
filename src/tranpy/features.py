"""Physics-informed features of fault-event trajectories.

:class:`TrajectoryFeatures` turns a :class:`~tranpy_core.SignalDataset` of events into a
feature table: rotor-angle and speed dynamics per generator, system voltage recovery,
frequency response, power swings and inter-generator coherency. All events are processed
together with segment reductions (no per-event DataFrames).

The observation window ends ``window_s`` seconds after a *reference* instant per event:

* ``"fault_start"`` - detected fault inception (causal: known when it happens);
* ``"fault_clear"`` - detected fault clearing (**not causal**, see
  :func:`tranpy.datasets.detect_fault_timing`; kept to reproduce earlier results);
* ``"absolute"`` - simulation time ``window_s``.

Every window starts at the beginning of the recording, so pre-fault data provide the
baseline values (subscript 0 below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from tranpy.datasets import FAULT_CLEAR, FAULT_START
from tranpy.vocabulary import ElementKind, Variable, element_kind
from tranpy_core.data import SignalDataset, TabularDataset
from tranpy_core.errors import ConfigurationError, DataError
from tranpy_core.features import crop_to_window
from tranpy_core.features import segments as seg

__all__ = ["TrajectoryFeatures", "WindowReference", "window_end_times"]

WindowReference = Literal["fault_start", "fault_clear", "absolute"]
FloatArray = npt.NDArray[np.float64]

_REFERENCE_ANNOTATION = {"fault_start": FAULT_START, "fault_clear": FAULT_CLEAR}


def window_end_times(
    dataset: SignalDataset, window_s: float, reference: WindowReference
) -> FloatArray:
    """End time of the observation window of every event."""
    if reference == "absolute":
        return np.full(dataset.n_samples, float(window_s))
    key = _REFERENCE_ANNOTATION[reference]
    if key not in dataset.annotations:
        raise DataError(f"dataset has no {key!r} annotation needed for reference={reference!r}")
    ref = np.asarray(dataset.annotations[key])
    if np.isnan(ref).any():
        assert dataset.sample_ids is not None
        bad = dataset.sample_ids[np.isnan(ref)]
        raise DataError(
            f"{bad.size} events have no detected {key} (sample ids {bad[:5].tolist()} ...)"
        )
    return ref + window_s


@dataclass(frozen=True)
class TrajectoryFeatures:
    """Feature extractor for event trajectories.

    Attributes:
        window_s: observation time after the reference instant; None = whole trajectory.
        reference: reference instant of the window (required when ``window_s`` is set).
        include_rotor: per-generator rotor angle, speed and frequency-deviation features.
        include_voltage: system voltage nadir, deviation and recovery features.
        include_frequency: system frequency nadir, RoCoF and spread features.
        include_power: per-generator active/reactive power swing features.
        include_coherency: rotor-angle spread across generators.
        nominal_frequency_hz: fallback frequency baseline when an event has no pre-fault data.
        voltage_recovery_pu: threshold for ``sys_voltage_recovery_time``.
    """

    window_s: float | None = None
    reference: WindowReference | None = None
    include_rotor: bool = True
    include_voltage: bool = True
    include_frequency: bool = True
    include_power: bool = True
    include_coherency: bool = True
    nominal_frequency_hz: float = 50.0
    voltage_recovery_pu: float = 0.9

    def __post_init__(self) -> None:
        problems = []
        if self.window_s is not None:
            if self.window_s < 0:
                problems.append(f"window_s must be >= 0, got {self.window_s}")
            if self.reference is None:
                problems.append(
                    "reference is required when window_s is set "
                    "(fault_start, fault_clear or absolute)"
                )
        if self.reference not in (None, "fault_start", "fault_clear", "absolute"):
            problems.append(f"unknown reference {self.reference!r}")
        if problems:
            raise ConfigurationError("invalid trajectory feature settings:", tuple(problems))

    def transform(self, dataset: SignalDataset) -> TabularDataset:
        """Feature table with one row per event; columns sorted by name."""
        data = dataset
        if self.window_s is not None:
            assert self.reference is not None
            data = crop_to_window(
                dataset, window_end_times(dataset, self.window_s, self.reference), min_steps=3
            )
        features = _Extraction(data, self).run()
        names = tuple(sorted(features))
        return TabularDataset(
            features=np.column_stack([features[n] for n in names])
            if names
            else np.empty((data.n_samples, 0)),
            labels=data.labels,
            feature_names=names,
            label_names=data.label_names,
            name=f"{data.name}_features",
            description=(
                f"trajectory features of {data.name} "
                f"(window_s={self.window_s}, reference={self.reference})"
            ),
            sample_ids=data.sample_ids,
            metadata={"window_s": self.window_s, "reference": self.reference},
        )


class _Extraction:
    """State shared by the feature groups for one (windowed) dataset."""

    def __init__(self, data: SignalDataset, settings: TrajectoryFeatures) -> None:
        self.data = data
        self.settings = settings
        self.t = data.time
        self.off = data.offsets
        self.row_sample = data.sample_index
        fault = np.asarray(
            data.annotations.get(FAULT_START, np.zeros(data.n_samples)), dtype=np.float64
        )
        self.fault = np.where(np.isnan(fault), 0.0, fault)
        self.prefault = self.t < self.fault[self.row_sample]
        self.dt, self.off_d = seg.segment_diff(self.t, self.off)
        self.dt_safe = np.where(self.dt == 0, 1e-6, self.dt)
        self.out: dict[str, FloatArray] = {}
        self.generators: dict[str, dict[str, int]] = {}
        self.buses: dict[str, dict[str, int]] = {}
        for col, channel in enumerate(data.channel_names):
            element, _, variable = channel.partition("|")
            kind = element_kind(element)
            if kind is ElementKind.GENERATOR:
                self.generators.setdefault(element, {})[variable] = col
            elif kind is ElementKind.BUS:
                self.buses.setdefault(element, {})[variable] = col

    def run(self) -> dict[str, FloatArray]:
        s = self.settings
        if s.include_rotor:
            self._rotor()
        if s.include_voltage:
            self._voltage()
        if s.include_frequency:
            self._frequency()
        if s.include_power:
            self._power()
        if s.include_coherency and len(self.generators) > 1:
            self._coherency()
        return self.out

    # -- helpers -------------------------------------------------------------------------
    def _matrix(
        self, elements: dict[str, dict[str, int]], variable: Variable
    ) -> tuple[list[str], FloatArray]:
        names = [e for e, cols in elements.items() if variable.value in cols]
        cols = [elements[e][variable.value] for e in names]
        return names, self.data.values[:, cols]

    def _baseline(self, values: FloatArray) -> FloatArray:
        """Last pre-fault value per event (first value if there is no pre-fault data)."""
        return seg.segment_last_where(values, self.off, self.prefault)

    def _rate(self, values: FloatArray) -> FloatArray:
        diff, _ = seg.segment_diff(values, self.off)
        return diff / (self.dt_safe if values.ndim == 1 else self.dt_safe[:, None])

    @staticmethod
    def _prefix(element: str) -> str:
        return element.replace(" ", "")

    # -- feature groups ------------------------------------------------------------------
    def _rotor(self) -> None:
        names, angle = self._matrix(self.generators, Variable.ROTOR_ANGLE_DEG)
        if names:
            a0 = self._baseline(angle)
            rate = self._rate(angle)
            accel_diff, off_2 = seg.segment_diff(rate, self.off_d)
            keep = np.ones(self.dt_safe.shape[0], dtype=bool)
            keep[self.off_d[1:] - 1] = False
            accel = np.abs(accel_diff / self.dt_safe[keep][:, None])
            delta_max = seg.segment_max(np.abs(angle - a0[self.row_sample]), self.off)
            delta_final = seg.segment_last(angle, self.off) - a0
            rate_max = seg.segment_max(np.abs(rate), self.off_d)
            rate_mean = seg.segment_mean(np.abs(rate), self.off_d)
            accel_max = seg.segment_max(accel, off_2)
            first_swing, damping = self._swings(angle, a0)
            for g, name in enumerate(names):
                p = self._prefix(name)
                self.out[f"{p}_delta_max"] = delta_max[:, g]
                self.out[f"{p}_delta_final"] = delta_final[:, g]
                self.out[f"{p}_delta_rate_max"] = rate_max[:, g]
                self.out[f"{p}_delta_rate_mean"] = rate_mean[:, g]
                self.out[f"{p}_delta_accel_max"] = accel_max[:, g]
                self.out[f"{p}_first_swing_amp"] = first_swing[:, g]
                self.out[f"{p}_damping_ratio"] = damping[:, g]
        names, speed = self._matrix(self.generators, Variable.ROTOR_SPEED_PU)
        if names:
            s0 = self._baseline(speed)
            not_first = np.ones(self.t.shape[0], dtype=bool)
            not_first[self.off[:-1]] = False
            deviation = np.abs(speed[not_first] - s0[self.row_sample[not_first]])
            integral = seg.segment_sum(deviation * self.dt[:, None], self.off_d)
            peak = seg.segment_max(np.abs(speed - s0[self.row_sample]), self.off)
            final = seg.segment_last(speed, self.off) - s0
            for g, name in enumerate(names):
                p = self._prefix(name)
                self.out[f"{p}_speed_dev_peak"] = peak[:, g]
                self.out[f"{p}_speed_dev_final"] = final[:, g]
                self.out[f"{p}_speed_integral"] = integral[:, g]
        names, freq_dev = self._matrix(self.generators, Variable.ROTOR_FREQUENCY_DEVIATION_HZ)
        if names:
            peak = seg.segment_max(np.abs(freq_dev), self.off)
            for g, name in enumerate(names):
                self.out[f"{self._prefix(name)}_freq_dev_max"] = peak[:, g]

    def _swings(self, angle: FloatArray, a0: FloatArray) -> tuple[FloatArray, FloatArray]:
        """First-swing amplitude and damping ratio (second vs first peak) after the fault."""
        n, n_gen = a0.shape
        first = np.zeros((n, n_gen))
        damping = np.ones((n, n_gen))
        for i in range(n):
            lo, hi = int(self.off[i]), int(self.off[i + 1])
            post = angle[lo:hi][~self.prefault[lo:hi]]
            if post.shape[0] <= 3:
                continue
            excursion = np.abs(post - a0[i])
            for g in range(n_gen):
                peaks, _ = find_peaks(excursion[:, g], distance=3)
                if peaks.size == 0:
                    first[i, g] = excursion[:, g].max()
                    continue
                a1 = excursion[peaks[0], g]
                first[i, g] = a1
                if peaks.size >= 2:
                    damping[i, g] = (a1 - excursion[peaks[1], g]) / a1 if a1 > 1e-6 else 0.0
        return first, damping

    def _voltage(self) -> None:
        _, voltage = self._matrix(self.buses, Variable.BUS_VOLTAGE_PU)
        if voltage.shape[1] == 0:
            return
        v0 = self._baseline(voltage)
        deviation = np.abs(voltage - v0[self.row_sample])
        lowest = voltage.min(axis=1)
        post = ~self.prefault
        recovered = seg.first_true_index(
            post & (lowest >= self.settings.voltage_recovery_pu), self.off
        )
        t_end = seg.segment_last(self.t, self.off)
        t_recovered = np.where(
            recovered >= 0, self.t[self.off[:-1] + np.maximum(recovered, 0)], t_end
        )
        post_max = seg.segment_max(np.where(post[:, None], voltage, -np.inf), self.off).max(axis=1)
        self.out["sys_voltage_nadir"] = seg.segment_min(voltage, self.off).min(axis=1)
        self.out["sys_voltage_dev_mean"] = seg.segment_mean(deviation, self.off).mean(axis=1)
        self.out["sys_voltage_dev_max"] = seg.segment_max(deviation, self.off).max(axis=1)
        self.out["sys_voltage_recovery_time"] = t_recovered - self.fault
        self.out["sys_voltage_overshoot"] = post_max - v0.max(axis=1)
        self.out["sys_voltage_final_dev"] = np.abs(seg.segment_last(voltage, self.off) - v0).mean(
            axis=1
        )

    def _frequency(self) -> None:
        _, freq = self._matrix(self.buses, Variable.BUS_FREQUENCY_HZ)
        if freq.shape[1] == 0:
            return
        system = freq.mean(axis=1)
        has_prefault = seg.segment_any(self.prefault, self.off)
        nominal = np.where(has_prefault, self._baseline(system), self.settings.nominal_frequency_hz)
        rocof = np.abs(self._rate(system))
        spread = freq.max(axis=1) - freq.min(axis=1)
        self.out["sys_freq_nadir"] = seg.segment_min(system, self.off)
        self.out["sys_freq_peak"] = seg.segment_max(system, self.off)
        self.out["sys_freq_dev_max"] = seg.segment_max(
            np.abs(system - nominal[self.row_sample]), self.off
        )
        self.out["sys_rocof_max"] = seg.segment_max(rocof, self.off_d)
        self.out["sys_rocof_mean"] = seg.segment_mean(rocof, self.off_d)
        self.out["sys_freq_spread_max"] = seg.segment_max(spread, self.off)
        self.out["sys_freq_spread_mean"] = seg.segment_mean(spread, self.off)
        self.out["sys_freq_final_dev"] = np.abs(seg.segment_last(system, self.off) - nominal)

    def _power(self) -> None:
        names, active = self._matrix(self.generators, Variable.ACTIVE_POWER_MW)
        total_p = np.zeros(self.t.shape[0])
        if names:
            p0 = self._baseline(active)
            swing = seg.segment_max(active, self.off) - seg.segment_min(active, self.off)
            dev = seg.segment_max(np.abs(active - p0[self.row_sample]), self.off)
            for g, name in enumerate(names):
                p = self._prefix(name)
                self.out[f"{p}_power_swing_amp"] = swing[:, g]
                self.out[f"{p}_power_dev_max"] = dev[:, g]
                total_p += active[:, g]
        names, reactive = self._matrix(self.generators, Variable.REACTIVE_POWER_MVAR)
        total_q = np.zeros(self.t.shape[0])
        if names:
            peak = seg.segment_max(np.abs(reactive), self.off)
            for g, name in enumerate(names):
                self.out[f"{self._prefix(name)}_reactive_peak"] = peak[:, g]
                total_q += reactive[:, g]
        self.out["sys_total_power_swing"] = seg.segment_max(total_p, self.off) - seg.segment_min(
            total_p, self.off
        )
        self.out["sys_total_reactive_peak"] = seg.segment_max(np.abs(total_q), self.off)

    def _coherency(self) -> None:
        names, angle = self._matrix(self.generators, Variable.ROTOR_ANGLE_DEG)
        if len(names) < 2:
            return
        spread = angle.max(axis=1) - angle.min(axis=1)
        divergence = np.abs(self._rate(spread))
        coi = angle.mean(axis=1)
        std = angle.std(axis=1)
        spread_max = seg.segment_max(spread, self.off)
        self.out["sys_angle_spread_max"] = spread_max
        self.out["sys_angle_spread_mean"] = seg.segment_mean(spread, self.off)
        self.out["sys_angle_spread_final"] = seg.segment_last(spread, self.off)
        self.out["sys_angle_divergence_rate_max"] = seg.segment_max(divergence, self.off_d)
        self.out["sys_angle_divergence_rate_mean"] = seg.segment_mean(divergence, self.off_d)
        # largest pairwise angle difference == largest spread; kept for continuity of results
        self.out["sys_max_relative_angle"] = spread_max
        self.out["sys_coi_angle_max_dev"] = seg.segment_max(
            np.abs(coi - seg.segment_first(coi, self.off)[self.row_sample]), self.off
        )
        self.out["sys_angle_std_max"] = seg.segment_max(std, self.off)
        self.out["sys_angle_std_final"] = seg.segment_last(std, self.off)
