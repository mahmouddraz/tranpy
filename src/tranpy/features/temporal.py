"""Temporal feature extraction for power system trajectory analysis.

Extracts physics-informed features from time series event trajectories,
including rotor dynamics, voltage recovery, frequency response, and
inter-generator coherency metrics.

These features can be used with any sklearn-compatible classifier,
and support windowed extraction for early prediction analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..datasets.temporal_loaders import (
    TemporalEventData,
    TemporalDataset,
    get_generator_columns,
    get_bus_columns,
)


class TemporalFeatureExtractor:
    """Extract physics-informed features from event trajectories.

    Computes rotor dynamics, voltage recovery, frequency response,
    power flow, and system coherency features from time series data.

    Args:
        time_window: Observation time in seconds after fault clearing
            (None = full trajectory). When relative_to_clearing=True (default),
            this is measured from the fault clearing time of each event.
            When False, this is an absolute time cutoff.
        relative_to_clearing: If True, time_window is measured relative to
            fault clearing time (per-event). If False, uses absolute time.
        include_rotor: Include rotor dynamics features
        include_voltage: Include voltage recovery features
        include_frequency: Include frequency response features
        include_power: Include power flow features
        include_coherency: Include inter-generator coherency features

    Examples:
        >>> # Extract features from 2 seconds of post-clearing observation
        >>> extractor = TemporalFeatureExtractor(time_window=2.0)
        >>> features = extractor.extract(event)

        >>> # Extract features for all events as a matrix
        >>> X, names = extractor.transform(dataset)
        >>> print(X.shape)
        (100, 85)
    """

    def __init__(
        self,
        time_window: Optional[float] = None,
        relative_to_clearing: bool = True,
        include_rotor: bool = True,
        include_voltage: bool = True,
        include_frequency: bool = True,
        include_power: bool = True,
        include_coherency: bool = True,
    ):
        self.time_window = time_window
        self.relative_to_clearing = relative_to_clearing
        self.include_rotor = include_rotor
        self.include_voltage = include_voltage
        self.include_frequency = include_frequency
        self.include_power = include_power
        self.include_coherency = include_coherency
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> Optional[List[str]]:
        """Return feature names after first extraction."""
        return self._feature_names

    def extract(self, event: TemporalEventData) -> Dict[str, float]:
        """Extract all temporal features from a single event.

        When relative_to_clearing=True, features are computed from the full
        trajectory up to (clearing_time + time_window). This captures pre-fault
        baseline, fault dynamics, and post-clearing observation.

        Args:
            event: TemporalEventData object

        Returns:
            Dictionary mapping feature names to values
        """
        df = event.trajectory
        time = event.time

        # Apply time window
        if self.time_window is not None:
            if self.relative_to_clearing and event.clearing_time is not None:
                # Window = everything up to clearing_time + observation_time
                end_time = event.clearing_time + self.time_window
                mask = time <= end_time
            else:
                # Absolute time cutoff (legacy behavior)
                mask = time <= self.time_window
            df = df[mask]
            time = time[mask]

        # Get pre-fault snapshot (t < fault_time, or t < 0 as fallback)
        fault_t = event.fault_time if event.fault_time is not None else 0.0
        prefault_mask = time < fault_t
        postfault_mask = time >= fault_t

        features = {}

        gen_cols = get_generator_columns(df)
        bus_cols = get_bus_columns(df)

        if self.include_rotor:
            features.update(self._extract_rotor_features(df, time, gen_cols, prefault_mask))

        if self.include_voltage:
            features.update(self._extract_voltage_features(df, time, bus_cols, prefault_mask))

        if self.include_frequency:
            features.update(self._extract_frequency_features(df, time, bus_cols, prefault_mask))

        if self.include_power:
            features.update(self._extract_power_features(df, time, gen_cols, prefault_mask))

        if self.include_coherency and len(gen_cols) > 1:
            features.update(self._extract_coherency_features(df, time, gen_cols, prefault_mask))

        return features

    def transform(
        self,
        dataset: TemporalDataset
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from all events in a dataset.

        Args:
            dataset: TemporalDataset with multiple events

        Returns:
            Tuple of (feature_matrix, feature_names)
            feature_matrix shape: (n_events, n_features)
        """
        all_features = []
        for event in dataset.events:
            feat_dict = self.extract(event)
            all_features.append(feat_dict)

        # Build feature matrix
        if not all_features:
            return np.empty((0, 0)), []

        self._feature_names = sorted(all_features[0].keys())
        X = np.array([
            [feat_dict[name] for name in self._feature_names]
            for feat_dict in all_features
        ])

        return X, self._feature_names

    # ---------------------------------------------------------------
    # Rotor dynamics features
    # ---------------------------------------------------------------

    def _extract_rotor_features(
        self,
        df: pd.DataFrame,
        time: np.ndarray,
        gen_cols: Dict[str, List[str]],
        prefault_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Extract rotor angle and speed features per generator."""
        features = {}

        for gen_id, cols in gen_cols.items():
            # Find the rotor angle column (c:firot)
            firot_col = _find_col(cols, 'c:firot')
            speed_col = _find_col(cols, 's:xspeed')
            dfrot_col = _find_col(cols, 'c:dfrotx')
            dorhz_col = _find_col(cols, 'c:dorhz')

            prefix = gen_id.replace(' ', '')

            # --- Rotor angle features ---
            if firot_col:
                angle = df[firot_col].values
                angle_prefault = angle[prefault_mask]
                angle_0 = angle_prefault[-1] if len(angle_prefault) > 0 else angle[0]

                # Max rotor angle excursion from pre-fault
                features[f'{prefix}_delta_max'] = float(np.max(np.abs(angle - angle_0)))

                # Rotor angle at end of window
                features[f'{prefix}_delta_final'] = float(angle[-1] - angle_0)

                # Rate of change of rotor angle (numerical derivative)
                dt = np.diff(time)
                dt[dt == 0] = 1e-6  # avoid division by zero
                dangle = np.diff(angle) / dt
                features[f'{prefix}_delta_rate_max'] = float(np.max(np.abs(dangle)))
                features[f'{prefix}_delta_rate_mean'] = float(np.mean(np.abs(dangle)))

                # Second derivative (angular acceleration)
                if len(dangle) > 1:
                    dt2 = dt[:-1]
                    dt2[dt2 == 0] = 1e-6
                    d2angle = np.diff(dangle) / dt2
                    features[f'{prefix}_delta_accel_max'] = float(np.max(np.abs(d2angle)))
                else:
                    features[f'{prefix}_delta_accel_max'] = 0.0

                # First swing amplitude (first local max after fault)
                postfault_angle = angle[~prefault_mask]
                if len(postfault_angle) > 3:
                    peaks, _ = find_peaks(np.abs(postfault_angle - angle_0), distance=3)
                    if len(peaks) > 0:
                        features[f'{prefix}_first_swing_amp'] = float(
                            np.abs(postfault_angle[peaks[0]] - angle_0)
                        )
                    else:
                        features[f'{prefix}_first_swing_amp'] = float(
                            np.max(np.abs(postfault_angle - angle_0))
                        )
                else:
                    features[f'{prefix}_first_swing_amp'] = 0.0

                # Oscillation damping: ratio of second to first peak amplitude
                if len(postfault_angle) > 3:
                    peaks, _ = find_peaks(np.abs(postfault_angle - angle_0), distance=3)
                    if len(peaks) >= 2:
                        a1 = np.abs(postfault_angle[peaks[0]] - angle_0)
                        a2 = np.abs(postfault_angle[peaks[1]] - angle_0)
                        features[f'{prefix}_damping_ratio'] = float(
                            (a1 - a2) / a1 if a1 > 1e-6 else 0.0
                        )
                    else:
                        features[f'{prefix}_damping_ratio'] = 1.0  # fully damped
                else:
                    features[f'{prefix}_damping_ratio'] = 1.0

            # --- Speed deviation features ---
            if speed_col:
                speed = df[speed_col].values
                speed_prefault = speed[prefault_mask]
                speed_0 = speed_prefault[-1] if len(speed_prefault) > 0 else speed[0]

                features[f'{prefix}_speed_dev_peak'] = float(np.max(np.abs(speed - speed_0)))
                features[f'{prefix}_speed_dev_final'] = float(speed[-1] - speed_0)

                # Cumulative speed deviation (integral of |omega - omega_s|)
                dt_full = np.diff(time)
                speed_dev = np.abs(speed[1:] - speed_0)
                features[f'{prefix}_speed_integral'] = float(np.sum(speed_dev * dt_full))

            # --- Direct derivative columns if available ---
            if dorhz_col:
                dorhz = df[dorhz_col].values
                features[f'{prefix}_freq_dev_max'] = float(np.max(np.abs(dorhz)))

        return features

    # ---------------------------------------------------------------
    # Voltage recovery features
    # ---------------------------------------------------------------

    def _extract_voltage_features(
        self,
        df: pd.DataFrame,
        time: np.ndarray,
        bus_cols: Dict[str, List[str]],
        prefault_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Extract voltage magnitude recovery features per bus."""
        features = {}

        # Collect all bus voltages for system-level features
        all_voltages = []
        all_prefault_v = []

        for bus_id, cols in bus_cols.items():
            v_col = _find_col(cols, 'm:u in p.u.')
            if not v_col:
                continue

            voltage = df[v_col].values
            v_prefault = voltage[prefault_mask]
            v_0 = v_prefault[-1] if len(v_prefault) > 0 else voltage[0]

            all_voltages.append(voltage)
            all_prefault_v.append(v_0)

        if not all_voltages:
            return features

        all_voltages = np.array(all_voltages)  # (n_buses, n_timesteps)
        all_prefault_v = np.array(all_prefault_v)

        # System-level voltage features
        # Minimum voltage across all buses at any time (voltage nadir)
        features['sys_voltage_nadir'] = float(np.min(all_voltages))

        # Average voltage deviation from pre-fault
        v_dev = np.abs(all_voltages - all_prefault_v[:, np.newaxis])
        features['sys_voltage_dev_mean'] = float(np.mean(v_dev))
        features['sys_voltage_dev_max'] = float(np.max(v_dev))

        # Voltage recovery: time for minimum bus voltage to recover above 0.9 p.u.
        min_v_over_time = np.min(all_voltages, axis=0)
        postfault_time = time[~prefault_mask]
        postfault_minv = min_v_over_time[~prefault_mask]

        recovery_idx = np.where(postfault_minv >= 0.9)[0]
        if len(recovery_idx) > 0 and len(postfault_time) > 0:
            features['sys_voltage_recovery_time'] = float(postfault_time[recovery_idx[0]])
        else:
            features['sys_voltage_recovery_time'] = float(time[-1])  # never recovered

        # Voltage overshoot (max voltage after fault clearing)
        max_v_postfault = np.max(all_voltages[:, ~prefault_mask])
        features['sys_voltage_overshoot'] = float(max_v_postfault - np.max(all_prefault_v))

        # Final voltage deviation
        final_v = all_voltages[:, -1]
        features['sys_voltage_final_dev'] = float(np.mean(np.abs(final_v - all_prefault_v)))

        return features

    # ---------------------------------------------------------------
    # Frequency features
    # ---------------------------------------------------------------

    def _extract_frequency_features(
        self,
        df: pd.DataFrame,
        time: np.ndarray,
        bus_cols: Dict[str, List[str]],
        prefault_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Extract frequency response features from bus measurements."""
        features = {}

        all_freq = []
        for bus_id, cols in bus_cols.items():
            f_col = _find_col(cols, 'm:fehz')
            if not f_col:
                continue
            freq = df[f_col].values
            all_freq.append(freq)

        if not all_freq:
            return features

        all_freq = np.array(all_freq)  # (n_buses, n_timesteps)

        # System frequency (mean across buses)
        sys_freq = np.mean(all_freq, axis=0)
        nominal_freq = sys_freq[prefault_mask][-1] if np.any(prefault_mask) else 50.0

        # Frequency nadir
        features['sys_freq_nadir'] = float(np.min(sys_freq))

        # Frequency peak
        features['sys_freq_peak'] = float(np.max(sys_freq))

        # Max frequency deviation from nominal
        features['sys_freq_dev_max'] = float(np.max(np.abs(sys_freq - nominal_freq)))

        # Rate of change of frequency (RoCoF)
        dt = np.diff(time)
        dt[dt == 0] = 1e-6
        dfreq = np.diff(sys_freq) / dt
        features['sys_rocof_max'] = float(np.max(np.abs(dfreq)))
        features['sys_rocof_mean'] = float(np.mean(np.abs(dfreq)))

        # Frequency spread across buses (coherency indicator)
        freq_spread = np.max(all_freq, axis=0) - np.min(all_freq, axis=0)
        features['sys_freq_spread_max'] = float(np.max(freq_spread))
        features['sys_freq_spread_mean'] = float(np.mean(freq_spread))

        # Final frequency deviation
        features['sys_freq_final_dev'] = float(np.abs(sys_freq[-1] - nominal_freq))

        return features

    # ---------------------------------------------------------------
    # Power flow features
    # ---------------------------------------------------------------

    def _extract_power_features(
        self,
        df: pd.DataFrame,
        time: np.ndarray,
        gen_cols: Dict[str, List[str]],
        prefault_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Extract active/reactive power features per generator."""
        features = {}

        total_p = np.zeros(len(df))
        total_q = np.zeros(len(df))

        for gen_id, cols in gen_cols.items():
            p_col = _find_col(cols, 's:P1')
            q_col = _find_col(cols, 's:Q1')
            prefix = gen_id.replace(' ', '')

            if p_col:
                p = df[p_col].values
                p_prefault = p[prefault_mask]
                p_0 = p_prefault[-1] if len(p_prefault) > 0 else p[0]

                # Power swing amplitude
                features[f'{prefix}_power_swing_amp'] = float(np.max(p) - np.min(p))
                # Power deviation from pre-fault
                features[f'{prefix}_power_dev_max'] = float(np.max(np.abs(p - p_0)))

                total_p += p

            if q_col:
                q = df[q_col].values
                features[f'{prefix}_reactive_peak'] = float(np.max(np.abs(q)))
                total_q += q

        # System-level power features
        features['sys_total_power_swing'] = float(np.max(total_p) - np.min(total_p))
        features['sys_total_reactive_peak'] = float(np.max(np.abs(total_q)))

        return features

    # ---------------------------------------------------------------
    # Inter-generator coherency features
    # ---------------------------------------------------------------

    def _extract_coherency_features(
        self,
        df: pd.DataFrame,
        time: np.ndarray,
        gen_cols: Dict[str, List[str]],
        prefault_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Extract inter-generator coherency and angle spread features."""
        features = {}

        # Collect all rotor angles
        angles = {}
        for gen_id, cols in gen_cols.items():
            firot_col = _find_col(cols, 'c:firot')
            if firot_col:
                angles[gen_id] = df[firot_col].values

        if len(angles) < 2:
            return features

        gen_ids = list(angles.keys())
        angle_matrix = np.array([angles[g] for g in gen_ids])  # (n_gen, n_timesteps)

        # Maximum angle spread at any time instant
        angle_spread = np.max(angle_matrix, axis=0) - np.min(angle_matrix, axis=0)
        features['sys_angle_spread_max'] = float(np.max(angle_spread))
        features['sys_angle_spread_mean'] = float(np.mean(angle_spread))
        features['sys_angle_spread_final'] = float(angle_spread[-1])

        # Rate of angle spread change (divergence rate)
        dt = np.diff(time)
        dt[dt == 0] = 1e-6
        d_spread = np.diff(angle_spread) / dt
        features['sys_angle_divergence_rate_max'] = float(np.max(np.abs(d_spread)))
        features['sys_angle_divergence_rate_mean'] = float(np.mean(np.abs(d_spread)))

        # Maximum pairwise relative angle
        max_rel = 0.0
        for i in range(len(gen_ids)):
            for j in range(i + 1, len(gen_ids)):
                rel = np.max(np.abs(angle_matrix[i] - angle_matrix[j]))
                max_rel = max(max_rel, rel)
        features['sys_max_relative_angle'] = float(max_rel)

        # Center of inertia angle (using equal weights since H is unknown)
        coi_angle = np.mean(angle_matrix, axis=0)
        features['sys_coi_angle_max_dev'] = float(np.max(np.abs(coi_angle - coi_angle[0])))

        # Angle standard deviation across generators (coherency measure)
        angle_std = np.std(angle_matrix, axis=0)
        features['sys_angle_std_max'] = float(np.max(angle_std))
        features['sys_angle_std_final'] = float(angle_std[-1])

        return features


# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------

def _find_col(cols: List[str], pattern: str) -> Optional[str]:
    """Find a column matching a pattern from a list of columns.

    Args:
        cols: List of column names
        pattern: Pattern to search for (case-insensitive substring match)

    Returns:
        First matching column name, or None
    """
    pattern_lower = pattern.lower()
    for c in cols:
        if pattern_lower in c.lower():
            return c
    return None
