"""Command-centre cognitive signal derivation from EEG band powers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


EPS = 1e-9


@dataclass
class BandPowers:
    delta: np.ndarray
    theta: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray


class CommandCentreSignalProcessor:
    """Derive gameplay-facing command-centre signals from filtered EEG."""

    def __init__(
        self,
        channel_names: list[str],
        sampling_rate: int,
        window_seconds: float = 4.0,
        emit_hz: float = 15.0,
    ) -> None:
        self.channel_names = [name.upper() for name in channel_names]
        self.sampling_rate = int(sampling_rate)
        self.window_size = max(int(self.sampling_rate * window_seconds), 128)
        self.min_samples = max(int(self.sampling_rate * 1.0), 128)
        self.emit_interval = 1.0 / max(emit_hz, 1.0)

        self.buffer = np.zeros((len(channel_names), self.window_size))
        self.samples_seen = 0
        self.last_emit_monotonic = 0.0
        self.smoothed_signals: dict[str, float] | None = None
        self.baseline: dict[str, float] | None = None

        self.region_frontal, self.region_midline, self.region_left, self.region_right = (
            self._resolve_regions()
        )

    def reset(self) -> None:
        self.buffer.fill(0.0)
        self.samples_seen = 0
        self.last_emit_monotonic = 0.0
        self.smoothed_signals = None
        self.baseline = None

    def update(self, filtered_chunk: np.ndarray) -> dict | None:
        """Update processor state and return the latest signal payload."""
        if filtered_chunk.size == 0 or filtered_chunk.ndim != 2:
            return None

        n_channels, chunk_len = filtered_chunk.shape
        if n_channels != self.buffer.shape[0] or chunk_len <= 0:
            return None

        self._append_chunk(filtered_chunk)

        if self.samples_seen < self.min_samples:
            return None

        now = time.monotonic()
        if now - self.last_emit_monotonic < self.emit_interval:
            return None

        window = self._active_window()
        powers = self._compute_band_powers(window)
        raw = self._compute_raw_features(powers)
        self._update_baseline(raw)
        normalized = self._compute_normalized_signals(raw)
        smoothed = self._smooth_signals(normalized)

        self.last_emit_monotonic = now
        return {
            "timestamp_ms": int(time.time() * 1000),
            "signals": {key: round(float(value), 4) for key, value in smoothed.items()},
            "raw": {
                "focus_ratio": round(raw["focus_ratio"], 4),
                "drowsiness_ratio": round(raw["drowsiness_ratio"], 4),
                "beta_alpha_ratio": round(raw["beta_alpha_ratio"], 4),
                "theta_alpha_ratio": round(raw["theta_alpha_ratio"], 4),
                "engagement_index": round(raw["engagement_index"], 4),
                "relaxation_index": round(raw["relaxation_index"], 4),
                "theta_asymmetry": round(raw["theta_asymmetry"], 4),
            },
        }

    def _append_chunk(self, filtered_chunk: np.ndarray) -> None:
        chunk_len = filtered_chunk.shape[1]
        if chunk_len >= self.window_size:
            self.buffer = filtered_chunk[:, -self.window_size :].copy()
            self.samples_seen += chunk_len
            return

        self.buffer = np.hstack((self.buffer[:, chunk_len:], filtered_chunk))
        self.samples_seen += chunk_len

    def _active_window(self) -> np.ndarray:
        active_len = min(self.window_size, self.samples_seen)
        return self.buffer[:, -active_len:]

    def _compute_band_powers(self, window: np.ndarray) -> BandPowers:
        demeaned = window - np.mean(window, axis=1, keepdims=True)
        taper = np.hanning(demeaned.shape[1])
        windowed = demeaned * taper

        fft = np.fft.rfft(windowed, axis=1)
        freqs = np.fft.rfftfreq(windowed.shape[1], d=1.0 / self.sampling_rate)
        psd = (np.abs(fft) ** 2) / max(np.sum(taper**2), EPS)

        return BandPowers(
            delta=self._band_power(psd, freqs, 1.0, 4.0),
            theta=self._band_power(psd, freqs, 4.0, 8.0),
            alpha=self._band_power(psd, freqs, 8.0, 13.0),
            beta=self._band_power(psd, freqs, 13.0, 30.0),
        )

    @staticmethod
    def _band_power(psd: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return np.zeros(psd.shape[0], dtype=float)
        return np.trapezoid(psd[:, mask], freqs[mask], axis=1)

    def _resolve_regions(self) -> tuple[list[int], list[int], list[int], list[int]]:
        n = len(self.channel_names)

        frontal_named = self._indices_by_name(
            {"AF7", "AF8", "FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"}
        )
        midline_named = self._indices_by_name({"AFZ", "FZ", "CZ"})
        left_named = self._indices_by_name({"AF7", "FP1", "F3", "F7", "TP9"})
        right_named = self._indices_by_name({"AF8", "FP2", "F4", "F8", "TP10"})

        # Fallback mapping for number-labeled channels.
        if not frontal_named:
            frontal_named = [idx for idx, name in enumerate(self.channel_names) if name in {"1", "2", "3", "4"}]
        if not left_named:
            left_named = [idx for idx, name in enumerate(self.channel_names) if name in {"1", "2", "3", "4", "TP9", "AF7"}]
        if not right_named:
            right_named = [idx for idx, name in enumerate(self.channel_names) if name in {"5", "6", "7", "8", "TP10", "AF8"}]
        if not midline_named:
            midline_named = [idx for idx, name in enumerate(self.channel_names) if name in {"3", "4"}]

        frontal = frontal_named or list(range(max(2, n // 2)))
        left = left_named or list(range(max(1, n // 2)))
        right = right_named or list(range(max(1, n // 2), n))

        if midline_named:
            midline = midline_named
        elif n >= 2:
            midline = [max((n // 2) - 1, 0), min(n // 2, n - 1)]
        else:
            midline = [0]

        return frontal, midline, left, right

    def _indices_by_name(self, target_names: set[str]) -> list[int]:
        return [idx for idx, name in enumerate(self.channel_names) if name in target_names]

    @staticmethod
    def _region_mean(values: np.ndarray, indices: list[int]) -> float:
        if not indices:
            return float(np.mean(values))
        return float(np.mean(values[indices]))

    def _compute_raw_features(self, powers: BandPowers) -> dict[str, float]:
        frontal_delta = self._region_mean(powers.delta, self.region_frontal)
        frontal_theta = self._region_mean(powers.theta, self.region_frontal)
        frontal_alpha = self._region_mean(powers.alpha, self.region_frontal)
        frontal_beta = self._region_mean(powers.beta, self.region_frontal)

        midline_theta = self._region_mean(powers.theta, self.region_midline)
        left_theta = self._region_mean(powers.theta, self.region_left)
        right_theta = self._region_mean(powers.theta, self.region_right)

        beta_alpha_ratio = frontal_beta / (frontal_alpha + EPS)
        theta_alpha_ratio = frontal_theta / (frontal_alpha + EPS)
        engagement_index = frontal_beta / (frontal_alpha + frontal_theta + EPS)
        relaxation_index = (frontal_alpha + frontal_theta) / (frontal_beta + EPS)
        theta_asymmetry = (right_theta - left_theta) / (right_theta + left_theta + EPS)

        focus_ratio = midline_theta
        drowsiness_ratio = frontal_delta

        return {
            "midline_theta": midline_theta,
            "frontal_delta": frontal_delta,
            "frontal_theta": frontal_theta,
            "frontal_alpha": frontal_alpha,
            "frontal_beta": frontal_beta,
            "focus_ratio": focus_ratio,
            "drowsiness_ratio": drowsiness_ratio,
            "beta_alpha_ratio": beta_alpha_ratio,
            "theta_alpha_ratio": theta_alpha_ratio,
            "engagement_index": engagement_index,
            "relaxation_index": relaxation_index,
            "theta_asymmetry": theta_asymmetry,
        }

    def _update_baseline(self, raw: dict[str, float]) -> None:
        if self.baseline is None:
            self.baseline = {
                "midline_theta": max(raw["midline_theta"], EPS),
                "frontal_delta": max(raw["frontal_delta"], EPS),
            }
            raw["focus_ratio"] = 1.0
            raw["drowsiness_ratio"] = 1.0
            return

        alpha = 0.01
        self.baseline["midline_theta"] = (
            (1.0 - alpha) * self.baseline["midline_theta"] + alpha * max(raw["midline_theta"], EPS)
        )
        self.baseline["frontal_delta"] = (
            (1.0 - alpha) * self.baseline["frontal_delta"] + alpha * max(raw["frontal_delta"], EPS)
        )

        raw["focus_ratio"] = raw["midline_theta"] / (self.baseline["midline_theta"] + EPS)
        raw["drowsiness_ratio"] = raw["frontal_delta"] / (self.baseline["frontal_delta"] + EPS)

    def _compute_normalized_signals(self, raw: dict[str, float]) -> dict[str, float]:
        drowsiness = self._clamp01((raw["drowsiness_ratio"] - 1.0) / 1.2)
        alertness = 1.0 - drowsiness

        focus = self._clamp01((raw["focus_ratio"] - 0.8) / 0.9)
        stress = self._sigmoid(raw["beta_alpha_ratio"], center=1.2, steepness=2.4)
        workload = self._sigmoid(raw["theta_alpha_ratio"], center=1.1, steepness=2.1)
        engagement = self._sigmoid(raw["engagement_index"], center=0.75, steepness=3.0)
        relaxation = self._sigmoid(raw["relaxation_index"], center=1.3, steepness=1.8)

        beta_share = raw["frontal_beta"] / (
            raw["frontal_alpha"] + raw["frontal_theta"] + raw["frontal_beta"] + EPS
        )
        beta_moderation = math.exp(-((beta_share - 0.34) ** 2) / (2.0 * (0.12**2)))
        low_delta = 1.0 - drowsiness
        theta_balance = 1.0 - self._clamp01(abs(raw["theta_alpha_ratio"] - 1.1) / 1.2)
        flow = self._clamp01(
            (0.40 * engagement) + (0.20 * beta_moderation) + (0.25 * low_delta) + (0.15 * theta_balance)
        )

        theta_elev = self._clamp01((raw["theta_alpha_ratio"] - 1.0) / 1.2)
        beta_elev = self._clamp01((raw["beta_alpha_ratio"] - 1.0) / 1.2)
        right_leaning_theta = self._clamp01((raw["theta_asymmetry"] + 0.05) / 0.35)
        frustration = self._clamp01(
            (0.40 * theta_elev) + (0.40 * beta_elev) + (0.20 * right_leaning_theta)
        )

        return {
            "focus": focus,
            "alertness": alertness,
            "drowsiness": drowsiness,
            "stress": stress,
            "workload": workload,
            "engagement": engagement,
            "relaxation": relaxation,
            "flow": flow,
            "frustration": frustration,
        }

    def _smooth_signals(self, signals: dict[str, float]) -> dict[str, float]:
        if self.smoothed_signals is None:
            self.smoothed_signals = signals.copy()
            return self.smoothed_signals

        alpha = 0.25
        for key, value in signals.items():
            prev = self.smoothed_signals.get(key, value)
            self.smoothed_signals[key] = (alpha * value) + ((1.0 - alpha) * prev)
        return self.smoothed_signals

    @staticmethod
    def _sigmoid(value: float, center: float, steepness: float) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (value - center)))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))
