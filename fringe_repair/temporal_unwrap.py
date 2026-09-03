"""Temporal phase-unwrapping algorithms reviewed by Zuo et al. (2016).

The functions operate on wrapped phases in ``[-pi, pi)`` and assume that the
underlying unit-frequency phase lies in the same interval.  ``fh`` and ``fl``
are the high- and low-frequency fringe counts over the unambiguous range.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


TAU = 2.0 * np.pi


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """Wrap a phase array to ``[-pi, pi)``."""
    phase = np.asarray(phase)
    return (phase + np.pi) % TAU - np.pi


def simulate_three_step_phase(
    unit_phase: np.ndarray,
    frequency: int,
    rng: np.random.Generator,
    *,
    background: float = 128.0,
    modulation: float = 70.0,
    noise_variance: float = 5.0,
) -> np.ndarray:
    """Simulate the three-step phase measurement used in Zuo et al.

    The intensity model is ``I_n=A+B*cos(f*theta+2*pi*n/3)+noise``.
    """
    theta = np.asarray(unit_phase, dtype=np.float64)
    cosine_sum = np.zeros_like(theta)
    sine_sum = np.zeros_like(theta)
    noise_sigma = math.sqrt(noise_variance)
    for step in range(3):
        shift = TAU * step / 3.0
        intensity = background + modulation * np.cos(frequency * theta + shift)
        intensity += rng.normal(0.0, noise_sigma, size=theta.shape)
        cosine_sum += intensity * math.cos(shift)
        sine_sum += intensity * math.sin(shift)
    return np.arctan2(-sine_sum, cosine_sum)


def unwrap_multi_frequency(
    high_phase: np.ndarray, unit_phase: np.ndarray, high_frequency: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reduced hierarchical (two-frequency) unwrapping, paper Eq. (6)."""
    order = np.rint(
        (high_frequency * np.asarray(unit_phase) - np.asarray(high_phase)) / TAU
    ).astype(np.int32)
    return np.asarray(high_phase) + TAU * order, order


def unwrap_multi_wavelength(
    high_phase: np.ndarray, adjacent_phase: np.ndarray, high_frequency: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two-wavelength/heterodyne unwrapping, paper Eqs. (7)-(9)."""
    equivalent_phase = wrap_phase(np.asarray(high_phase) - np.asarray(adjacent_phase))
    order = np.rint(
        (high_frequency * equivalent_phase - np.asarray(high_phase)) / TAU
    ).astype(np.int32)
    return np.asarray(high_phase) + TAU * order, order


def select_coprime_frequency(high_frequency: int, preference: str = "medium") -> int:
    """Select a valid number-theoretical low frequency.

    This follows the paper's small/medium/large selection rules while excluding
    the degenerate frequencies 1 and ``fh - 1``.
    """
    valid = [
        value
        for value in range(2, high_frequency - 1)
        if math.gcd(value, high_frequency) == 1
    ]
    if not valid:
        valid = [high_frequency - 1]
    if preference == "small":
        return valid[0]
    if preference == "large":
        return valid[-1]
    if preference == "medium":
        return min(valid, key=lambda value: (abs(value - high_frequency / 2), -value))
    raise ValueError("preference must be small, medium, or large")


def unwrap_number_theoretical(
    high_phase: np.ndarray,
    low_phase: np.ndarray,
    high_frequency: int,
    low_frequency: int,
) -> tuple[np.ndarray, np.ndarray]:
    """LUT-equivalent number-theoretical unwrapping, paper Eqs. (12)-(13)."""
    if math.gcd(high_frequency, low_frequency) != 1:
        raise ValueError("high_frequency and low_frequency must be coprime")
    observed_entry = np.rint(
        (
            high_frequency * np.asarray(low_phase)
            - low_frequency * np.asarray(high_phase)
        )
        / TAU
    ).astype(np.int64)

    entries, orders = _number_theoretical_lut(high_frequency, low_frequency)
    position = np.searchsorted(entries, observed_entry)
    lower = np.clip(position - 1, 0, len(entries) - 1)
    upper = np.clip(position, 0, len(entries) - 1)
    use_upper = (
        np.abs(entries[upper] - observed_entry)
        < np.abs(entries[lower] - observed_entry)
    )
    lut_index = np.where(use_upper, upper, lower)
    order = orders[lut_index]
    return np.asarray(high_phase) + TAU * order, order


@lru_cache(maxsize=None)
def _number_theoretical_lut(
    high_frequency: int, low_frequency: int
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the paper's integer LUT for theta in ``[-pi, pi)``."""
    boundaries = {-np.pi, np.pi}
    for frequency in (high_frequency, low_frequency):
        first = math.floor((-frequency - 1) / 2) - 1
        last = math.ceil((frequency - 1) / 2) + 1
        for index in range(first, last + 1):
            value = (2 * index + 1) * np.pi / frequency
            if -np.pi < value < np.pi:
                boundaries.add(value)
    sorted_boundaries = np.array(sorted(boundaries))
    widths = np.diff(sorted_boundaries)
    midpoints = (
        sorted_boundaries[:-1][widths > 1e-12]
        + sorted_boundaries[1:][widths > 1e-12]
    ) / 2
    high = wrap_phase(high_frequency * midpoints)
    low = wrap_phase(low_frequency * midpoints)
    high_order = np.rint((high_frequency * midpoints - high) / TAU).astype(np.int32)
    low_order = np.rint((low_frequency * midpoints - low) / TAU).astype(np.int32)
    entries = high_order * low_frequency - low_order * high_frequency

    unique: dict[int, int] = {}
    for entry, order in zip(entries.tolist(), high_order.tolist()):
        if entry in unique and unique[entry] != order:
            raise RuntimeError("Number-theoretical LUT is not unique")
        unique[entry] = order
    sorted_entries = np.array(sorted(unique), dtype=np.int64)
    sorted_orders = np.array([unique[int(entry)] for entry in sorted_entries], dtype=np.int32)
    return sorted_entries, sorted_orders
