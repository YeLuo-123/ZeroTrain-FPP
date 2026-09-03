"""Simulation helpers for frequency-2 phase-order boundary completion.

The source MATLAB program uses twelve internally phase-shifted projector
patterns during each of three camera exposures.  The implementation below
reproduces its integer projection counts and c2/c3 demodulation correction,
while exposing a common interface for a conventional three-step acquisition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter


TAU = 2.0 * np.pi


def wrap_positive(phase: np.ndarray) -> np.ndarray:
    """Wrap a phase array to [0, 2*pi)."""
    return np.mod(np.asarray(phase), TAU)


def circular_error(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Signed shortest angular error in [-pi, pi)."""
    return (np.asarray(estimate) - np.asarray(target) + np.pi) % TAU - np.pi


@dataclass(frozen=True)
class InternalSuperposition:
    """Parameters copied from ``Exp3_Simulation.m``."""

    internal_steps: int = 12
    scale: int = 10

    @property
    def shifts(self) -> np.ndarray:
        return TAU * np.arange(self.internal_steps) / self.internal_steps

    @property
    def ideal_counts(self) -> np.ndarray:
        return self.scale * (np.cos(self.shifts) + 1.0)

    @property
    def counts(self) -> np.ndarray:
        return np.rint(self.ideal_counts).astype(np.int32)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum())

    @property
    def c2_c3(self) -> tuple[float, float]:
        rounding_error = self.ideal_counts - self.counts
        c2 = (
            self.internal_steps * self.scale / 2.0
            - np.sum(np.cos(self.shifts) * rounding_error)
        )
        c3 = np.sum(np.sin(self.shifts) * rounding_error)
        return float(c2), float(c3)

    def spectral_amplitude(self, harmonic_order: int) -> float:
        coefficient = np.sum(
            self.counts * np.exp(1j * harmonic_order * self.shifts)
        )
        return float(np.abs(coefficient))


def make_scene(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate a smooth scene with one frequency-2 order boundary per row."""
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / max(width - 1, 1)
    y = yy / max(height - 1, 1)

    phase_1 = rng.uniform(0.0, TAU)
    phase_2 = rng.uniform(0.0, TAU)
    shift = (
        0.055 * np.sin(TAU * y + phase_1)
        + 0.025 * np.sin(2.0 * TAU * y + phase_2)
    )
    bump_x = rng.uniform(0.35, 0.65)
    bump_y = rng.uniform(0.25, 0.75)
    bump = rng.choice((-1.0, 1.0)) * 0.025 * np.exp(
        -((x - bump_x) ** 2 / 0.035 + (y - bump_y) ** 2 / 0.06)
    )
    ripple = 0.012 * np.sin(TAU * x) * np.cos(TAU * y + phase_2)
    projector_coordinate = x + shift + bump + ripple

    # Affine normalization preserves monotonicity along x and keeps K2 binary.
    minimum = float(projector_coordinate.min())
    maximum = float(projector_coordinate.max())
    projector_coordinate = (projector_coordinate - minimum) / (maximum - minimum)
    projector_coordinate = np.clip(projector_coordinate, 0.0, 1.0 - 1e-7)

    background = 0.5 + 0.035 * np.sin(TAU * x) * np.sin(TAU * y)
    modulation = 0.4 * (
        0.90
        + 0.08 * np.cos(TAU * x + phase_1)
        + 0.04 * np.sin(2.0 * TAU * y)
    )
    modulation = np.clip(modulation, 0.25, 0.46)
    return {
        "projector_coordinate": projector_coordinate,
        "background": background,
        "modulation": modulation,
    }


def make_boundary_gap(
    order_gt: np.ndarray,
    rng: np.random.Generator,
    *,
    minimum_fraction: float = 0.18,
    maximum_fraction: float = 0.32,
    half_width: int = 8,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Create a band of weak modulation crossing the true order boundary."""
    height, width = order_gt.shape
    boundary = boundary_from_order(order_gt)
    length = int(rng.uniform(minimum_fraction, maximum_fraction) * height)
    start = int(rng.integers(max(2, height // 10), max(3, height - length - height // 10)))
    stop = min(height - 1, start + length)
    mask = np.zeros_like(order_gt, dtype=bool)
    for row in range(start, stop):
        columns = np.flatnonzero(boundary[row])
        center = int(columns[0]) if len(columns) else width // 2
        left = max(0, center - half_width)
        right = min(width, center + half_width + 1)
        mask[row, left:right] = True
    return mask, (start, stop)


def render_three_step(
    absolute_phase: np.ndarray,
    background: np.ndarray,
    modulation: np.ndarray,
    rng: np.random.Generator,
    *,
    harmonic_ratio: float,
    read_noise_sigma: float,
    source_superposition: bool,
    superposition: InternalSuperposition | None = None,
) -> np.ndarray:
    """Render three camera exposures using conventional or source-code timing."""
    superposition = superposition or InternalSuperposition()
    outer_shifts = TAU * np.arange(3) / 3.0
    images = []
    harmonic = harmonic_ratio * modulation
    if source_superposition:
        for outer in outer_shifts:
            exposure = np.zeros_like(absolute_phase, dtype=np.float64)
            for count, inner in zip(superposition.counts, superposition.shifts):
                phase = absolute_phase + outer + inner
                intensity = (
                    background
                    + modulation * np.cos(phase)
                    + harmonic * np.cos(5.0 * phase)
                )
                exposure += count * intensity
            exposure += rng.normal(0.0, read_noise_sigma, exposure.shape)
            images.append(exposure)
    else:
        total = superposition.total_count
        for outer in outer_shifts:
            phase = absolute_phase + outer
            exposure = total * (
                background
                + modulation * np.cos(phase)
                + harmonic * np.cos(5.0 * phase)
            )
            exposure += rng.normal(0.0, read_noise_sigma, exposure.shape)
            images.append(exposure)
    return np.stack(images, axis=0)


def demodulate_three_step(
    images: np.ndarray,
    *,
    source_superposition: bool,
    superposition: InternalSuperposition | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover wrapped phase, mean intensity and relative modulation."""
    images = np.asarray(images, dtype=np.float64)
    if images.shape[0] != 3:
        raise ValueError("images must have shape [3,H,W]")
    superposition = superposition or InternalSuperposition()
    shifts = TAU * np.arange(3) / 3.0
    cosine_sum = np.sum(images * np.cos(shifts)[:, None, None], axis=0)
    sine_sum = np.sum(images * np.sin(shifts)[:, None, None], axis=0)

    if source_superposition:
        c2, c3 = superposition.c2_c3
        numerator = -c3 * cosine_sum + c2 * sine_sum
        denominator = c2 * cosine_sum + c3 * sine_sum
    else:
        numerator = sine_sum
        denominator = cosine_sum
    phase = wrap_positive(-np.arctan2(numerator, denominator))
    mean = images.mean(axis=0)
    amplitude = (2.0 / 3.0) * np.hypot(cosine_sum, sine_sum)
    quality = amplitude / np.maximum(np.abs(mean), 1e-8)
    return phase, mean, quality


def boundary_from_order(order: np.ndarray) -> np.ndarray:
    """Return a one-pixel boundary where a binary order changes along x."""
    order = np.asarray(order)
    boundary = np.zeros_like(order, dtype=bool)
    boundary[:, 1:] = np.diff(order, axis=1) != 0
    return boundary


def detect_boundary_rows(
    wrapped_phase: np.ndarray,
    quality: np.ndarray,
    *,
    quality_threshold: float = 0.055,
) -> np.ndarray:
    """Detect one reliable positive-coordinate wrap boundary in each row."""
    phase_drop = np.diff(wrapped_phase, axis=1)
    local_quality = median_filter(quality, size=(3, 5), mode="nearest")
    pair_quality = np.minimum(local_quality[:, :-1], local_quality[:, 1:])
    candidates = (phase_drop < -np.pi) & (pair_quality > quality_threshold)
    score = np.where(candidates, -phase_drop * pair_quality, -np.inf)
    best = np.argmax(score, axis=1)
    has_candidate = np.isfinite(score[np.arange(score.shape[0]), best])
    observed = np.full(wrapped_phase.shape[0], np.nan, dtype=np.float64)
    observed[has_candidate] = best[has_candidate] + 1

    # A physical f=2 boundary is smooth in the simulated single-surface
    # protocol.  Suppress isolated harmonic/noise wraps before completion,
    # without inventing values for rows where no reliable seed was detected.
    valid = np.isfinite(observed)
    if valid.sum() >= 5:
        values = observed[valid]
        global_center = np.median(values)
        global_mad = np.median(np.abs(values - global_center)) + 1e-6
        global_threshold = max(20.0, 3.5 * 1.4826 * global_mad)
        globally_plausible = np.abs(values - global_center) <= global_threshold
        globally_filtered = np.full_like(observed, np.nan)
        globally_filtered[np.flatnonzero(valid)[globally_plausible]] = values[
            globally_plausible
        ]
        observed = globally_filtered
        valid = np.isfinite(observed)
    if valid.sum() >= 5:
        rows = np.arange(len(observed))
        interpolated = np.interp(rows, rows[valid], observed[valid])
        local_center = median_filter(interpolated, size=11, mode="nearest")
        local_residual = np.abs(observed[valid] - local_center[valid])
        median = np.median(local_residual)
        mad = np.median(np.abs(local_residual - median)) + 1e-6
        threshold = max(5.0, median + 5.0 * 1.4826 * mad)
        keep = local_residual <= threshold
        filtered = np.full_like(observed, np.nan)
        filtered[np.flatnonzero(valid)[keep]] = observed[valid][keep]
        observed = filtered
    return observed


def complete_boundary_rows(
    observed: np.ndarray,
    width: int,
    *,
    smoothing: float = 1.5,
) -> np.ndarray:
    """Robustly complete a smooth single-valued boundary x(y)."""
    observed = np.asarray(observed, dtype=np.float64)
    rows = np.arange(len(observed), dtype=np.float64)
    valid = np.isfinite(observed)
    if valid.sum() < 4:
        fill = np.nanmedian(observed)
        if not np.isfinite(fill):
            fill = (width - 1) / 2.0
        return np.full_like(observed, np.clip(fill, 1, width - 1))

    y = rows[valid]
    x = observed[valid]
    # Shape-preserving cubic interpolation avoids the severe overshoot that a
    # global smoothing spline can produce across a long missing interval.
    interpolator = PchipInterpolator(y, x, extrapolate=True)
    completed = interpolator(rows)
    # Reliable traditional detections are physical seeds, not regression
    # targets that the completion stage is allowed to move.
    completed[valid] = observed[valid]
    return np.clip(completed, 1, width - 1)


def order_from_boundary(boundary_columns: np.ndarray, width: int) -> np.ndarray:
    """Assign K2=0/1 on the left/right of each boundary column."""
    columns = np.arange(width)[None, :]
    boundary_columns = np.asarray(boundary_columns)[:, None]
    return (columns >= boundary_columns).astype(np.int32)


def unresolved_order_from_observations(observed: np.ndarray, width: int) -> np.ndarray:
    """Use detected rows only; unresolved rows default to K2=0."""
    completed = np.where(np.isfinite(observed), observed, width + 1)
    return order_from_boundary(completed, width)


def unwrap_high_from_frequency_two(
    low_absolute: np.ndarray,
    high_wrapped: np.ndarray,
    high_frequency: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap a high frequency using an absolute frequency-2 phase."""
    ratio = high_frequency / 2.0
    order = np.rint((ratio * low_absolute - high_wrapped) / TAU).astype(np.int32)
    return high_wrapped + TAU * order, order
