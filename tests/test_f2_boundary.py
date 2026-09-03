import numpy as np

from fringe_repair.f2_boundary import (
    InternalSuperposition,
    complete_boundary_rows,
    demodulate_three_step,
    render_three_step,
    unwrap_high_from_frequency_two,
)


def test_exp3_projection_counts_and_coefficients():
    parameters = InternalSuperposition()
    assert parameters.counts.tolist() == [20, 19, 15, 10, 5, 1, 0, 1, 5, 10, 15, 19]
    assert parameters.total_count == 120
    c2, c3 = parameters.c2_c3
    assert np.isclose(c2, 61.17691453623979)
    assert abs(c3) < 1e-10
    assert np.isclose(parameters.spectral_amplitude(5), 1.1769145362398508)


def test_source_demodulation_recovers_noiseless_phase():
    rng = np.random.default_rng(4)
    phase = np.linspace(0, 8 * np.pi, 256, endpoint=False)[None, :]
    background = np.full_like(phase, 0.5)
    modulation = np.full_like(phase, 0.4)
    images = render_three_step(
        phase,
        background,
        modulation,
        rng,
        harmonic_ratio=0.0,
        read_noise_sigma=0.0,
        source_superposition=True,
    )
    recovered, _, _ = demodulate_three_step(images, source_superposition=True)
    error = (recovered - phase + np.pi) % (2 * np.pi) - np.pi
    assert np.max(np.abs(error)) < 1e-10


def test_boundary_completion_fills_missing_rows():
    rows = np.arange(120)
    target = 70 + 8 * np.sin(2 * np.pi * rows / 120)
    observed = target.copy()
    observed[40:75] = np.nan
    completed = complete_boundary_rows(observed, 160)
    assert np.isfinite(completed).all()
    assert np.mean(np.abs(completed-target)) < 1.0


def test_frequency_two_unwrap_is_exact_without_noise():
    up = np.linspace(0, 1, 300, endpoint=False)[None, :]
    low_absolute = 2 * np.pi * 2 * up
    high_true = 2 * np.pi * 48 * up
    high_wrapped = np.mod(high_true, 2 * np.pi)
    recovered, _ = unwrap_high_from_frequency_two(low_absolute, high_wrapped, 48)
    assert np.max(np.abs(recovered-high_true)) < 1e-10
