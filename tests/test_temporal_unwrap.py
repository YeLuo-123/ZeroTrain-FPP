import numpy as np

from fringe_repair.temporal_unwrap import (
    TAU,
    select_coprime_frequency,
    unwrap_multi_frequency,
    unwrap_multi_wavelength,
    unwrap_number_theoretical,
    wrap_phase,
)


def test_temporal_methods_are_exact_without_noise():
    theta = np.linspace(-np.pi + 1e-4, np.pi - 1e-4, 10001)
    for high_frequency in (17, 23, 27, 48, 54):
        high = wrap_phase(high_frequency * theta)

        recovered, _ = unwrap_multi_frequency(high, theta, high_frequency)
        np.testing.assert_allclose(recovered, high_frequency * theta, atol=1e-10)

        adjacent = wrap_phase((high_frequency - 1) * theta)
        recovered, _ = unwrap_multi_wavelength(high, adjacent, high_frequency)
        np.testing.assert_allclose(recovered, high_frequency * theta, atol=1e-10)

        low_frequency = select_coprime_frequency(high_frequency)
        low = wrap_phase(low_frequency * theta)
        recovered, _ = unwrap_number_theoretical(
            high, low, high_frequency, low_frequency
        )
        np.testing.assert_allclose(recovered, high_frequency * theta, atol=1e-10)


def test_wrap_phase_range():
    wrapped = wrap_phase(np.array([-7 * np.pi, -np.pi, 0, np.pi, 7 * np.pi]))
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    np.testing.assert_allclose(np.exp(1j * wrapped), np.exp(1j * np.array(
        [-7 * np.pi, -np.pi, 0, np.pi, 7 * np.pi]
    )), atol=1e-12)


def test_coprime_selection():
    for high_frequency in range(5, 60):
        low_frequency = select_coprime_frequency(high_frequency)
        assert 1 < low_frequency <= high_frequency - 1
        assert np.gcd(high_frequency, low_frequency) == 1
        assert abs(low_frequency - high_frequency / 2) <= high_frequency / 2
