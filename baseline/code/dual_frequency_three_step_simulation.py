#!/usr/bin/env python3
"""Reproducible noiseless dual-frequency N-step FPP simulation.

The default experiment generates 3-step and 12-step fringes for f=2 and f=64,
demodulates their wrapped phases, recovers fringe orders and absolute phases,
and verifies that a +1 low-frequency fringe-order error becomes a +32 error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TWO_PI = 2.0 * np.pi


def wrap_to_2pi(phase: np.ndarray) -> np.ndarray:
    """Wrap phase into [0, 2*pi)."""
    wrapped = np.mod(phase, TWO_PI)
    # atan2 may return a tiny negative number for a theoretical zero. After
    # modulo this becomes 2*pi-eps and introduces a spurious global cycle when
    # the phase is unwrapped. Canonicalize both representations to exactly 0.
    return np.where(np.isclose(wrapped, TWO_PI, rtol=0.0, atol=1e-12), 0.0, wrapped)


def generate_n_step_fringe(
    absolute_phase: np.ndarray,
    background: float,
    modulation: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate I_k=A+B*cos(Phi+delta_k), delta_k=2*pi*k/N."""
    if steps < 3:
        raise ValueError("steps must be at least 3")
    shifts = TWO_PI * np.arange(steps, dtype=np.float64) / steps
    images = np.stack(
        [background + modulation * np.cos(absolute_phase + shift) for shift in shifts],
        axis=0,
    )
    return images, shifts


def generate_three_step_fringe(
    absolute_phase: np.ndarray,
    background: float,
    modulation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible three-step wrapper."""
    return generate_n_step_fringe(absolute_phase, background, modulation, 3)


def demodulate_n_step(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Demodulate any uniformly shifted N-step sequence, N>=3."""
    if images.ndim != 3 or images.shape[0] < 3:
        raise ValueError("Expected images with shape (N,H,W), N>=3")
    if shifts.shape != (images.shape[0],):
        raise ValueError("Number of shifts must match the image sequence")
    cosine_component = np.sum(images * np.cos(shifts)[:, None, None], axis=0)
    sine_component = np.sum(images * np.sin(shifts)[:, None, None], axis=0)
    # For uniform shifts and cos(Phi+delta):
    # C=(NB/2)cos(Phi), S=-(NB/2)sin(Phi).
    return wrap_to_2pi(np.arctan2(-sine_component, cosine_component))


def demodulate_three_step(images: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Backward-compatible three-step demodulation wrapper."""
    if images.shape[0] != 3:
        raise ValueError("Expected exactly three images")
    return demodulate_n_step(images, shifts)


def circular_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Smallest signed angular error in [-pi, pi)."""
    return np.angle(np.exp(1j * (estimate - truth)))


def recover_reference_absolute_phase(wrapped_phase: np.ndarray) -> np.ndarray:
    """One-dimensional reference unwrap along x for this continuous scene.

    This is used only to make the noiseless simulation self-contained. In the
    proposed research system, K_2 is supplied by the boundary/order network or
    by a high-accuracy ground-truth acquisition protocol.
    """
    return np.unwrap(wrapped_phase, axis=1, period=TWO_PI)


def recover_fringe_order(
    absolute_phase: np.ndarray, wrapped_phase: np.ndarray
) -> np.ndarray:
    """Recover integer K from Phi=phi+2*pi*K."""
    return np.rint((absolute_phase - wrapped_phase) / TWO_PI).astype(np.int64)


def high_order_from_low_absolute(
    low_absolute: np.ndarray,
    high_wrapped: np.ndarray,
    frequency_ratio: int,
) -> np.ndarray:
    """Temporal dual-frequency high-order recovery."""
    return np.rint(
        (frequency_ratio * low_absolute - high_wrapped) / TWO_PI
    ).astype(np.int64)


def max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


def save_visualization(
    output_path: Path,
    low_images: np.ndarray,
    high_images: np.ndarray,
    phi_low: np.ndarray,
    phi_high: np.ndarray,
    k_low: np.ndarray,
    k_high: np.ndarray,
    k_high_wrong: np.ndarray,
    high_order_error: np.ndarray,
    low_frequency: int,
    high_frequency: int,
) -> None:
    """Save an overview figure containing observations and recovered phases."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), constrained_layout=True)
    for k in range(3):
        axes[0, k].imshow(low_images[k], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, k].set_title(f"f={low_frequency}, step {k}, delta={k * 120} deg")
        axes[1, k].imshow(high_images[k], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, k].set_title(f"f={high_frequency}, step {k}, delta={k * 120} deg")

    panels = [
        (phi_low, f"Recovered wrapped phase phi_{low_frequency}", "twilight", 0, TWO_PI),
        (k_low, f"Recovered low order K_{low_frequency}", "viridis", None, None),
        (phi_high, f"Recovered wrapped phase phi_{high_frequency}", "twilight", 0, TWO_PI),
        (k_high, f"Recovered high order K_{high_frequency}", "viridis", None, None),
        (k_high_wrong, f"K_{high_frequency} from wrong K_{low_frequency}+1", "viridis", None, None),
        (high_order_error, f"Propagation error in K_{high_frequency}", "coolwarm", None, None),
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes[2:].flat, panels):
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.78)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Noiseless dual-frequency three-step phase simulation", fontsize=16)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, float | int | str]:
    if args.high_frequency % args.low_frequency != 0:
        raise ValueError("high-frequency must be an integer multiple of low-frequency")
    if not (0.0 <= args.background <= 1.0):
        raise ValueError("background must be in [0,1]")
    if args.modulation <= 0.0:
        raise ValueError("modulation must be positive")
    if args.background - args.modulation < 0.0 or args.background + args.modulation > 1.0:
        raise ValueError("background +/- modulation must remain in [0,1]")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ratio = args.high_frequency // args.low_frequency
    x = np.arange(args.width, dtype=np.float64) / args.width
    normalized_projector_x = np.broadcast_to(x[None, :], (args.height, args.width)).copy()

    phi_low_true_absolute = TWO_PI * args.low_frequency * normalized_projector_x
    phi_high_true_absolute = TWO_PI * args.high_frequency * normalized_projector_x
    phi_low_true_wrapped = wrap_to_2pi(phi_low_true_absolute)
    phi_high_true_wrapped = wrap_to_2pi(phi_high_true_absolute)

    low_images, shifts = generate_n_step_fringe(
        phi_low_true_absolute, args.background, args.modulation, 3
    )
    high_images, _ = generate_n_step_fringe(
        phi_high_true_absolute, args.background, args.modulation, 3
    )

    low_images_12, shifts_12 = generate_n_step_fringe(
        phi_low_true_absolute, args.background, args.modulation, 12
    )
    high_images_12, _ = generate_n_step_fringe(
        phi_high_true_absolute, args.background, args.modulation, 12
    )

    phi_low = demodulate_three_step(low_images, shifts)
    phi_high = demodulate_three_step(high_images, shifts)
    phi_low_12 = demodulate_n_step(low_images_12, shifts_12)
    phi_high_12 = demodulate_n_step(high_images_12, shifts_12)

    # Reference low-frequency unwrapping for the smooth synthetic plane.
    phi_low_absolute = recover_reference_absolute_phase(phi_low)
    k_low = recover_fringe_order(phi_low_absolute, phi_low)

    # Recover high-frequency order and absolute phase from the low absolute phase.
    k_high = high_order_from_low_absolute(phi_low_absolute, phi_high, ratio)
    phi_high_absolute = phi_high + TWO_PI * k_high

    k_low_true = np.floor(phi_low_true_absolute / TWO_PI + 1e-12).astype(np.int64)
    k_high_true = np.floor(phi_high_true_absolute / TWO_PI + 1e-12).astype(np.int64)

    # Inject exactly one low-frequency order error everywhere.
    k_low_wrong = k_low + 1
    phi_low_wrong = phi_low + TWO_PI * k_low_wrong
    k_high_wrong = high_order_from_low_absolute(phi_low_wrong, phi_high, ratio)
    high_order_error = k_high_wrong - k_high

    wrapped_low_error = max_abs(circular_error(phi_low, phi_low_true_wrapped))
    wrapped_high_error = max_abs(circular_error(phi_high, phi_high_true_wrapped))
    wrapped_low_12_error = max_abs(circular_error(phi_low_12, phi_low_true_wrapped))
    wrapped_high_12_error = max_abs(circular_error(phi_high_12, phi_high_true_wrapped))
    low_absolute_error = max_abs(phi_low_absolute - phi_low_true_absolute)
    high_absolute_error = max_abs(phi_high_absolute - phi_high_true_absolute)
    low_order_error_count = int(np.count_nonzero(k_low != k_low_true))
    high_order_error_count = int(np.count_nonzero(k_high != k_high_true))
    propagation_values = np.unique(high_order_error)

    tolerance = 1e-10
    assert wrapped_low_error < tolerance, wrapped_low_error
    assert wrapped_high_error < tolerance, wrapped_high_error
    assert wrapped_low_12_error < tolerance, wrapped_low_12_error
    assert wrapped_high_12_error < tolerance, wrapped_high_12_error
    assert low_absolute_error < tolerance, low_absolute_error
    assert high_absolute_error < tolerance, high_absolute_error
    assert low_order_error_count == 0
    assert high_order_error_count == 0
    assert np.array_equal(propagation_values, np.array([ratio], dtype=np.int64))

    np.savez_compressed(
        output_dir / "dual_frequency_results.npz",
        normalized_projector_x=normalized_projector_x,
        shifts=shifts,
        low_images=low_images,
        high_images=high_images,
        low_images_12=low_images_12,
        high_images_12=high_images_12,
        phi_low=phi_low,
        phi_high=phi_high,
        phi_low_12=phi_low_12,
        phi_high_12=phi_high_12,
        K_low=k_low,
        K_high=k_high,
        Phi_low=phi_low_absolute,
        Phi_high=phi_high_absolute,
        K_low_wrong=k_low_wrong,
        K_high_wrong=k_high_wrong,
        K_high_error=high_order_error,
    )
    for frequency, images in (
        (args.low_frequency, low_images),
        (args.high_frequency, high_images),
    ):
        for step in range(3):
            plt.imsave(
                output_dir / f"fringe_f{frequency}_step{step}.png",
                images[step],
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

    save_visualization(
        output_dir / "simulation_overview.png",
        low_images,
        high_images,
        phi_low,
        phi_high,
        k_low,
        k_high,
        k_high_wrong,
        high_order_error,
        args.low_frequency,
        args.high_frequency,
    )

    report: dict[str, float | int | str] = {
        "height": args.height,
        "width": args.width,
        "low_frequency": args.low_frequency,
        "high_frequency": args.high_frequency,
        "frequency_ratio": ratio,
        "max_wrapped_phase_error_low_rad": wrapped_low_error,
        "max_wrapped_phase_error_high_rad": wrapped_high_error,
        "max_wrapped_phase_error_12step_low_rad": wrapped_low_12_error,
        "max_wrapped_phase_error_12step_high_rad": wrapped_high_12_error,
        "max_absolute_phase_error_low_rad": low_absolute_error,
        "max_absolute_phase_error_high_rad": high_absolute_error,
        "wrong_K_low_pixel_count": low_order_error_count,
        "wrong_K_high_pixel_count": high_order_error_count,
        "injected_low_order_error": 1,
        "resulting_high_order_error": int(propagation_values[0]),
        "status": "PASS",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Noiseless f=2/f=64 dual-frequency 3-step/12-step simulation"
    )
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--low-frequency", type=int, default=2)
    parser.add_argument("--high-frequency", type=int, default=64)
    parser.add_argument("--background", type=float, default=0.5)
    parser.add_argument("--modulation", type=float, default=0.45)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "dual_frequency_output_f2_f64"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Results saved to: {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
