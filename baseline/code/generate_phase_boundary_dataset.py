#!/usr/bin/env python3
"""Generate the first reproducible f=2/f=64 phase-boundary dataset.

Each synthetic scene contains:
* f=2 and f=64 three-step inputs;
* f=2 and f=64 twelve-step sequences for high-quality phase supervision;
* geometric wrapped/absolute phase and fringe-order ground truth;
* a shadow crossing the f=2 wrap boundary;
* valid_mask, shadow_mask, complete boundary, and observed broken boundary.

The complete boundary is derived from geometric K_2 and deliberately remains
continuous through the shadow. The phase valid_mask is zero in the shadow, so
the dataset never treats an invented shadow phase as a measured phase target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dual_frequency_three_step_simulation import (
    TWO_PI,
    circular_error,
    demodulate_n_step,
    generate_n_step_fringe,
    wrap_to_2pi,
)


def projector_coordinate(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Create a monotonic coordinate whose K_2 boundary is gently curved."""
    yy, xx = np.mgrid[0:height, 0:width]
    phase = rng.uniform(0.0, TWO_PI)
    amplitude = rng.uniform(0.035, 0.075) * width
    boundary_x = width * 0.5 + amplitude * np.sin(TWO_PI * yy / height + phase)

    # Map x=0 -> u=0, x=boundary -> u=0.5, x=W-1 -> u just below 1.
    left = 0.5 * xx / np.maximum(boundary_x, 1.0)
    right = 0.5 + 0.5 * (xx - boundary_x) / np.maximum(width - boundary_x, 1.0)
    return np.where(xx < boundary_x, left, right).clip(0.0, np.nextafter(1.0, 0.0))


def masks_from_low_order(k_low: np.ndarray) -> np.ndarray:
    """Return a two-pixel-wide complete boundary from discrete K_2 changes."""
    boundary = np.zeros_like(k_low, dtype=bool)
    change_x = k_low[:, 1:] != k_low[:, :-1]
    change_y = k_low[1:, :] != k_low[:-1, :]
    boundary[:, 1:] |= change_x
    boundary[:, :-1] |= change_x
    boundary[1:, :] |= change_y
    boundary[:-1, :] |= change_y
    return boundary


def make_shadow(
    k_boundary: np.ndarray,
    rng: np.random.Generator,
    min_radius: int,
) -> np.ndarray:
    """Place an elliptical shadow so it intersects the complete boundary."""
    height, width = k_boundary.shape
    coordinates = np.argwhere(k_boundary)
    center_y = int(rng.integers(height // 4, 3 * height // 4))
    nearby = coordinates[np.argmin(np.abs(coordinates[:, 0] - center_y))]
    center_x = int(nearby[1] + rng.integers(-3, 4))
    radius_x = int(rng.integers(min_radius, max(min_radius + 1, width // 10)))
    radius_y = int(rng.integers(min_radius, max(min_radius + 1, height // 6)))
    yy, xx = np.mgrid[0:height, 0:width]
    ellipse = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
    return ellipse <= 1.0


def synthesize_sequence(
    absolute_phase: np.ndarray,
    steps: int,
    background: np.ndarray,
    modulation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    images, shifts = generate_n_step_fringe(absolute_phase, background, modulation, steps)
    return np.clip(images, 0.0, 1.0).astype(np.float32), shifts


def build_scene(
    index: int,
    height: int,
    width: int,
    low_frequency: int,
    high_frequency: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + index)
    coordinate = projector_coordinate(height, width, rng)
    phi_low_abs = TWO_PI * low_frequency * coordinate
    phi_high_abs = TWO_PI * high_frequency * coordinate
    phi_low_gt = wrap_to_2pi(phi_low_abs)
    phi_high_gt = wrap_to_2pi(phi_high_abs)
    k_low_gt = np.floor(phi_low_abs / TWO_PI + 1e-12).astype(np.int16)
    k_high_gt = np.floor(phi_high_abs / TWO_PI + 1e-12).astype(np.int16)
    complete_boundary = masks_from_low_order(k_low_gt)

    shadow_mask = make_shadow(complete_boundary, rng, min_radius=max(8, width // 40))
    valid_mask = ~shadow_mask
    observed_boundary = complete_boundary & valid_mask

    yy, xx = np.mgrid[0:height, 0:width]
    reflectance = 0.88 + 0.08 * np.sin(TWO_PI * xx / width + rng.uniform(0, TWO_PI))
    reflectance *= 0.96 + 0.04 * np.cos(TWO_PI * yy / height)
    background = (0.48 * reflectance).astype(np.float64)
    modulation = (0.42 * reflectance).astype(np.float64)
    # Keep realistic residual brightness but destroy usable modulation in shadow.
    background[shadow_mask] = 0.035
    modulation[shadow_mask] = 0.003

    low_3, shifts_3 = synthesize_sequence(phi_low_abs, 3, background, modulation)
    high_3, _ = synthesize_sequence(phi_high_abs, 3, background, modulation)
    low_12, shifts_12 = synthesize_sequence(phi_low_abs, 12, background, modulation)
    high_12, _ = synthesize_sequence(phi_high_abs, 12, background, modulation)

    phi_low_12 = demodulate_n_step(low_12.astype(np.float64), shifts_12)
    phi_high_12 = demodulate_n_step(high_12.astype(np.float64), shifts_12)
    low_error = np.abs(circular_error(phi_low_12, phi_low_gt))
    high_error = np.abs(circular_error(phi_high_12, phi_high_gt))

    # Numerical 12-step labels must agree with geometric truth on valid pixels.
    assert float(low_error[valid_mask].max()) < 2e-6
    assert float(high_error[valid_mask].max()) < 2e-6
    assert np.any(complete_boundary & shadow_mask), "Shadow must cross the boundary"
    assert np.count_nonzero(observed_boundary) < np.count_nonzero(complete_boundary)

    return {
        "input_f2_3step": low_3,
        "input_f64_3step": high_3,
        "gt_capture_f2_12step": low_12,
        "gt_capture_f64_12step": high_12,
        "phase_shifts_3step": shifts_3.astype(np.float32),
        "phase_shifts_12step": shifts_12.astype(np.float32),
        "projector_u": coordinate.astype(np.float32),
        "phi2_gt": phi_low_gt.astype(np.float32),
        "phi64_gt": phi_high_gt.astype(np.float32),
        "phi2_12step": phi_low_12.astype(np.float32),
        "phi64_12step": phi_high_12.astype(np.float32),
        "C2_gt": np.cos(phi_low_gt).astype(np.float32),
        "S2_gt": np.sin(phi_low_gt).astype(np.float32),
        "K2_gt": k_low_gt,
        "K64_gt": k_high_gt,
        "Phi2_gt": phi_low_abs.astype(np.float32),
        "Phi64_gt": phi_high_abs.astype(np.float32),
        "wrap_boundary_complete": complete_boundary.astype(np.uint8),
        "wrap_boundary_observed": observed_boundary.astype(np.uint8),
        "valid_mask": valid_mask.astype(np.uint8),
        "shadow_mask": shadow_mask.astype(np.uint8),
        "edge_valid_mask": np.ones_like(valid_mask, dtype=np.uint8),
        "modulation_map": modulation.astype(np.float32),
    }


def save_preview(path: Path, sample: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    panels = [
        (sample["input_f2_3step"][0], "f=2 three-step input", "gray"),
        (sample["input_f64_3step"][0], "f=64 three-step input", "gray"),
        (sample["phi2_gt"], "12-step/geometric phi_2 GT", "twilight"),
        (sample["K2_gt"], "K_2 GT", "viridis"),
        (sample["shadow_mask"], "shadow_mask", "gray"),
        (sample["valid_mask"], "valid_mask", "gray"),
        (sample["wrap_boundary_observed"], "observed broken boundary", "gray"),
        (sample["wrap_boundary_complete"], "complete boundary label", "gray"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if cmap not in {"gray"}:
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="phase_boundary_dataset_v1")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--low-frequency", type=int, default=2)
    parser.add_argument("--high-frequency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.high_frequency % args.low_frequency != 0:
        raise ValueError("High frequency must be an integer multiple of low frequency")

    output_dir = Path(args.output_dir).expanduser().resolve()
    samples_dir = output_dir / "samples"
    previews_dir = output_dir / "previews"
    samples_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "num_samples": args.num_samples,
        "height": args.height,
        "width": args.width,
        "low_frequency": args.low_frequency,
        "high_frequency": args.high_frequency,
        "frequency_ratio": args.high_frequency // args.low_frequency,
        "seed": args.seed,
        "phase_supervision": "12-step phase, masked by valid_mask",
        "boundary_supervision": "complete K2-derived boundary crossing shadow",
        "samples": [],
    }
    for index in range(args.num_samples):
        sample = build_scene(
            index, args.height, args.width,
            args.low_frequency, args.high_frequency, args.seed,
        )
        name = f"scene_{index:04d}"
        np.savez_compressed(samples_dir / f"{name}.npz", **sample)
        save_preview(previews_dir / f"{name}.png", sample)
        manifest["samples"].append({
            "id": name,
            "file": f"samples/{name}.npz",
            "preview": f"previews/{name}.png",
            "shadow_pixels": int(sample["shadow_mask"].sum()),
            "valid_pixels": int(sample["valid_mask"].sum()),
            "complete_boundary_pixels": int(sample["wrap_boundary_complete"].sum()),
            "observed_boundary_pixels": int(sample["wrap_boundary_observed"].sum()),
        })

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
