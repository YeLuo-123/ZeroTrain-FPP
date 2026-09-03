#!/usr/bin/env python3
"""Generate curriculum synthetic data for f=2/f=64 phase-boundary learning.

Levels:
  clean       no invalid region;
  ellipse     simple binary ellipse (unit-test baseline);
  irregular   smooth random visibility with non-elliptic boundaries;
  geometry    shadows cast by a synthetic height field along projector rays;
  dark_valid  dark reflectance with high relative modulation (hard negative);
  bright_bad  bright ambient/saturation with weak useful modulation.

The image model separates surface reflectance from projector visibility, so a
dark pixel is not automatically labelled as shadow. Phase supervision is
masked by valid_mask; the complete K2-derived boundary remains available as a
topological label through valid short gaps.
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
    demodulate_n_step,
    generate_n_step_fringe,
    wrap_to_2pi,
)
from generate_phase_boundary_dataset import masks_from_low_order, projector_coordinate


LEVELS = ("clean", "ellipse", "irregular", "geometry", "dark_valid", "bright_bad")


def box_blur(array: np.ndarray, radius: int, repetitions: int = 2) -> np.ndarray:
    """Dependency-free smooth blur using repeated integral-image box filters."""
    result = array.astype(np.float64, copy=True)
    for _ in range(repetitions):
        padded = np.pad(result, ((radius, radius), (radius, radius)), mode="reflect")
        integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
        size = 2 * radius + 1
        result = (
            integral[size:, size:] - integral[:-size, size:]
            - integral[size:, :-size] + integral[:-size, :-size]
        ) / (size * size)
    return result


def random_field(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    coarse_h, coarse_w = max(4, height // 24), max(4, width // 24)
    coarse = rng.normal(size=(coarse_h, coarse_w))
    up = np.repeat(np.repeat(coarse, int(np.ceil(height / coarse_h)), 0),
                   int(np.ceil(width / coarse_w)), 1)[:height, :width]
    field = box_blur(up, max(2, min(height, width) // 50), repetitions=3)
    field -= field.min()
    return field / (field.max() + 1e-12)


def ellipse_visibility(shape: tuple[int, int], boundary: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    coords = np.argwhere(boundary)
    cy = int(rng.integers(height // 4, 3 * height // 4))
    cx = int(coords[np.argmin(np.abs(coords[:, 0] - cy)), 1])
    rx = int(rng.integers(max(8, width // 35), max(10, width // 9)))
    ry = int(rng.integers(max(10, height // 16), max(12, height // 5)))
    yy, xx = np.mgrid[0:height, 0:width]
    distance = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    binary = (distance > 1.0).astype(np.float64)
    # A soft transition models penumbra, while the core remains near zero.
    return np.clip(box_blur(binary, max(1, width // 160), 2), 0.015, 1.0)


def irregular_visibility(shape: tuple[int, int], boundary: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    field = random_field(height, width, rng)
    yy, xx = np.mgrid[0:height, 0:width]
    coords = np.argwhere(boundary)
    cy = int(rng.integers(height // 5, 4 * height // 5))
    cx = int(coords[np.argmin(np.abs(coords[:, 0] - cy)), 1])
    envelope = np.exp(-(((xx - cx) / (0.16 * width)) ** 2
                        + ((yy - cy) / (0.25 * height)) ** 2))
    occlusion = (0.62 * field + 0.55 * envelope) > rng.uniform(0.62, 0.76)
    visibility = 1.0 - 0.985 * box_blur(occlusion.astype(float), 3, 2)
    return np.clip(visibility, 0.015, 1.0)


def height_field(shape: tuple[int, int], boundary: np.ndarray,
                 rng: np.random.Generator) -> np.ndarray:
    """Make a foreground ridge close to the low-frequency wrap boundary."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    coords = np.argwhere(boundary)
    cy = int(rng.integers(height // 4, 3 * height // 4))
    cx = int(coords[np.argmin(np.abs(coords[:, 0] - cy)), 1] - rng.integers(8, 25))
    sx = rng.uniform(width * 0.025, width * 0.055)
    sy = rng.uniform(height * 0.08, height * 0.20)
    ridge = rng.uniform(0.6, 1.1) * np.exp(
        -0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2)
    )
    base = 0.04 * random_field(height, width, rng)
    return ridge + base


def geometry_visibility(z: np.ndarray, projector_slope: float,
                        penumbra_radius: int) -> np.ndarray:
    """Approximate projector-ray visibility for an orthographic epipolar model.

    The projector is on the left. Along every row, a point is shadowed if a
    previous point has a larger value of z + slope*x, i.e. blocks that ray.
    """
    _, width = z.shape
    ray_height = z + projector_slope * np.arange(width)[None, :]
    previous_max = np.maximum.accumulate(ray_height, axis=1)
    previous_max = np.concatenate(
        [np.full((z.shape[0], 1), -np.inf), previous_max[:, :-1]], axis=1
    )
    shadow = ray_height < previous_max - 0.015
    visibility = 1.0 - 0.985 * box_blur(shadow.astype(float), penumbra_radius, 2)
    return np.clip(visibility, 0.015, 1.0)


def choose_level(index: int, rng: np.random.Generator) -> str:
    # Curriculum-inspired mixture: simple samples remain, geometry dominates.
    # The first six items guarantee that even a small validation batch covers
    # every semantic case; later items follow the target mixture.
    if index < len(LEVELS):
        return LEVELS[index]
    probabilities = np.array([0.05, 0.10, 0.25, 0.35, 0.15, 0.10])
    return str(rng.choice(LEVELS, p=probabilities))


def render_sequences(absolute_phase: np.ndarray, background: np.ndarray,
                     modulation: np.ndarray, rng: np.random.Generator,
                     add_noise: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images_3, shifts_3 = generate_n_step_fringe(absolute_phase, background, modulation, 3)
    images_12, shifts_12 = generate_n_step_fringe(absolute_phase, background, modulation, 12)
    if add_noise:
        # Shot-like signal-dependent noise plus read noise, deterministic per scene.
        for images in (images_3, images_12):
            sigma = np.sqrt(np.clip(images, 0.0, 1.0) / 50000.0 + 2.5e-7)
            images += rng.normal(0.0, sigma)
    return (np.clip(images_3, 0.0, 1.0).astype(np.float32), shifts_3,
            np.clip(images_12, 0.0, 1.0).astype(np.float32), shifts_12)


def build_scene(index: int, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(args.seed + index)
    shape = (args.height, args.width)
    coordinate = projector_coordinate(*shape, rng)
    phi2_abs = TWO_PI * args.low_frequency * coordinate
    phi64_abs = TWO_PI * args.high_frequency * coordinate
    phi2 = wrap_to_2pi(phi2_abs)
    phi64 = wrap_to_2pi(phi64_abs)
    k2 = np.floor(phi2_abs / TWO_PI + 1e-12).astype(np.int16)
    k64 = np.floor(phi64_abs / TWO_PI + 1e-12).astype(np.int16)
    complete_boundary = masks_from_low_order(k2)
    level = choose_level(index, rng)

    reflectance = 0.65 + 0.32 * random_field(*shape, rng)
    ambient = np.full(shape, rng.uniform(0.018, 0.045), dtype=np.float64)
    visibility = np.ones(shape, dtype=np.float64)
    height = np.zeros(shape, dtype=np.float64)

    if level == "ellipse":
        visibility = ellipse_visibility(shape, complete_boundary, rng)
    elif level == "irregular":
        visibility = irregular_visibility(shape, complete_boundary, rng)
    elif level == "geometry":
        height = height_field(shape, complete_boundary, rng)
        # A steeper ray slope limits the cast shadow to a local gap. Very small
        # slopes can hide the complete boundary, which is a later-stage failure
        # case rather than a useful first boundary-completion sample.
        visibility = geometry_visibility(height, rng.uniform(0.015, 0.030), 3)
    elif level == "dark_valid":
        dark = random_field(*shape, rng) > 0.48
        reflectance[dark] *= rng.uniform(0.08, 0.18)
        # It is dark but projector visibility remains one and relative modulation remains useful.
    elif level == "bright_bad":
        bad = random_field(*shape, rng) > 0.55
        ambient[bad] = rng.uniform(0.72, 0.88)
        visibility[bad] = rng.uniform(0.03, 0.10)

    projector_dc = 0.46
    projector_ac = 0.42
    background = ambient + reflectance * projector_dc * visibility
    modulation = reflectance * projector_ac * visibility

    low3, shifts3, low12, shifts12 = render_sequences(
        phi2_abs, background.copy(), modulation.copy(), rng, args.noise
    )
    high3, _, high12, _ = render_sequences(
        phi64_abs, background.copy(), modulation.copy(), rng, args.noise
    )

    # Quality is driven by useful modulation and clipping, not darkness alone.
    saturated = np.any(low12 >= 0.995, axis=0) | np.any(high12 >= 0.995, axis=0)
    shadow_mask = visibility < 0.20
    low_quality = modulation < args.modulation_threshold
    valid = ~(shadow_mask | low_quality | saturated)
    observed_boundary = complete_boundary & valid
    edge_valid = np.ones(shape, dtype=bool)

    phi2_12 = demodulate_n_step(low12.astype(float), shifts12)
    phi64_12 = demodulate_n_step(high12.astype(float), shifts12)

    sample = {
        "input_f2_3step": low3,
        "input_f64_3step": high3,
        "gt_capture_f2_12step": low12,
        "gt_capture_f64_12step": high12,
        "phase_shifts_3step": shifts3.astype(np.float32),
        "phase_shifts_12step": shifts12.astype(np.float32),
        "projector_u": coordinate.astype(np.float32),
        "phi2_gt": phi2.astype(np.float32),
        "phi64_gt": phi64.astype(np.float32),
        "phi2_12step": phi2_12.astype(np.float32),
        "phi64_12step": phi64_12.astype(np.float32),
        "C2_gt": np.cos(phi2).astype(np.float32),
        "S2_gt": np.sin(phi2).astype(np.float32),
        "K2_gt": k2,
        "K64_gt": k64,
        "Phi2_gt": phi2_abs.astype(np.float32),
        "Phi64_gt": phi64_abs.astype(np.float32),
        "wrap_boundary_complete": complete_boundary.astype(np.uint8),
        "wrap_boundary_observed": observed_boundary.astype(np.uint8),
        "valid_mask": valid.astype(np.uint8),
        "shadow_mask": shadow_mask.astype(np.uint8),
        "saturation_mask": saturated.astype(np.uint8),
        "edge_valid_mask": edge_valid.astype(np.uint8),
        "reflectance_map": reflectance.astype(np.float32),
        "visibility_map": visibility.astype(np.float32),
        "modulation_map": modulation.astype(np.float32),
        "height_map": height.astype(np.float32),
    }
    metadata = {
        "level": level,
        "valid_pixels": int(valid.sum()),
        "shadow_pixels": int(shadow_mask.sum()),
        "saturated_pixels": int(saturated.sum()),
        "complete_boundary_pixels": int(complete_boundary.sum()),
        "observed_boundary_pixels": int(observed_boundary.sum()),
        "hidden_boundary_pixels": int((complete_boundary & ~valid).sum()),
        "mean_reflectance": float(reflectance.mean()),
        "mean_visibility": float(visibility.mean()),
        "mean_valid_intensity": float(low3[:, valid].mean()) if np.any(valid) else 0.0,
    }
    return sample, metadata


def save_preview(path: Path, sample: dict[str, np.ndarray], level: str) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    panels = [
        (sample["input_f2_3step"][0], "f=2 input", "gray", 0, 1),
        (sample["input_f64_3step"][0], "f=64 input", "gray", 0, 1),
        (sample["reflectance_map"], "reflectance", "gray", 0, 1),
        (sample["visibility_map"], "projector visibility", "gray", 0, 1),
        (sample["modulation_map"], "modulation", "magma", None, None),
        (sample["shadow_mask"], "shadow_mask", "gray", 0, 1),
        (sample["valid_mask"], "valid_mask", "gray", 0, 1),
        (sample["saturation_mask"], "saturation_mask", "gray", 0, 1),
        (sample["phi2_gt"], "phi_2 GT", "twilight", 0, TWO_PI),
        (sample["K2_gt"], "K_2 GT", "viridis", 0, 1),
        (sample["wrap_boundary_observed"], "observed boundary", "gray", 0, 1),
        (sample["wrap_boundary_complete"], "complete boundary", "gray", 0, 1),
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes.flat, panels):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Phase-boundary synthetic scene: {level}", fontsize=16)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="phase_boundary_dataset_v2")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--low-frequency", type=int, default=2)
    parser.add_argument("--high-frequency", type=int, default=64)
    parser.add_argument("--modulation-threshold", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--noise", action="store_true")
    args = parser.parse_args()
    if args.high_frequency // args.low_frequency != 32:
        raise ValueError("This dataset version expects f=2/f=64 (ratio 32)")

    root = Path(args.output_dir).expanduser().resolve()
    samples_dir, previews_dir = root / "samples", root / "previews"
    samples_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 2,
        "configuration": vars(args),
        "label_policy": {
            "phase": "12-step result supervised only where valid_mask=1",
            "boundary": "complete boundary derived from geometric K2",
            "shadow": "projector visibility < 0.20",
            "valid": "not shadow, not low modulation, not saturated",
        },
        "samples": [],
    }
    for index in range(args.num_samples):
        sample, metadata = build_scene(index, args)
        name = f"scene_{index:05d}"
        np.savez_compressed(samples_dir / f"{name}.npz", **sample)
        save_preview(previews_dir / f"{name}.png", sample, metadata["level"])
        manifest["samples"].append({"id": name, **metadata})

    counts = {level: 0 for level in LEVELS}
    for item in manifest["samples"]:
        counts[item["level"]] += 1
    manifest["level_counts"] = counts
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(root), "level_counts": counts,
                      "num_samples": args.num_samples}, indent=2))


if __name__ == "__main__":
    main()
