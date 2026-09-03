#!/usr/bin/env python3
"""Compare released UDPR results with Zuo-style temporal unwrapping simulations.

The official UDPR dataset does not contain the second phase-shifting sequence
required by temporal phase unwrapping.  Consequently, the temporal methods are
evaluated in a clearly labelled protocol-matched simulation: the real UDPR
absolute-phase surfaces define geometry, while missing three-step measurements
are synthesized with the parameters used by Zuo et al. (A=128, B=70,
Gaussian noise variance=5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fringe_repair.temporal_unwrap import (
    TAU,
    select_coprime_frequency,
    simulate_three_step_phase,
    unwrap_multi_frequency,
    unwrap_multi_wavelength,
    unwrap_number_theoretical,
)
from fringe_repair.udpr_io import OfficialUDPRDataset


SECTIONS = ("3.2.1", "3.2.2", "3.2.3")
NATIVE_FREQUENCIES = {"3.2.1": 54, "3.2.2": 54, "3.2.3": 48}
PAPER_OPTIMAL = {
    "TPU-MF": (27, 1),
    "TPU-MW": (17, 16),
    "TPU-NT": (23, 12),
}
LABELS = {
    "coarse": "WFT coarse phase",
    "udpr": "UDPR released output",
    "TPU-MF": "Multi-frequency TPU",
    "TPU-MW": "Multi-wavelength TPU",
    "TPU-NT": "Number-theoretical TPU",
}


@dataclass
class Accumulator:
    absolute_sum: float = 0.0
    squared_sum: float = 0.0
    count: int = 0
    bad_01: int = 0
    bad_02: int = 0
    bad_05: int = 0
    correct_order: int = 0
    order_count: int = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        *,
        order_error: np.ndarray | None = None,
    ) -> None:
        error = np.asarray(prediction, dtype=np.float64) - np.asarray(
            target, dtype=np.float64
        )
        finite = np.isfinite(error)
        error = error[finite]
        absolute = np.abs(error)
        self.absolute_sum += float(absolute.sum())
        self.squared_sum += float(np.square(error).sum())
        self.count += int(error.size)
        self.bad_01 += int(np.count_nonzero(absolute > 0.1))
        self.bad_02 += int(np.count_nonzero(absolute > 0.2))
        self.bad_05 += int(np.count_nonzero(absolute > 0.5))
        if order_error is not None:
            order_error = np.asarray(order_error)[finite]
            self.correct_order += int(np.count_nonzero(order_error == 0))
            self.order_count += int(order_error.size)

    def merge(self, other: "Accumulator") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))

    def compute(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "valid_pixels": self.count,
            "mae_rad": self.absolute_sum / self.count,
            "rmse_rad": math.sqrt(self.squared_sum / self.count),
            "bad_0.1_rate": self.bad_01 / self.count,
            "bad_0.2_rate": self.bad_02 / self.count,
            "bad_0.5_rate": self.bad_05 / self.count,
        }
        if self.order_count:
            result["unwrap_success_rate"] = self.correct_order / self.order_count
        return result


def deterministic_rng(seed: int, *parts: object) -> np.random.Generator:
    token = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def temporal_result(
    target_native: np.ndarray,
    native_frequency: int,
    method: str,
    high_frequency: int,
    low_frequency: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.pi * (native_frequency - 1)
    theta = (np.asarray(target_native, dtype=np.float64) - origin) / native_frequency
    high_phase = simulate_three_step_phase(theta, high_frequency, rng)
    high_true = high_frequency * theta

    if method == "TPU-MF":
        low_phase = simulate_three_step_phase(theta, 1, rng)
        recovered, _ = unwrap_multi_frequency(high_phase, low_phase, high_frequency)
    elif method == "TPU-MW":
        low_phase = simulate_three_step_phase(
            theta, high_frequency - 1, rng
        )
        recovered, _ = unwrap_multi_wavelength(
            high_phase, low_phase, high_frequency
        )
    elif method == "TPU-NT":
        low_phase = simulate_three_step_phase(theta, low_frequency, rng)
        recovered, _ = unwrap_number_theoretical(
            high_phase, low_phase, high_frequency, low_frequency
        )
    else:
        raise ValueError(method)

    native_prediction = recovered * native_frequency / high_frequency + origin
    order_error = np.rint((recovered - high_true) / TAU).astype(np.int32)
    return native_prediction, order_error


def evaluate(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    section_results: dict[str, dict] = {}
    global_meters: dict[str, Accumulator] = {
        "coarse": Accumulator(),
        "udpr": Accumulator(),
        **{f"{method}_optimal": Accumulator() for method in PAPER_OPTIMAL},
        **{f"{method}_native": Accumulator() for method in PAPER_OPTIMAL},
    }
    representative: dict[str, np.ndarray] = {}

    for section in SECTIONS:
        dataset = OfficialUDPRDataset(
            args.root,
            section=section,
            stage=2,
            include_output=True,
            normalize=False,
        )
        native_frequency = NATIVE_FREQUENCIES[section]
        meters = {name: Accumulator() for name in global_meters}

        for sample_index, sample in enumerate(dataset):
            for view in ("left", "right"):
                mask = sample[f"mask_{view}"].numpy() > 0.5
                target_map = sample[f"target_{view}"].numpy()
                target = target_map[mask]
                meters["coarse"].update(sample[f"coarse_{view}"].numpy()[mask], target)
                meters["udpr"].update(sample[f"output_{view}"].numpy()[mask], target)

                for method, (high_frequency, low_frequency) in PAPER_OPTIMAL.items():
                    prediction, order_error = temporal_result(
                        target,
                        native_frequency,
                        method,
                        high_frequency,
                        low_frequency,
                        deterministic_rng(
                            args.seed, section, sample["stem"], view, method, "optimal"
                        ),
                    )
                    meters[f"{method}_optimal"].update(
                        prediction, target, order_error=order_error
                    )

                    native_low = (
                        1
                        if method == "TPU-MF"
                        else native_frequency - 1
                        if method == "TPU-MW"
                        else select_coprime_frequency(native_frequency)
                    )
                    prediction, order_error = temporal_result(
                        target,
                        native_frequency,
                        method,
                        native_frequency,
                        native_low,
                        deterministic_rng(
                            args.seed, section, sample["stem"], view, method, "native"
                        ),
                    )
                    meters[f"{method}_native"].update(
                        prediction, target, order_error=order_error
                    )

                if section == "3.2.1" and sample_index == 0 and view == "left":
                    representative["fringe"] = sample["fringe_left"].numpy()
                    representative["target"] = target_map
                    representative["mask"] = mask
                    representative["coarse"] = sample["coarse_left"].numpy()
                    representative["udpr"] = sample["output_left"].numpy()
                    for method, (high_frequency, low_frequency) in PAPER_OPTIMAL.items():
                        pred, _ = temporal_result(
                            target,
                            native_frequency,
                            method,
                            high_frequency,
                            low_frequency,
                            deterministic_rng(
                                args.seed, section, sample["stem"], view, method, "optimal"
                            ),
                        )
                        image = np.full(target_map.shape, np.nan, dtype=np.float64)
                        image[mask] = pred
                        representative[method] = image

        section_results[section] = {
            "samples": len(dataset),
            "native_frequency": native_frequency,
            "methods": {name: meter.compute() for name, meter in meters.items()},
        }
        for name, meter in meters.items():
            global_meters[name].merge(meter)

    return {
        "protocol": {
            "udpr": "released outputs on real measurements",
            "temporal": (
                "protocol-matched simulation on UDPR ground-truth surfaces; "
                "three-step phase shifting, A=128, B=70, Gaussian variance=5"
            ),
            "seed": args.seed,
            "paper_optimal_frequencies": PAPER_OPTIMAL,
        },
        "sections": section_results,
        "aggregate": {name: meter.compute() for name, meter in global_meters.items()},
    }, representative


def plot_aggregate(results: dict, output: Path) -> None:
    keys = [
        "coarse",
        "udpr",
        "TPU-MF_optimal",
        "TPU-MW_optimal",
        "TPU-NT_optimal",
    ]
    labels = ["WFT coarse", "UDPR", "TPU-MF", "TPU-MW", "TPU-NT"]
    colors = ["#7f8c8d", "#2468a2", "#2ca25f", "#f39c12", "#8e44ad"]
    aggregate = results["aggregate"]
    mae = [aggregate[key]["mae_rad"] for key in keys]
    bad = [aggregate[key]["bad_0.2_rate"] * 100 for key in keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].bar(labels, mae, color=colors)
    axes[0].set_ylabel("Absolute-phase MAE (rad)")
    axes[0].set_title("Aggregate phase error")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, bad, color=colors)
    axes[1].set_ylabel("Pixels with |error| > 0.2 rad (%)")
    axes[1].set_title("Aggregate bad-pixel rate")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "UDPR real-output evaluation vs. temporal-unwrapping simulation",
        fontsize=14,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sections(results: dict, output: Path) -> None:
    methods = ["coarse", "udpr", "TPU-MF_optimal", "TPU-MW_optimal", "TPU-NT_optimal"]
    labels = ["WFT coarse", "UDPR", "TPU-MF", "TPU-MW", "TPU-NT"]
    x = np.arange(len(SECTIONS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for index, (method, label) in enumerate(zip(methods, labels)):
        values = [
            results["sections"][section]["methods"][method]["mae_rad"]
            for section in SECTIONS
        ]
        ax.bar(x + (index - 2) * width, values, width, label=label)
    ax.set_xticks(x, [f"Section {section}" for section in SECTIONS])
    ax.set_ylabel("Absolute-phase MAE (rad)")
    ax.set_title("Phase error by UDPR dataset section")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representative(data: dict[str, np.ndarray], output: Path) -> None:
    mask = data["mask"]
    target = data["target"]
    methods = ["coarse", "udpr", "TPU-MF", "TPU-MW", "TPU-NT"]
    labels = ["WFT coarse", "UDPR", "TPU-MF", "TPU-MW", "TPU-NT"]
    vmax = np.nanpercentile(target[mask], 99)
    vmin = np.nanpercentile(target[mask], 1)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6.7), constrained_layout=True)
    axes[0, 0].imshow(data["fringe"], cmap="gray")
    axes[0, 0].set_title("Captured fringe")
    axes[1, 0].imshow(np.where(mask, target, np.nan), cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Reference phase")
    for column, (method, label) in enumerate(zip(methods, labels), start=1):
        phase = np.where(mask, data[method], np.nan)
        axes[0, column].imshow(phase, cmap="turbo", vmin=vmin, vmax=vmax)
        axes[0, column].set_title(label)
        error = np.where(mask, np.abs(phase - target), np.nan)
        image = axes[1, column].imshow(error, cmap="magma", vmin=0, vmax=0.5)
        axes[1, column].set_title(f"|error|, MAE={np.nanmean(error):.3f} rad")
    for ax in axes.flat:
        ax.axis("off")
    fig.colorbar(image, ax=axes[1, 1:].tolist(), fraction=0.018, pad=0.01, label="rad")
    fig.suptitle("Representative sample 101-1, left view", fontsize=15)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(output: Path) -> None:
    frequencies = np.arange(2, 61)
    phase_sigma = math.sqrt(2 * 5.0 / (3 * 70.0**2))
    curves: dict[str, list[float]] = {"TPU-MF": [], "TPU-MW": [], "TPU-NT": []}
    for high in frequencies:
        low_nt = select_coprime_frequency(int(high))
        gammas = {
            "TPU-MF": high**2 + 1,
            "TPU-MW": high**2 + (high - 1) ** 2,
            "TPU-NT": high**2 + low_nt**2,
        }
        for method, gamma in gammas.items():
            sigma_delta = phase_sigma * math.sqrt(gamma)
            curves[method].append(
                math.erf(np.pi / (math.sqrt(2) * sigma_delta))
            )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method, values in curves.items():
        ax.plot(frequencies, 100 * np.asarray(values), label=LABELS[method], linewidth=2)
    ax.axhline(99.9993, color="black", linestyle="--", linewidth=1, label="4.5σ criterion")
    ax.set_xlabel("High-frequency fringe count $f_h$")
    ax.set_ylabel("Predicted unwrap success rate (%)")
    ax.set_ylim(70, 100.1)
    ax.set_title("Zuo et al. stochastic reliability model")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/udpr_tpu_comparison"))
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results, representative = evaluate(args)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_aggregate(results, args.output_dir / "aggregate_metrics.png")
    plot_sections(results, args.output_dir / "section_mae.png")
    plot_representative(representative, args.output_dir / "representative_sample.png")
    plot_reliability(args.output_dir / "zuo_reliability_curve.png")
    print(json.dumps(results["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
