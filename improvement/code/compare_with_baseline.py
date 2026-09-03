#!/usr/bin/env python3
"""Compare the learned f=2/f=64 method with deterministic baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline" / "code"))

from fringe_repair.f2_boundary import (  # noqa: E402
    complete_boundary_rows, detect_boundary_rows, order_from_boundary,
    unresolved_order_from_observations,
)
from boundary_multitask import (  # noqa: E402
    FREQUENCY_RATIO, BoundaryMultiTaskUNet, demodulate_three_step, make_input,
)


def boundary_columns(mask: np.ndarray) -> np.ndarray:
    columns = np.full(mask.shape[0], np.nan)
    for row in range(mask.shape[0]):
        found = np.flatnonzero(mask[row])
        if found.size:
            columns[row] = found[0]
    return columns


def evaluate_arrays(k2: np.ndarray, phi2: np.ndarray, phi64: np.ndarray, sample: dict) -> dict[str, float]:
    absolute2 = phi2 + 2.0 * np.pi * k2
    k64 = np.rint((FREQUENCY_RATIO * absolute2 - phi64) / (2.0 * np.pi)).astype(np.int32)
    absolute64 = phi64 + 2.0 * np.pi * k64
    phase_error = absolute64 - sample["Phi64_gt"]
    order_error = k64 - sample["K64_gt"]
    return {
        "k2_accuracy": float(np.mean(k2 == sample["K2_gt"])),
        "k64_accuracy": float(np.mean(k64 == sample["K64_gt"])),
        "phi64_mae_rad": float(np.mean(np.abs(phase_error))),
        "phi64_rmse_rad": float(np.sqrt(np.mean(phase_error ** 2))),
        "catastrophic_order_rate": float(np.mean(np.abs(order_error) >= 2)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--manifest", required=True, help="Training run_manifest.json; its validation split is evaluated")
    p.add_argument("--output-dir", default="improvement/result/comparison_f2_f64")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    files = [Path(path) for path in manifest["validation_files"]]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model = BoundaryMultiTaskUNet(base=int(checkpoint["args"].get("base", 32))).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    methods = {name: [] for name in ("traditional_unresolved", "pchip_completion", "network_k2", "network_full", "oracle_k2")}
    with torch.no_grad():
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                sample = {key: data[key] for key in data.files}
            phi2, quality = demodulate_three_step(sample["input_f2_3step"].astype(np.float32))
            phi64, _ = demodulate_three_step(sample["input_f64_3step"].astype(np.float32))
            observed = detect_boundary_rows(phi2, quality)
            k2_unresolved = unresolved_order_from_observations(observed, phi2.shape[1])
            k2_pchip = order_from_boundary(complete_boundary_rows(observed, phi2.shape[1]), phi2.shape[1])
            output = model(torch.from_numpy(make_input(sample))[None].to(device))
            k2_network = torch.argmax(output["k2_logits"], dim=1)[0].cpu().numpy()
            pair = output["phase_pair"][0].cpu().numpy()
            phi2_network = np.mod(np.arctan2(pair[1], pair[0]), 2.0 * np.pi)
            methods["traditional_unresolved"].append(evaluate_arrays(k2_unresolved, phi2, phi64, sample))
            methods["pchip_completion"].append(evaluate_arrays(k2_pchip, phi2, phi64, sample))
            methods["network_k2"].append(evaluate_arrays(k2_network, phi2, phi64, sample))
            methods["network_full"].append(evaluate_arrays(k2_network, phi2_network, phi64, sample))
            methods["oracle_k2"].append(evaluate_arrays(sample["K2_gt"], phi2, phi64, sample))
    summary = {
        method: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for method, rows in methods.items()
    }
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    report = {"protocol": {"low_frequency": 2, "high_frequency": 64, "ratio": 32},
              "sample_count": len(files), "split": "fixed validation split", "metrics": summary}
    (output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *next(iter(summary.values())).keys()])
        writer.writeheader()
        for method, values in summary.items(): writer.writerow({"method": method, **values})
    lines = ["# f=2/f=64 同协议对比", "", f"验证场景数：{len(files)}；频率比：32。", "",
             "| 方法 | K2准确率 | K64准确率 | Phi64 MAE/rad | 灾难性错误率 |",
             "|---|---:|---:|---:|---:|"]
    labels = {"traditional_unresolved": "传统检测（缺失行不补全）", "pchip_completion": "PCHIP补线",
              "network_k2": "改进网络K2+观测相位", "network_full": "改进网络K2+修复相位", "oracle_k2": "Oracle K2"}
    for method, value in summary.items():
        lines.append(f"| {labels[method]} | {value['k2_accuracy']:.4%} | {value['k64_accuracy']:.4%} | "
                     f"{value['phi64_mae_rad']:.4f} | {value['catastrophic_order_rate']:.4%} |")
    (output_dir / "comparison_report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
