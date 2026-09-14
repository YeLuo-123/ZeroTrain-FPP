#!/usr/bin/env python3
"""Run multi-task inference and physics-based f=64 phase unwrapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from boundary_multitask import (
    FREQUENCY_RATIO, BoundaryMultiTaskUNet, demodulate_three_step, make_input,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--input", required=True, help="One V2 scene_*.npz")
    p.add_argument("--output-dir", default="boundary_multitask_inference")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(Path(args.weights), map_location=device, weights_only=False)
    saved = checkpoint.get("args", {})
    model = BoundaryMultiTaskUNet(base=int(saved.get("base", 32))).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    with np.load(args.input, allow_pickle=False) as data:
        sample = {key: data[key] for key in data.files}
    x = torch.from_numpy(make_input(sample))[None].to(device)
    with torch.no_grad():
        output = model(x)
    boundary_probability = torch.sigmoid(output["boundary_logits"])[0, 0].cpu().numpy()
    k2 = torch.argmax(output["k2_logits"], dim=1)[0].cpu().numpy().astype(np.int16)
    confidence = torch.sigmoid(output["confidence_logits"])[0, 0].cpu().numpy()
    pair = output["phase_pair"][0].cpu().numpy()
    phi2 = np.mod(np.arctan2(pair[1], pair[0]), 2.0 * np.pi)
    phi64, _ = demodulate_three_step(sample["input_f64_3step"].astype(np.float32))
    absolute2 = phi2 + 2.0 * np.pi * k2
    k64 = np.rint((FREQUENCY_RATIO * absolute2 - phi64) / (2.0 * np.pi)).astype(np.int16)
    absolute64 = phi64 + 2.0 * np.pi * k64
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "prediction.npz", boundary_probability=boundary_probability,
                        K2_pred=k2, phi2_pred=phi2, confidence=confidence,
                        K64_pred=k64, Phi64_pred=absolute64)
    report = {
        "input": str(Path(args.input).resolve()), "weights": str(Path(args.weights).resolve()),
        "device": str(device), "frequency_ratio": FREQUENCY_RATIO,
        "k2_accuracy": float(np.mean(k2 == sample["K2_gt"])),
        "k64_accuracy": float(np.mean(k64 == sample["K64_gt"])),
        "boundary_iou": float(np.sum((boundary_probability >= .5) & (sample["wrap_boundary_complete"] > 0)) /
                              max(1, np.sum((boundary_probability >= .5) | (sample["wrap_boundary_complete"] > 0)))),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
