#!/usr/bin/env python3
"""Recompute metrics from the UCNNet outputs shipped in the official dataset."""
import argparse
import json
from pathlib import Path

import numpy as np

from fringe_repair.udpr_io import OfficialUDPRDataset
from fringe_repair.udpr_metrics import MetricAccumulator


def evaluate(root: Path, section: str, stage: int) -> dict:
    dataset = OfficialUDPRDataset(
        root, section=section, stage=stage, include_output=True, normalize=False
    )
    views = {}
    joint = MetricAccumulator(stage)
    for view in ("left", "right"):
        meter = MetricAccumulator(stage)
        for sample in dataset:
            prediction = sample[f"output_{view}"].numpy()
            target = sample[f"target_{view}"].numpy()
            mask = sample[f"mask_{view}"].numpy()
            meter.update(prediction, target, mask)
            joint.update(prediction, target, mask)
        views[view] = meter.compute()
    return {
        "section": section,
        "stage": stage,
        "samples": len(dataset),
        "left": views["left"],
        "right": views["right"],
        "both_views": joint.compute(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("Dataset"))
    parser.add_argument("--section", choices=["3.2.1", "3.2.2", "3.2.3", "all"], default="all")
    parser.add_argument("--stage", choices=["1", "2", "all"], default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sections = ["3.2.1", "3.2.2", "3.2.3"] if args.section == "all" else [args.section]
    stages = [1, 2] if args.stage == "all" else [int(args.stage)]
    results = [evaluate(args.root, section, stage) for section in sections for stage in stages]
    text = json.dumps(results, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
