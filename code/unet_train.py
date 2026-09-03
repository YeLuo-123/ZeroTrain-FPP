"""Train the standalone Classic U-Net using an already-built cached dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import traditional_unet_phase as core


def build_parser() -> argparse.ArgumentParser:
    p = core.build_parser()
    p.description = "Train Classic U-Net from pre-built ordinary-fringe train.pt/val.pt"
    return p


def main() -> None:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else core.ROOT / "traditional_unet_data"
    data_dir = data_dir.resolve()
    train_path = data_dir / "train.pt"
    val_path = data_dir / "val.pt"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Cached dataset is missing. Run build_traditional_unet_dataset.py first: "
            f"expected {train_path} and {val_path}"
        )
    args.mode = "train"
    args.data_dir = str(data_dir)
    print("Cached dataset found; online data generation is disabled for this entry point.")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    core.train(args)


if __name__ == "__main__":
    main()
