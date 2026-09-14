
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import traditional_unet_phase as core
except ModuleNotFoundError as exc:
    if exc.name == "traditional_unet_phase":
        raise ModuleNotFoundError(
            "Missing traditional_unet_phase.py. Keep it in the same folder as "
            "unet_train.py."
        ) from exc
    raise


def build_parser() -> argparse.ArgumentParser:
    p = core.build_parser()
    p.description = "Train Classic U-Net from pre-built ordinary-fringe train.pt/val.pt"
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Verify that train.pt and val.pt are present without starting training",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "infer":
        raise ValueError("Use unet_infer.py for inference with trained weights.")
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else core.ROOT / "traditional_unet_data"
    data_dir = data_dir.resolve()
    train_path = data_dir / "train.pt"
    val_path = data_dir / "val.pt"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            "Cached dataset is missing. Run unet_dataset.py first: "
            f"expected {train_path} and {val_path}"
        )
    args.mode = "train"
    args.data_dir = str(data_dir)
    print("Cached dataset found; online data generation is disabled for this entry point.")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")
    if args.check_only:
        print("Dataset check passed. Training was not started.")
        return
    print("This script starts training from the cached dataset.")
    print("It does not use or overwrite the packaged best_unet.pth.")
    out_dir = core.train(args)
    best_path = out_dir / "best_unet_ordinary_physical_md.pth"
    print(f"Training finished. Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
