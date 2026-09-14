
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import traditional_unet_phase as core
except ModuleNotFoundError as exc:
    if exc.name == "traditional_unet_phase":
        raise ModuleNotFoundError(
            "Missing traditional_unet_phase.py. Keep it in the same folder as "
            "unet_dataset.py."
        ) from exc
    raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build ordinary single-frame fringe data and 12-step phase labels"
    )
    p.add_argument(
        "--data-dir",
        default=str(core.ROOT / "traditional_unet_data"),
        help="Output directory containing train.pt and val.pt",
    )
    p.add_argument("--H", type=int, default=600)
    p.add_argument("--W", type=int, default=800)
    p.add_argument("--freq", type=int, default=10)
    p.add_argument("--A-min", dest="A_min", type=float, default=0.30)
    p.add_argument("--A-max", dest="A_max", type=float, default=0.70)
    p.add_argument("--B-min", dest="B_min", type=float, default=0.08)
    p.add_argument("--noise-photons", dest="noise_photons", type=int, default=100000)
    p.add_argument("--read-noise", dest="read_noise", type=float, default=0.0005)
    p.add_argument("--gamma-aug", dest="gamma_aug", action="store_true")
    p.add_argument("--n-train", dest="n_train", type=int, default=5000)
    p.add_argument("--n-val", dest="n_val", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quick-test",
        action="store_true",
        help="Build only 24 train and 8 validation samples at 64x64",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate train.pt and val.pt even when both already exist",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.quick_test:
        args.H = 64
        args.W = 64
        args.n_train = 24
        args.n_val = 8
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.pt"
    val_path = data_dir / "val.pt"

    if train_path.exists() and val_path.exists() and not args.force:
        print("Dataset already exists; nothing was regenerated.")
        print(f"Training data: {train_path}")
        print(f"Validation data: {val_path}")
        print("Run unet_train.py to start training.")
        return
    if (train_path.exists() or val_path.exists()) and not args.force:
        raise FileExistsError(
            "Only one cached split exists. Remove the incomplete cache or rerun "
            "unet_dataset.py with --force."
        )

    estimated_bytes = (args.n_train + args.n_val) * 7 * args.H * args.W * 2
    print("Ordinary traditional U-Net dataset builder")
    print("Input: ordinary single-frame fringe I = A + B*cos(phi)")
    print("Label: physical 12-step quadrature C_phi/S_phi")
    print(f"Output: {args.data_dir}")
    print(f"Train/val: {args.n_train}/{args.n_val}, shape={args.H}x{args.W}")
    print(f"Estimated cache size: {estimated_bytes / 1e9:.2f} GB")
    print("This script only builds data; it does not train the model.")
    core.make_cached_dataset(args)
    print("Dataset construction finished. Run unet_train.py next.")


if __name__ == "__main__":
    main()
