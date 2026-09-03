"""Build the cached ordinary-fringe dataset for the standalone Classic U-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import traditional_unet_phase as core


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
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    print("Ordinary traditional U-Net dataset builder")
    print("Input: ordinary single-frame fringe I = A + B*cos(phi)")
    print("Label: physical 12-step quadrature C_phi/S_phi")
    print(f"Output: {args.data_dir}")
    print(f"Train/val: {args.n_train}/{args.n_val}, shape={args.H}x{args.W}")
    core.make_cached_dataset(args)
    print("Dataset construction finished. Run train_traditional_unet.py next.")


if __name__ == "__main__":
    main()
