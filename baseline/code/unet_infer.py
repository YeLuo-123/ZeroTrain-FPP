"""Direct-run inference entry point. This file never starts model training."""

from __future__ import annotations

from pathlib import Path

try:
    import traditional_unet_phase as core
except ModuleNotFoundError as exc:
    if exc.name == "traditional_unet_phase":
        raise ModuleNotFoundError(
            "Missing traditional_unet_phase.py. Keep it in the same folder as "
            "unet_infer.py."
        ) from exc
    raise


ROOT = Path(__file__).resolve().parent
WEIGHT_NAME = "best_unet_ordinary_physical_md.pth"
INPUT_NAMES = (
    "input.npy",
    "input.npz",
    "input.pt",
    "input.png",
    "input.jpg",
    "input.jpeg",
    "input.bmp",
    "input.tif",
    "input.tiff",
)


def find_default_weights() -> Path:
    candidates = [ROOT / "best_unet.pth"]
    runs_dir = ROOT / "traditional_unet_outputs" / "runs"
    if runs_dir.exists():
        candidates.extend(runs_dir.glob(f"*/{WEIGHT_NAME}"))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            "No trained weights were found. Put best_unet.pth beside "
            "unet_infer.py, or pass --weights PATH."
        )
    return max(existing, key=lambda path: path.stat().st_mtime)


def find_default_input() -> Path | None:
    for name in INPUT_NAMES:
        path = ROOT / name
        if path.is_file():
            return path
    return None


def main() -> None:
    args = core.build_parser().parse_args()
    args.mode = "infer"

    if not args.weights:
        args.weights = str(find_default_weights())
    if not args.input:
        input_path = find_default_input()
        if input_path is not None:
            args.input = str(input_path)
    if not args.output_dir:
        args.output_dir = str(ROOT / "inference_output")

    print("Classic U-Net inference only; training is disabled in this entry point.")
    print(f"Weights: {Path(args.weights).expanduser().resolve()}")
    if args.input:
        print(f"Input: {Path(args.input).expanduser().resolve()}")
    else:
        print("Input: built-in synthetic fringe (place input.npy/input.png beside this script for your own input)")
    core.infer(args)


if __name__ == "__main__":
    main()
