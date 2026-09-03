#!/usr/bin/env python3
"""Train the f=2/f=64 boundary/K2/phase/confidence multi-task U-Net."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from boundary_multitask import (
    FREQUENCY_RATIO, HIGH_FREQUENCY, INPUT_CHANNELS, LOW_FREQUENCY,
    BoundaryMultiTaskUNet, PhaseBoundaryDataset, metrics, move_batch,
    multitask_loss, save_csv, save_manifest, split_files,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="phase_boundary_dataset_v2")
    p.add_argument("--output-dir", default="boundary_multitask_outputs")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--w-boundary", type=float, default=1.0)
    p.add_argument("--w-k2", type=float, default=1.0)
    p.add_argument("--w-phase", type=float, default=0.5)
    p.add_argument("--w-confidence", type=float, default=0.2)
    p.add_argument("--w-topology", type=float, default=0.2)
    return p


@torch.no_grad()
def evaluate(model, loader, device, weights) -> dict[str, float]:
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["input"])
        loss, _ = multitask_loss(output, batch, weights)
        rows.append({"loss": float(loss.item()), **metrics(output, batch)})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}


def main() -> None:
    args = parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.smoke:
        args.epochs, args.batch, args.base, args.workers = 1, 2, min(args.base, 8), 0
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    data_dir, output_dir = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files = split_files(data_dir, args.val_fraction, args.seed)
    train_loader = DataLoader(PhaseBoundaryDataset(train_files, True), args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(PhaseBoundaryDataset(val_files), args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = BoundaryMultiTaskUNet(base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weights = {name: getattr(args, f"w_{name}") for name in ("boundary", "k2", "phase", "confidence", "topology")}
    latest, best_path = output_dir / "latest.pth", output_dir / "best.pth"
    start, best, history = 1, math.inf, []
    if args.resume and latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, history = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["history"]
    print(f"protocol=f{LOW_FREQUENCY}/f{HIGH_FREQUENCY}, ratio={FREQUENCY_RATIO}, input_ch={INPUT_CHANNELS}")
    print(f"train={len(train_files)}, val={len(val_files)}, device={device}, amp={amp}")
    for epoch in range(start, args.epochs + 1):
        model.train(); running = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                output = model(batch["input"])
                loss, _ = multitask_loss(output, batch, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running += float(loss.item()) * batch["input"].size(0)
        val = evaluate(model, val_loader, device, weights)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": running / len(train_files), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row); save_csv(output_dir / "history.csv", history)
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history, "args": vars(args)}, latest)
        print(f"epoch={epoch:03d} train={row['train_loss']:.5f} val={val['loss']:.5f} "
              f"boundary_f1={val['boundary_f1']:.4f} k2_acc={val['k2_accuracy']:.4f}")
    save_manifest(output_dir / "run_manifest.json", {
        "protocol": {"low_frequency": LOW_FREQUENCY, "high_frequency": HIGH_FREQUENCY, "ratio": FREQUENCY_RATIO},
        "input_channels": ["f2_step0", "f2_step1", "f2_step2", "f64_step0", "f64_step1", "f64_step2",
                           "sin_phi2", "cos_phi2", "quality", "observed_boundary"],
        "outputs": ["complete_boundary", "K2", "f2_phase_sin_cos", "confidence"],
        "loss_weights": weights, "args": vars(args), "best_validation_loss": best,
        "train_files": [str(p) for p in train_files], "validation_files": [str(p) for p in val_files],
    })
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
