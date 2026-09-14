#!/usr/bin/env python3
"""Multi-task f=2/f=64 boundary completion built on the teacher U-Net.

Input (10 channels): f2 three-step frames, f64 three-step frames,
sin/cos of observed f2 phase, physical quality Q, observed boundary seed.
Output: complete-boundary logits, K2 logits, repaired f2 sin/cos, confidence.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# The teacher implementation is kept intact under baseline/code.  Import its
# ConvBlock instead of duplicating it so the architectural provenance is clear.
BASELINE_CODE = Path(__file__).resolve().parents[2] / "baseline" / "code"
if str(BASELINE_CODE) not in sys.path:
    sys.path.insert(0, str(BASELINE_CODE))
from traditional_unet_phase import ConvBlock

TWO_PI = 2.0 * math.pi
LOW_FREQUENCY = 2
HIGH_FREQUENCY = 64
FREQUENCY_RATIO = HIGH_FREQUENCY // LOW_FREQUENCY
INPUT_CHANNELS = 10


def demodulate_three_step(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifts = TWO_PI * np.arange(3, dtype=np.float32) / 3.0
    c = np.sum(images * np.cos(shifts)[:, None, None], axis=0)
    s = -np.sum(images * np.sin(shifts)[:, None, None], axis=0)
    phi = np.mod(np.arctan2(s, c), TWO_PI).astype(np.float32)
    amplitude = (2.0 / 3.0) * np.sqrt(c * c + s * s)
    mean = np.mean(images, axis=0)
    quality = (amplitude / (np.abs(mean) + 1e-6)).astype(np.float32)
    return phi, quality


def make_input(sample: dict[str, np.ndarray]) -> np.ndarray:
    low = sample["input_f2_3step"].astype(np.float32)
    high = sample["input_f64_3step"].astype(np.float32)
    phi2, quality = demodulate_three_step(low)
    quality = np.clip(quality, 0.0, 2.0) / 2.0
    seed = sample["wrap_boundary_observed"].astype(np.float32)
    return np.concatenate(
        [low, high, np.sin(phi2)[None], np.cos(phi2)[None], quality[None], seed[None]],
        axis=0,
    ).astype(np.float32)


class PhaseBoundaryDataset(Dataset):
    def __init__(self, files: list[Path], augment: bool = False):
        if not files:
            raise ValueError("No NPZ samples were provided")
        self.files = [Path(p) for p in files]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            sample = {key: data[key] for key in data.files}
        x = make_input(sample)
        boundary = sample["wrap_boundary_complete"].astype(np.float32)[None]
        k2 = sample["K2_gt"].astype(np.int64)
        phase_pair = np.stack([sample["C2_gt"], sample["S2_gt"]]).astype(np.float32)
        valid = sample["valid_mask"].astype(np.float32)[None]

        # Horizontal flips would reverse the projector phase direction and K2
        # convention, so only vertical flipping is physically label-preserving.
        if self.augment and torch.rand(()) < 0.5:
            x = x[:, ::-1].copy()
            boundary = boundary[:, ::-1].copy()
            k2 = k2[::-1].copy()
            phase_pair = phase_pair[:, ::-1].copy()
            valid = valid[:, ::-1].copy()
        return {
            "input": torch.from_numpy(x),
            "boundary": torch.from_numpy(boundary),
            "k2": torch.from_numpy(k2),
            "phase_pair": torch.from_numpy(phase_pair),
            "valid": torch.from_numpy(valid),
            "path": str(path),
        }


def split_files(data_dir: Path, val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    files = sorted((data_dir / "samples").glob("*.npz"))
    if len(files) < 2:
        raise ValueError(f"Need at least two samples under {data_dir / 'samples'}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    val_n = max(1, int(round(len(files) * val_fraction)))
    val_n = min(val_n, len(files) - 1)
    val_ids = set(order[:val_n].tolist())
    train = [p for i, p in enumerate(files) if i not in val_ids]
    val = [p for i, p in enumerate(files) if i in val_ids]
    return train, val


class BoundaryMultiTaskUNet(nn.Module):
    """Teacher ClassicUNetMD encoder/decoder with four task-specific heads."""

    def __init__(self, in_ch: int = INPUT_CHANNELS, base: int = 32):
        super().__init__()
        b = int(base)
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.bot = ConvBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(b * 8 + b * 4, b * 4)
        self.dec2 = ConvBlock(b * 4 + b * 2, b * 2)
        self.dec1 = ConvBlock(b * 2 + b, b)
        self.boundary_head = nn.Conv2d(b, 1, 1)
        self.k2_head = nn.Conv2d(b, 2, 1)
        self.phase_head = nn.Conv2d(b, 2, 1)
        self.confidence_head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bot(self.pool(e3))
        d3 = F.interpolate(z, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(torch.cat([d1, e1], dim=1))
        return {
            "boundary_logits": self.boundary_head(features),
            "k2_logits": self.k2_head(features),
            "phase_pair": self.phase_head(features),
            "confidence_logits": self.confidence_head(features),
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    intersection = torch.sum(pred * target, dim=(1, 2, 3))
    denominator = torch.sum(pred + target, dim=(1, 2, 3))
    return torch.mean(1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))


def masked_phase_vector_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    pred = F.normalize(prediction, p=2, dim=1, eps=1e-6)
    truth = F.normalize(target, p=2, dim=1, eps=1e-6)
    circular = 1.0 - torch.sum(pred * truth, dim=1, keepdim=True).clamp(-1.0, 1.0)
    return torch.sum(circular * valid) / torch.sum(valid).clamp_min(1.0)


def topology_loss(k2_logits: torch.Tensor, boundary_target: torch.Tensor) -> torch.Tensor:
    p1 = torch.softmax(k2_logits, dim=1)[:, 1:2]
    edge = torch.abs(p1[..., 1:] - p1[..., :-1])
    target = torch.maximum(boundary_target[..., 1:], boundary_target[..., :-1])
    return F.l1_loss(edge, target)


def multitask_loss(
    output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]
) -> tuple[torch.Tensor, dict[str, float]]:
    boundary = batch["boundary"]
    positives = torch.sum(boundary)
    negatives = boundary.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0)
    l_boundary_bce = F.binary_cross_entropy_with_logits(
        output["boundary_logits"], boundary, pos_weight=pos_weight
    )
    l_boundary_dice = dice_loss(output["boundary_logits"], boundary)
    l_k2 = F.cross_entropy(output["k2_logits"], batch["k2"])
    l_phase = masked_phase_vector_loss(output["phase_pair"], batch["phase_pair"], batch["valid"])
    l_conf = F.binary_cross_entropy_with_logits(output["confidence_logits"], batch["valid"])
    l_topology = topology_loss(output["k2_logits"], boundary)
    parts = {
        "boundary": l_boundary_bce + l_boundary_dice,
        "k2": l_k2,
        "phase": l_phase,
        "confidence": l_conf,
        "topology": l_topology,
    }
    total = sum(float(weights[name]) * value for name, value in parts.items())
    return total, {name: float(value.detach().item()) for name, value in parts.items()}


@torch.no_grad()
def metrics(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    boundary_pred = torch.sigmoid(output["boundary_logits"]) >= 0.5
    boundary_gt = batch["boundary"] >= 0.5
    tp = torch.sum(boundary_pred & boundary_gt).float()
    fp = torch.sum(boundary_pred & ~boundary_gt).float()
    fn = torch.sum(~boundary_pred & boundary_gt).float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    k2_pred = torch.argmax(output["k2_logits"], dim=1)
    phase_pred = torch.atan2(output["phase_pair"][:, 1], output["phase_pair"][:, 0])
    phase_gt = torch.atan2(batch["phase_pair"][:, 1], batch["phase_pair"][:, 0])
    phase_error = torch.atan2(torch.sin(phase_pred - phase_gt), torch.cos(phase_pred - phase_gt)).abs()
    valid = batch["valid"][:, 0]
    return {
        "boundary_precision": float(precision.item()),
        "boundary_recall": float(recall.item()),
        "boundary_f1": float(f1.item()),
        "k2_accuracy": float((k2_pred == batch["k2"]).float().mean().item()),
        "phase_mae_rad": float(torch.sum(phase_error * valid).item() / torch.sum(valid).clamp_min(1.0).item()),
        "confidence_accuracy": float(((torch.sigmoid(output["confidence_logits"]) >= 0.5) == (batch["valid"] >= 0.5)).float().mean().item()),
    }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_manifest(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
