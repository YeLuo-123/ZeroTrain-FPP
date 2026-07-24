from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class DamageConfig:
    ratio: float = 0.2
    saturation_prob: float = 0.7
    block_prob: float = 0.8
    gaussian_sigma: float = 0.025
    salt_pepper: float = 0.005


def _irregular_mask(h, w, target, rng):
    mask = np.zeros((h, w), np.uint8)
    while mask.mean() < target:
        center = (int(rng.integers(w)), int(rng.integers(h)))
        axes = (int(rng.integers(max(2, w // 50), max(3, w // 8))),
                int(rng.integers(max(2, h // 50), max(3, h // 8))))
        cv2.ellipse(mask, center, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    k = max(3, (min(h, w) // 50) | 1)
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0) > 0.25


def degrade(fringe, cfg=DamageConfig(), seed=None):
    """Apply shared discontinuities, saturation, occlusions and sensor noise."""
    x = np.asarray(fringe, np.float32).copy()
    if x.shape[0] != 4:
        raise ValueError(f"Expected [4,H,W], got {x.shape}")
    rng = np.random.default_rng(seed)
    _, h, w = x.shape
    missing = _irregular_mask(h, w, cfg.ratio * 0.65, rng)
    mask = np.broadcast_to(missing, x.shape).copy()
    x[:, missing] = 0
    if rng.random() < cfg.block_prob:
        bh, bw = max(2, int(h * np.sqrt(cfg.ratio) / 3)), max(2, int(w * np.sqrt(cfg.ratio) / 3))
        y, z = int(rng.integers(0, max(1, h - bh))), int(rng.integers(0, max(1, w - bw)))
        x[:, y:y + bh, z:z + bw] = 0
        mask[:, y:y + bh, z:z + bw] = True
    if rng.random() < cfg.saturation_prob:
        sat = _irregular_mask(h, w, cfg.ratio * 0.25, rng)
        channels = rng.choice(4, int(rng.integers(1, 5)), replace=False)
        for k in channels:
            x[k, sat] = 1
            mask[k, sat] = True
    x += rng.normal(0, cfg.gaussian_sigma, x.shape).astype(np.float32)
    sp = rng.random(x.shape)
    x[sp < cfg.salt_pepper / 2] = 0
    x[sp > 1 - cfg.salt_pepper / 2] = 1
    mask |= (sp < cfg.salt_pepper / 2) | (sp > 1 - cfg.salt_pepper / 2)
    return np.clip(x, 0, 1), mask.astype(np.uint8)
