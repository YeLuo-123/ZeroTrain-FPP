from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .degradation import DamageConfig, degrade


class FringeDataset(Dataset):
    """NPZ schema: fringe[4,H,W], phase[H,W], optional depth and valid."""
    def __init__(self, root, split="train", damage_ratio=0.2, crop_size=256, seed=42):
        self.files = sorted((Path(root) / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No NPZ samples in {Path(root) / split}")
        self.ratio, self.crop, self.seed = damage_ratio, crop_size, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch
        d = np.load(self.files[i])
        clean, phase = d["fringe"].astype("float32"), d["phase"].astype("float32")
        depth = d["depth"].astype("float32") if "depth" in d else phase.copy()
        valid = d["valid"].astype("float32") if "valid" in d else np.ones_like(phase)
        h, w = phase.shape
        if self.crop and min(h, w) >= self.crop:
            rng = np.random.default_rng(self.seed + i)
            y, x = int(rng.integers(0, h-self.crop+1)), int(rng.integers(0, w-self.crop+1))
            sl = (..., slice(y, y+self.crop), slice(x, x+self.crop))
            clean, phase, depth, valid = clean[sl], phase[sl], depth[sl], valid[sl]
        damaged, mask = degrade(clean, DamageConfig(ratio=self.ratio), self.seed + i)
        return {k: torch.from_numpy(v).float() for k, v in {
            "damaged": damaged, "clean": clean, "mask": mask,
            "phase": phase, "depth": depth, "valid": valid}.items()}
