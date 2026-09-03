from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    HAS_MATPLOTLIB = False

    class _MatplotlibStub:
        @staticmethod
        def use(*args, **kwargs):
            return None

    class _PyplotStub:
        def __getattr__(self, name):
            raise RuntimeError("matplotlib is not installed; plotting is unavailable.")

    sys.modules.setdefault("matplotlib", _MatplotlibStub())
    sys.modules.setdefault("matplotlib.pyplot", _PyplotStub())
    plt = None


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "traditional_unet_outputs" / "runs"
N_STEPS = 12
def safe_torch_load(path: Path | str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def wrap_torch(phi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def wrap_np(phi: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(phi), np.cos(phi)).astype(np.float32)


def make_coords(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    u = (xx + 1.0) * 0.5
    return xx, yy, u


def random_smooth_field(
    xx: np.ndarray,
    yy: np.ndarray,
    rng: np.random.Generator,
    n_blobs: tuple[int, int] = (2, 6),
    amp: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    field = np.zeros_like(xx, dtype=np.float32)
    count = int(rng.integers(n_blobs[0], n_blobs[1] + 1))
    for _ in range(count):
        cx = float(rng.uniform(-0.8, 0.8))
        cy = float(rng.uniform(-0.8, 0.8))
        sx = float(rng.uniform(0.18, 0.65))
        sy = float(rng.uniform(0.18, 0.65))
        scale = float(rng.uniform(amp[0], amp[1]))
        field += scale * np.exp(
            -((xx - cx) ** 2 / (2.0 * sx**2) + (yy - cy) ** 2 / (2.0 * sy**2))
        )
    field += float(rng.uniform(-0.25, 0.25)) * xx
    field += float(rng.uniform(-0.25, 0.25)) * yy
    return field.astype(np.float32)


def apply_camera_noise(
    clean: np.ndarray,
    rng: np.random.Generator,
    noise_photons: int,
    read_noise_std: float,
    gamma: float,
) -> np.ndarray:
    image = np.clip(clean, 1e-5, 1.0) ** float(gamma)
    if noise_photons > 0:
        photons = image * float(noise_photons)
        image = (rng.poisson(np.clip(photons, 0, None)) / float(noise_photons)).astype(np.float32)
    if read_noise_std > 0:
        image = image + rng.normal(0.0, read_noise_std, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def make_carrier(batch: int, height: int, width: int, freq: int, device: torch.device) -> torch.Tensor:
    u = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
    u = u.expand(batch, 1, height, width)
    return 2.0 * math.pi * float(freq) * u


def get_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_ctx(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicUNetMD(nn.Module):
    """Classic encoder-decoder U-Net: one image -> ordinary physical C/S."""

    def __init__(self, in_ch: int = 1, out_ch: int = 2, base: int = 32):
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
        self.head = nn.Conv2d(b, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.bot(self.pool(e3))
        d3 = F.interpolate(z, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def phase_from_pair(pair: torch.Tensor) -> torch.Tensor:
    cos_like = pair[:, 0:1]
    sin_like = pair[:, 1:2]
    return torch.atan2(sin_like, cos_like)


def pair_phase_loss(pair_pred: torch.Tensor, pair_gt: torch.Tensor, phase_w: float, amp_w: float) -> torch.Tensor:
    c_pred, s_pred = pair_pred[:, 0:1], pair_pred[:, 1:2]
    c_gt, s_gt = pair_gt[:, 0:1], pair_gt[:, 1:2]
    l_pair = F.l1_loss(c_pred, c_gt) + F.l1_loss(s_pred, s_gt)
    eps = 1e-6
    pred_amp = torch.sqrt(c_pred * c_pred + s_pred * s_pred + eps)
    gt_amp = torch.sqrt(c_gt * c_gt + s_gt * s_gt + eps)
    c_pred_n, s_pred_n = c_pred / pred_amp, s_pred / pred_amp
    c_gt_n, s_gt_n = c_gt / gt_amp, s_gt / gt_amp
    cos_delta = torch.clamp(c_pred_n * c_gt_n + s_pred_n * s_gt_n, -1.0, 1.0)
    l_phase = torch.mean(1.0 - cos_delta)
    l_amp = F.l1_loss(pred_amp, gt_amp)
    return l_pair + float(phase_w) * l_phase + float(amp_w) * l_amp


def stable_phase_vector_loss(pair_pred: torch.Tensor, phi_gt: torch.Tensor) -> torch.Tensor:
    c_pred, s_pred = pair_pred[:, 0:1], pair_pred[:, 1:2]
    pred_amp = torch.sqrt(c_pred * c_pred + s_pred * s_pred + 1e-6)
    c_pred_n, s_pred_n = c_pred / pred_amp, s_pred / pred_amp
    c_gt_n, s_gt_n = torch.cos(phi_gt), torch.sin(phi_gt)
    cos_delta = torch.clamp(c_pred_n * c_gt_n + s_pred_n * s_gt_n, -1.0, 1.0)
    return torch.mean(1.0 - cos_delta)


@torch.no_grad()
def batch_metrics(pair_pred: torch.Tensor, batch: dict[str, torch.Tensor], freq: int, pair_metrics: bool) -> dict[str, float]:
    phi_pred = phase_from_pair(pair_pred)
    phi_gt = batch["phi_w"].to(phi_pred.device)
    c_gt = batch["C_phi"].to(phi_pred.device)
    s_gt = batch["S_phi"].to(phi_pred.device)
    pair_gt = torch.cat([c_gt, s_gt], dim=1)
    phase_err = torch.abs(wrap_torch(phi_pred - phi_gt))
    carrier = make_carrier(phi_pred.size(0), phi_pred.size(2), phi_pred.size(3), int(freq), phi_pred.device)
    phi_obj_pred = wrap_torch(phi_pred - carrier)
    phi_obj_gt = batch["phi_obj"].to(phi_pred.device)
    pred_amp = torch.sqrt(pair_pred[:, 0:1] ** 2 + pair_pred[:, 1:2] ** 2 + 1e-8)
    gt_amp = torch.sqrt(pair_gt[:, 0:1] ** 2 + pair_gt[:, 1:2] ** 2 + 1e-8)
    unit_pred = pair_pred / pred_amp
    unit_gt = pair_gt / gt_amp
    metrics = {
        "phi_w_mae_rad": float(torch.mean(phase_err).item()),
        "phi_w_rmse_rad": float(torch.sqrt(torch.mean(wrap_torch(phi_pred - phi_gt) ** 2)).item()),
        "phi_obj_mae_rad": float(torch.mean(torch.abs(wrap_torch(phi_obj_pred - phi_obj_gt))).item()),
        "unit_pair_l1": float(F.l1_loss(unit_pred, unit_gt).item()),
    }
    if pair_metrics:
        metrics["physical_pair_l1"] = float(F.l1_loss(pair_pred, pair_gt).item())
        metrics["physical_amp_l1"] = float(F.l1_loss(pred_amp, gt_amp).item())
    return metrics


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    out = {}
    keys = rows[0].keys()
    for key in keys:
        out[key] = float(np.mean([r[key] for r in rows]))
    return out


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def generate_ordinary_sample(
    args: argparse.Namespace,
    seed: int | None,
    plane: bool = False,
) -> dict[str, np.ndarray]:
    """Generate one ordinary-fringe sample and its physical 12-step labels."""
    rng = np.random.default_rng(seed)
    xx, yy, u = make_coords(args.H, args.W)

    carrier = 2.0 * math.pi * float(args.freq) * u
    if plane:
        obj = np.zeros_like(xx, dtype=np.float32)
    else:
        obj = random_smooth_field(xx, yy, rng, amp=(-1.2, 1.2))
    phi_abs = carrier + obj
    phi_w = wrap_np(phi_abs)

    if plane:
        A = np.full_like(xx, 0.50, dtype=np.float32)
        B = np.full_like(xx, 0.35, dtype=np.float32)
    else:
        A_field = 0.5 * (1.0 + np.tanh(
            random_smooth_field(xx, yy, rng, n_blobs=(1, 4), amp=(-1.0, 1.0))))
        A = args.A_min + (args.A_max - args.A_min) * A_field
        A = np.clip(A, args.A_min, args.A_max).astype(np.float32)

        B_field = 0.5 * (1.0 + np.tanh(
            random_smooth_field(xx, yy, rng, n_blobs=(2, 5), amp=(-1.0, 1.0))))
        B_upper = np.maximum(np.minimum(A, 1.0 - A) - 0.02, args.B_min + 1e-3)
        B = args.B_min + (B_upper - args.B_min) * B_field
        B = np.clip(B, args.B_min, B_upper).astype(np.float32)

    gamma = float(rng.uniform(0.985, 1.015)) if args.gamma_aug else 1.0
    A_gt = np.zeros_like(A)
    M_phi = np.zeros_like(A)
    D_phi = np.zeros_like(A)
    I_ordinary = None
    for k in range(N_STEPS):
        delta = 2.0 * math.pi * k / N_STEPS
        I_clean = A + B * np.cos(phi_w + delta)
        I_k = apply_camera_noise(I_clean, rng, args.noise_photons, args.read_noise, gamma)
        A_gt += I_k
        M_phi += I_k * (-np.sin(delta))
        D_phi += I_k * np.cos(delta)
        if k == 0:
            I_ordinary = I_k.copy()

    A_gt = (A_gt / N_STEPS).astype(np.float32)
    M_phi = (M_phi * (2.0 / N_STEPS)).astype(np.float32)
    D_phi = (D_phi * (2.0 / N_STEPS)).astype(np.float32)
    B_gt = np.sqrt(M_phi**2 + D_phi**2).astype(np.float32)
    phi_w_gt = wrap_np(np.arctan2(M_phi, D_phi))
    phi_obj_gt = wrap_np(phi_w_gt - carrier)
    C_phi_gt = D_phi.astype(np.float32)  # ordinary 12-step: B*cos(phi_w)
    S_phi_gt = M_phi.astype(np.float32)  # ordinary 12-step: B*sin(phi_w)

    return {
        "I": I_ordinary[None].astype(np.float32),
        "A": A_gt[None].astype(np.float32),
        "B": B_gt[None].astype(np.float32),
        "C_phi": C_phi_gt[None].astype(np.float32),
        "S_phi": S_phi_gt[None].astype(np.float32),
        "phi_w": phi_w_gt[None].astype(np.float32),
        "phi_obj": phi_obj_gt[None].astype(np.float32),
    }


class OrdinaryFringeDataset(Dataset):
    def __init__(self, args: argparse.Namespace, n: int, deterministic: bool, seed: int):
        self.args = args
        self.n = n
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seed = self.seed + idx if self.deterministic else None
        return {k: torch.from_numpy(v) for k, v in generate_ordinary_sample(self.args, seed).items()}


class CachedOrdinaryDataset(Dataset):
    keys = ["I", "A", "B", "C_phi", "S_phi", "phi_w", "phi_obj"]

    def __init__(self, path: str):
        self.obj = safe_torch_load(Path(path), map_location="cpu")
        self.data = self.obj["data"]
        self.meta = self.obj.get("meta", {})
        self.n = int(self.data["I"].shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: self.data[k][idx].float() for k in self.keys}


def make_cached_dataset(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    keys = ["I", "A", "B", "C_phi", "S_phi", "phi_w", "phi_obj"]

    def build(split: str, n: int, seed_offset: int) -> None:
        out_path = data_dir / f"{split}.pt"
        print("=" * 80)
        print(f"Generating {split}: n={n}, path={out_path}")
        print("=" * 80)
        data = {k: torch.empty((n, 1, args.H, args.W), dtype=torch.float16) for k in keys}
        for i in range(n):
            sample = generate_ordinary_sample(args, seed=args.seed + seed_offset + i)
            for key in keys:
                data[key][i].copy_(torch.from_numpy(sample[key]).to(torch.float16))
            if (i + 1) % max(1, min(50, n // 10 if n >= 10 else 1)) == 0 or i + 1 == n:
                print(f"  {split}: {i + 1}/{n}")
        meta = {
            "source": str(Path(__file__).resolve()),
            "H": args.H,
            "W": args.W,
            "freq": args.freq,
            "N_STEPS": N_STEPS,
            "n": n,
            "noise_photons": args.noise_photons,
            "read_noise": args.read_noise,
            "gamma_aug": bool(args.gamma_aug),
            "seed": args.seed + seed_offset,
            "baseline_input": "I is an ordinary high-frequency single-frame fringe.",
            "baseline_label": "C_phi/S_phi are physical ordinary 12-step quadrature labels: C=B*cos(phi_w), S=B*sin(phi_w).",
            "label_version": "ordinary_physical_quadrature_v1",
        }
        torch.save({"data": data, "meta": meta}, out_path)
        print(f"Saved {out_path} ({out_path.stat().st_size / (1024 ** 3):.2f} GB)")

    build("train", args.n_train, 0)
    build("val", args.n_val, 10000)


def build_datasets(args: argparse.Namespace):
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is not None:
        train_path = data_dir / "train.pt"
        val_path = data_dir / "val.pt"
        if train_path.exists() and val_path.exists():
            print(f"Loading cached ordinary-fringe datasets from {data_dir}")
            return CachedOrdinaryDataset(str(train_path)), CachedOrdinaryDataset(str(val_path))
        raise FileNotFoundError(f"Missing train.pt/val.pt in {data_dir}. Run --mode make_data first.")

    print("No --data-dir supplied; using online ordinary-fringe data generation.")
    train_set = OrdinaryFringeDataset(args, args.n_train, deterministic=False, seed=args.seed)
    val_set = OrdinaryFringeDataset(args, args.n_val, deterministic=True, seed=args.seed + 10000)
    return train_set, val_set


def evaluate_unet(model: nn.Module, loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_eval_batches and batch_idx >= args.max_eval_batches:
                break
            I = batch["I"].to(device, non_blocking=True)
            pair_pred = model(I)
            batch_on_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            metrics = batch_metrics(pair_pred, batch_on_device, args.freq, pair_metrics=True)
            rows.append({k: v * I.size(0) for k, v in metrics.items()} | {"n": float(I.size(0))})
    total_n = sum(r["n"] for r in rows)
    return {k: sum(r[k] for r in rows) / max(total_n, 1.0) for k in rows[0] if k != "n"} if rows else {}


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.save_dir)
    if not out_dir.is_absolute():
        out_dir = DEFAULT_OUT / args.save_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.amp = bool(args.amp and device.type == "cuda")
    train_set, val_set = build_datasets(args)
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, **loader_kwargs)

    model = ClassicUNetMD(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    scaler = get_scaler(args.amp)
    best = math.inf
    history: list[dict[str, object]] = []
    best_path = out_dir / "best_unet_ordinary_physical_md.pth"
    latest_path = out_dir / "latest_unet_ordinary_physical_md.pth"

    if args.resume and latest_path.exists():
        ckpt = safe_torch_load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sch.load_state_dict(ckpt["sch"])
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best", math.inf))
        history = list(ckpt.get("history", []))
        print(f"Resume U-Net: start_epoch={start_epoch}, best={best:.8f}")
    else:
        start_epoch = 1

    print("=" * 80)
    print("Classic U-Net ordinary-fringe baseline")
    print("U-Net input=ordinary single-frame I")
    print("label=ordinary 12-step C_phi/S_phi; phi_w=atan2(S_phi,C_phi)")
    print(f"H={args.H}, W={args.W}, train={len(train_set)}, val={len(val_set)}")
    print(f"base={args.base}, batch={args.batch}, epochs={args.epochs}, device={device}, amp={args.amp}")
    print(f"batches_per_epoch={len(train_loader)}, val_batches={len(val_loader)}, log_every={args.log_every}")
    print(f"save_dir={out_dir}")
    print("=" * 80, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        epoch_start = time.time()
        print(f"[U-Net] epoch {epoch:03d}/{args.epochs} started; waiting for first batch...", flush=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            I = batch["I"].to(device, non_blocking=True)
            pair_gt = torch.cat(
                [batch["C_phi"].to(device, non_blocking=True), batch["S_phi"].to(device, non_blocking=True)],
                dim=1,
            )
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(args.amp):
                pair_pred = model(I)
                loss = pair_phase_loss(pair_pred, pair_gt, args.phase_w, args.amp_w)
                phi_gt = batch["phi_w"].to(device, non_blocking=True)
                loss = loss + args.direct_phase_w * stable_phase_vector_loss(pair_pred, phi_gt)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}. "
                    "Restart with the patched stable-loss script; if it still happens, lower --lr or disable AMP."
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()) * I.size(0)
            if batch_idx == 1 or (args.log_every and batch_idx % args.log_every == 0) or batch_idx == len(train_loader):
                elapsed = time.time() - epoch_start
                avg_batch = elapsed / max(batch_idx, 1)
                eta_epoch = avg_batch * (len(train_loader) - batch_idx)
                running_loss = total / max(batch_idx * args.batch, 1)
                print(
                    f"[U-Net] epoch {epoch:03d}/{args.epochs} "
                    f"batch {batch_idx:04d}/{len(train_loader)} "
                    f"loss_now={float(loss.item()):.6f} "
                    f"loss_avg={running_loss:.6f} "
                    f"elapsed={fmt_seconds(elapsed)} "
                    f"eta_epoch={fmt_seconds(eta_epoch)}",
                    flush=True,
                )

        train_loss = total / len(train_loader.dataset)
        print(f"[U-Net] epoch {epoch:03d}/{args.epochs} training done; evaluating validation set...", flush=True)
        val_metrics = evaluate_unet(model, val_loader, args, device)
        val_key = val_metrics.get("phi_w_mae_rad", math.inf)
        sch.step()
        improved = val_key < best - args.min_delta
        if improved:
            best = val_key
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best}, best_path)

        row: dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "best_phi_w_mae_rad": best,
            "lr": opt.param_groups[0]["lr"],
            "improved": int(improved),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        write_csv(out_dir / "train_history.csv", history)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sch": sch.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best": best,
                "history": history,
                "args": vars(args),
            },
            latest_path,
        )
        print(
            f"[U-Net] {epoch:03d}/{args.epochs} train={train_loss:.6f} "
            f"val_phi_w_mae={val_metrics.get('phi_w_mae_rad', float('nan')):.6f} "
            f"val_phi_obj_mae={val_metrics.get('phi_obj_mae_rad', float('nan')):.6f} "
            f"best={best:.6f} "
            f"epoch_time={fmt_seconds(time.time() - epoch_start)}",
            flush=True,
        )

    if best_path.exists():
        model.load_state_dict(safe_torch_load(best_path, map_location=device)["model"])

    unet_metrics = evaluate_unet(model, val_loader, args, device)
    rows: list[dict[str, object]] = [{"method": "Classic_U-Net_ordinary_physical_md", **unet_metrics}]
    write_csv(out_dir / "eval_summary.csv", rows)
    visualize(model, val_set, args, device, out_dir / "visual_compare.png")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Standalone Classic U-Net ordinary-fringe phase retrieval with physical 12-step quadrature labels.",
        "unet_input": "sample['I'] ordinary high-frequency single-frame fringe",
        "label": "sample['C_phi'], sample['S_phi'] from ordinary 12-step phase shifting",
        "wrapped_phase": "atan2(S_phi, C_phi)",
        "simulation": "All scene, noise, and 12-step functions are embedded in this file.",
        "cached_data_used": bool(args.data_dir),
        "args": vars(args),
        "eval_summary": rows,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Outputs: {out_dir}")
    return out_dir


def _select_array_from_loaded(obj, source: Path) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        for key in ("I", "input", "image", "fringe"):
            if key in obj:
                value = obj[key]
                return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        for value in obj.values():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    raise ValueError(f"Could not find a numeric image array in {source}")


def load_inference_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = "I" if "I" in data.files else data.files[0]
            array = data[key]
    elif suffix in {".pt", ".pth", ".ckpt"}:
        array = _select_array_from_loaded(safe_torch_load(path, map_location="cpu"), path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        if not HAS_MATPLOTLIB:
            raise RuntimeError("Image input requires matplotlib, or convert the image to .npy first.")
        array = plt.imread(path)
    else:
        raise ValueError("Supported input formats: .npy, .npz, .pt, .pth, .ckpt, .png, .jpg, .bmp, .tif, .tiff")

    array = np.squeeze(np.asarray(array))
    if array.ndim == 3:
        if array.shape[-1] in (3, 4):
            array = array[..., :3].mean(axis=-1)
        elif array.shape[0] == 1:
            array = array[0]
        else:
            raise ValueError(f"Expected one grayscale image, got shape {array.shape} from {path}")
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {array.shape} from {path}")
    array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if float(np.max(array)) > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def load_inference_model(weights: Path, args: argparse.Namespace, device: torch.device) -> nn.Module:
    checkpoint = safe_torch_load(weights, map_location=device)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    base = int(saved_args.get("base", args.base)) if isinstance(saved_args, dict) else int(args.base)
    model = ClassicUNetMD(base=base).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Incompatible U-Net weights. missing={missing}, unexpected={unexpected}")
    model.eval()
    print(f"Loaded weights: {weights}")
    print(f"U-Net base channels: {base}")
    return model


@torch.no_grad()
def infer(args: argparse.Namespace) -> Path:
    if not args.weights:
        raise ValueError("--weights is required for --mode infer")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = load_inference_model(weights, args, device)

    ground_truth = None
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        image = load_inference_image(input_path)
        source = str(input_path)
    else:
        sample = generate_ordinary_sample(args, seed=args.seed)
        image = sample["I"][0]
        ground_truth = {k: torch.from_numpy(v).unsqueeze(0) for k, v in sample.items()}
        source = f"embedded synthetic sample (seed={args.seed})"

    height, width = image.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    image_tensor = torch.from_numpy(image)[None, None].to(device)
    if pad_h or pad_w:
        mode = "reflect" if height > 1 and width > 1 else "constant"
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode=mode)
    pair_pred = model(image_tensor)[..., :height, :width]
    phi_pred = phase_from_pair(pair_pred)[0, 0].cpu().numpy().astype(np.float32)
    pair_np = pair_pred[0].cpu().numpy().astype(np.float32)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUT / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "input_image.npy", image)
    np.save(out_dir / "predicted_pair.npy", pair_np)
    np.save(out_dir / "wrapped_phase.npy", phi_pred)

    metrics = {}
    if ground_truth is not None:
        metrics = batch_metrics(pair_pred.cpu(), ground_truth, args.freq, pair_metrics=True)
        print(f"Synthetic validation metrics: {metrics}")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "infer",
        "weights": str(weights),
        "input": source,
        "input_shape": [int(height), int(width)],
        "device": str(device),
        "outputs": ["input_image.npy", "predicted_pair.npy", "wrapped_phase.npy"],
        "synthetic_metrics": metrics,
    }
    (out_dir / "inference_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Input fringe")
        axes[1].imshow(pair_np[0], cmap="gray")
        axes[1].set_title("Predicted cos-like channel")
        axes[2].imshow(phi_pred, cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[2].set_title("Wrapped phase")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "inference_visual.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved inference outputs: {out_dir}")
    return out_dir


@torch.no_grad()
def visualize(
    unet_model: nn.Module,
    dataset,
    args: argparse.Namespace,
    device: torch.device,
    out_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib is not installed; skip visual_compare.png")
        return
    unet_model.eval()
    sample = dataset[0]
    I = sample["I"].unsqueeze(0).to(device)
    pair_unet = unet_model(I)
    phi_unet = phase_from_pair(pair_unet)[0, 0].detach().cpu().numpy()
    phi_gt = sample["phi_w"][0].numpy()
    err_unet = wrap_np(phi_unet - phi_gt)
    items = [
        (sample["I"][0].numpy(), "U-Net input: ordinary I", "gray"),
        (phi_gt, "GT phi_w", "twilight_shifted"),
        (phi_unet, "U-Net phi_w", "twilight_shifted"),
        (err_unet, f"U-Net error MAE={np.mean(np.abs(err_unet)):.4f}", "RdBu"),
    ]
    cols = 3
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), squeeze=False)
    for ax, item in zip(axes.flat, items):
        img, title, cmap = item
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visual comparison: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone Classic U-Net ordinary-fringe phase retrieval")
    p.add_argument("--mode", choices=["auto", "make_data", "train", "infer"], default="auto")
    p.add_argument(
        "--data-dir",
        default="",
        help="Optional train.pt/val.pt cache directory. Empty means online generation.",
    )
    p.add_argument("--save-dir", default="formal_ordinary_physical_base32")
    p.add_argument("--weights", default="", help="Trained U-Net checkpoint for --mode infer")
    p.add_argument("--input", default="", help="One fringe image (.npy/.npz/.pt/.png/.jpg...) for --mode infer")
    p.add_argument("--output-dir", dest="output_dir", default="", help="Inference output directory")
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
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--phase-w", dest="phase_w", type=float, default=0.5)
    p.add_argument("--direct-phase-w", dest="direct_phase_w", type=float, default=1.0)
    p.add_argument("--amp-w", dest="amp_w", type=float, default=0.2)
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    p.add_argument("--min-delta", dest="min_delta", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", dest="amp", action="store_true", default=False)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-eval-batches", dest="max_eval_batches", type=int, default=0)
    p.add_argument("--log-every", dest="log_every", type=int, default=25)
    p.add_argument("--smoke", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke:
        args.H = 64
        args.W = 64
        args.n_train = min(args.n_train, 24)
        args.n_val = min(args.n_val, 8)
        args.batch = min(args.batch, 2)
        args.workers = 0
        args.base = min(args.base, 8)
        args.epochs = min(args.epochs, 1)
        args.max_eval_batches = 2
        if args.save_dir == "formal_ordinary_physical_base32":
            args.save_dir = "smoke"
    print(f"Standalone script root: {ROOT}")
    if args.mode == "infer":
        infer(args)
        return
    if args.mode == "auto":
        if args.data_dir:
            data_dir = Path(args.data_dir)
            train_path = data_dir / "train.pt"
            val_path = data_dir / "val.pt"
            if train_path.exists() and val_path.exists():
                print(f"Auto mode: cached data already exists: {data_dir}")
            else:
                print(f"Auto mode: cached data not found, generating first: {data_dir}")
                make_cached_dataset(args)
        else:
            print("Auto mode: no data_dir supplied, using online data generation.")
        train(args)
        return
    if args.mode == "make_data":
        if not args.data_dir:
            raise ValueError("--data-dir is required for --mode make_data")
        make_cached_dataset(args)
        return
    train(args)


if __name__ == "__main__":
    main()
