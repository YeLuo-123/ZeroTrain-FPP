import numpy as np

SHIFTS = np.arange(4, dtype=np.float32) * (np.pi / 2)


def wrap_phase(x):
    """Wrap radians to [-pi, pi). Works for NumPy arrays and torch tensors."""
    if hasattr(x, "remainder"):
        import torch
        return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi
    return (x + np.pi) % (2 * np.pi) - np.pi


def psp4(fringe, eps=1e-8):
    """Four-step phase shifting: I_k=A+B cos(phi+k*pi/2)."""
    if hasattr(fringe, "atan2"):
        import torch
        return torch.atan2(fringe[..., 3, :, :] - fringe[..., 1, :, :],
                           fringe[..., 0, :, :] - fringe[..., 2, :, :] + eps)
    fringe = np.asarray(fringe)
    return np.arctan2(fringe[..., 3, :, :] - fringe[..., 1, :, :],
                      fringe[..., 0, :, :] - fringe[..., 2, :, :] + eps)


def synthesize_fringe(phase, amplitude=0.45, background=0.5):
    phase = np.asarray(phase, np.float32)
    return np.stack([background + amplitude * np.cos(phase + d) for d in SHIFTS])


def phase_to_depth(phase, scale=1.0, offset=0.0):
    """Linear calibrated surrogate. Real use must supply calibrated scale/offset."""
    return scale * phase + offset
