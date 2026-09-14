from pathlib import Path
import sys

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "improvement" / "code"
sys.path.insert(0, str(CODE))

from boundary_multitask import (  # noqa: E402
    FREQUENCY_RATIO, BoundaryMultiTaskUNet, make_input, multitask_loss,
)


def test_teacher_frequency_ratio_is_preserved():
    assert FREQUENCY_RATIO == 32


def test_input_output_and_multitask_loss_shapes():
    h, w = 32, 40
    phi = np.linspace(0, 4 * np.pi, h * w, endpoint=False).reshape(h, w)
    sample = {
        "input_f2_3step": np.stack([.5 + .4 * np.cos(phi + 2*np.pi*k/3) for k in range(3)]).astype(np.float32),
        "input_f64_3step": np.stack([.5 + .4 * np.cos(32*phi + 2*np.pi*k/3) for k in range(3)]).astype(np.float32),
        "wrap_boundary_observed": np.zeros((h, w), np.uint8),
    }
    x = make_input(sample)
    assert x.shape == (10, h, w)
    model = BoundaryMultiTaskUNet(base=8)
    output = model(torch.from_numpy(x)[None])
    assert output["boundary_logits"].shape == (1, 1, h, w)
    assert output["k2_logits"].shape == (1, 2, h, w)
    assert output["phase_pair"].shape == (1, 2, h, w)
    assert output["confidence_logits"].shape == (1, 1, h, w)
    batch = {
        "boundary": torch.zeros(1, 1, h, w),
        "k2": torch.zeros(1, h, w, dtype=torch.long),
        "phase_pair": torch.stack([torch.cos(torch.from_numpy(phi)), torch.sin(torch.from_numpy(phi))], 0).float()[None],
        "valid": torch.ones(1, 1, h, w),
    }
    loss, parts = multitask_loss(output, batch, {name: 1.0 for name in ("boundary", "k2", "phase", "confidence", "topology")})
    assert torch.isfinite(loss)
    assert set(parts) == {"boundary", "k2", "phase", "confidence", "topology"}
