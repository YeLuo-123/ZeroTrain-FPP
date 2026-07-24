"""Two-stage stereo phase retrieval models for the official UDPR protocol."""
from __future__ import annotations

import torch
from torch import nn

from .models import UNet


class OrdinalFringeOrderNet(nn.Module):
    """Shared-view ordinal predictor with differentiable soft-argmax."""

    def __init__(self, min_order: int = -1, max_order: int = 54, base: int = 32):
        super().__init__()
        if max_order < min_order:
            raise ValueError("max_order must not be below min_order")
        self.min_order = min_order
        self.max_order = max_order
        self.num_orders = max_order - min_order + 1
        self.backbone = UNet(2, self.num_orders, base)
        self.register_buffer(
            "order_values",
            torch.arange(min_order, max_order + 1, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def forward_view(
        self,
        fringe: torch.Tensor,
        wrapped_phase: torch.Tensor,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = torch.cat((fringe, wrapped_phase), dim=1)
        logits = self.backbone(x)
        if lower is not None and upper is not None:
            allowed = (self.order_values >= lower) & (self.order_values <= upper)
            logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        probability = logits.softmax(dim=1)
        order = (probability * self.order_values).sum(dim=1)
        entropy = -(probability.clamp_min(1e-8).log() * probability).sum(dim=1)
        entropy = entropy / torch.log(torch.tensor(self.num_orders, device=entropy.device))
        return {"logits": logits, "probability": probability, "order": order, "uncertainty": entropy}

    def forward(
        self,
        fringe_left: torch.Tensor,
        wrapped_left: torch.Tensor,
        fringe_right: torch.Tensor,
        wrapped_right: torch.Tensor,
        bounds: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        bounds = bounds or {}
        return {
            "left": self.forward_view(
                fringe_left, wrapped_left, bounds.get("lower_left"), bounds.get("upper_left")
            ),
            "right": self.forward_view(
                fringe_right, wrapped_right, bounds.get("lower_right"), bounds.get("upper_right")
            ),
        }


class BoundedPhaseRefinementNet(nn.Module):
    """Predict a residual constrained to the baseline's +/- pi search interval."""

    def __init__(self, base: int = 32, phase_scale: float = 2 * torch.pi * 54):
        super().__init__()
        self.backbone = UNet(2, 2, base)
        self.phase_scale = float(phase_scale)

    def forward_view(self, fringe: torch.Tensor, coarse_phase: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.backbone(torch.cat((fringe, coarse_phase / self.phase_scale), dim=1))
        residual = torch.pi * torch.tanh(raw[:, 0])
        log_variance = raw[:, 1].clamp(-8.0, 8.0)
        return {
            "phase": coarse_phase[:, 0] + residual,
            "residual": residual,
            "log_variance": log_variance,
            "confidence": torch.exp(-log_variance),
        }

    def forward(
        self,
        fringe_left: torch.Tensor,
        coarse_left: torch.Tensor,
        fringe_right: torch.Tensor,
        coarse_right: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "left": self.forward_view(fringe_left, coarse_left),
            "right": self.forward_view(fringe_right, coarse_right),
        }


class TwoStageUDPRNet(nn.Module):
    """Ordinal fringe-order inference followed by bounded phase refinement."""

    def __init__(self, min_order: int = -1, max_order: int = 54, base: int = 32):
        super().__init__()
        self.stage1 = OrdinalFringeOrderNet(min_order, max_order, base)
        self.stage2 = BoundedPhaseRefinementNet(base)

    def forward(
        self,
        fringe_left: torch.Tensor,
        wrapped_left: torch.Tensor,
        fringe_right: torch.Tensor,
        wrapped_right: torch.Tensor,
        bounds: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, object]:
        first = self.stage1(
            fringe_left, wrapped_left, fringe_right, wrapped_right, bounds=bounds
        )
        coarse_left = wrapped_left[:, 0] + 2 * torch.pi * first["left"]["order"]
        coarse_right = wrapped_right[:, 0] + 2 * torch.pi * first["right"]["order"]
        second = self.stage2(
            fringe_left,
            coarse_left.unsqueeze(1),
            fringe_right,
            coarse_right.unsqueeze(1),
        )
        return {
            "stage1": first,
            "stage2": second,
            "coarse_left": coarse_left,
            "coarse_right": coarse_right,
        }
