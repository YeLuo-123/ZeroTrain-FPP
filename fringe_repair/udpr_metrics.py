"""Metrics used to reproduce the official UCNNet release results."""
from __future__ import annotations

import numpy as np


class MetricAccumulator:
    def __init__(self, stage: int):
        self.stage = stage
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.count = 0
        self.threshold_counts = {0.1: 0, 0.2: 0, 0.5: 0}
        self.correct_order = 0

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        valid = np.asarray(mask) > 0.5
        finite = np.isfinite(prediction) & np.isfinite(target)
        valid &= finite
        error = np.asarray(prediction)[valid] - np.asarray(target)[valid]
        absolute = np.abs(error)
        self.absolute_error += float(absolute.sum())
        self.squared_error += float(np.square(error).sum())
        self.count += int(error.size)
        for threshold in self.threshold_counts:
            self.threshold_counts[threshold] += int((absolute > threshold).sum())
        if self.stage == 1:
            self.correct_order += int(
                (np.rint(prediction[valid]) == np.rint(target[valid])).sum()
            )

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            raise RuntimeError("No valid pixels were evaluated")
        result: dict[str, float | int] = {
            "valid_pixels": self.count,
            "mae": self.absolute_error / self.count,
            "rmse": (self.squared_error / self.count) ** 0.5,
        }
        for threshold, count in self.threshold_counts.items():
            result[f"bad_{threshold:g}_rate"] = count / self.count
        if self.stage == 1:
            result["rounded_order_accuracy"] = self.correct_order / self.count
        return result
