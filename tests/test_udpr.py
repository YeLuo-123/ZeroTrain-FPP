from pathlib import Path

import numpy as np
import pytest
import torch

from fringe_repair.udpr_io import OfficialUDPRDataset, load_mat_array, load_stereo_calibration
from fringe_repair.udpr_metrics import MetricAccumulator
from fringe_repair.udpr_models import TwoStageUDPRNet


ROOT = Path(__file__).parents[1]


def test_official_loader_real_sample():
    dataset_root = ROOT / "Dataset"
    if not dataset_root.exists():
        pytest.skip("official dataset is not installed")
    dataset = OfficialUDPRDataset(dataset_root, section="3.2.1", stage=1)
    sample = dataset[0]
    assert len(dataset) == 100
    assert sample["fringe_left"].shape == (480, 640)
    assert sample["target_left"].shape == (480, 640)
    assert 0 <= float(sample["fringe_left"].min()) <= float(sample["fringe_left"].max()) <= 1
    calibration = dataset.calibration()
    assert calibration.camera_matrix_1.shape == (3, 3)
    assert calibration.camera_matrix_1[0, 2] > 0
    assert calibration.rotation_2_from_1.shape == (3, 3)
    assert calibration.translation_2_from_1.shape == (3,)
    assert calibration.image_size == (480, 640)


def test_metrics_mask_and_order_accuracy():
    meter = MetricAccumulator(stage=1)
    prediction = np.array([[1.1, 4.0], [3.0, 9.0]])
    target = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[1, 0], [1, 0]])
    meter.update(prediction, target, mask)
    result = meter.compute()
    assert result["valid_pixels"] == 2
    assert result["mae"] == pytest.approx(0.05)
    assert result["rounded_order_accuracy"] == 1.0


def test_two_stage_shapes_and_bounds():
    model = TwoStageUDPRNet(base=4)
    fringe = torch.rand(2, 1, 32, 40)
    wrapped = torch.rand(2, 1, 32, 40) * 2 * torch.pi - torch.pi
    output = model(fringe, wrapped, fringe, wrapped)
    assert output["stage1"]["left"]["logits"].shape == (2, 56, 32, 40)
    residual = output["stage2"]["left"]["residual"]
    assert residual.shape == (2, 32, 40)
    assert torch.all(residual.abs() <= torch.pi)
