"""I/O for the official UDPR/UCNNet dataset released with OLE 164 (2023).

The release mixes classic MAT files with MATLAB v7.3 (HDF5) files.  MATLAB
stores HDF5 array dimensions in reverse order, so image arrays on disk are
640x480 but are returned here in the conventional HxW = 480x640 order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset


def _matlab_axes(array: np.ndarray) -> np.ndarray:
    """Restore MATLAB dimension order for data read directly through h5py."""
    array = np.asarray(array).squeeze()
    if array.ndim > 1:
        array = array.transpose(tuple(range(array.ndim - 1, -1, -1)))
    return array


def _first_hdf_dataset(handle: h5py.File) -> np.ndarray:
    arrays: list[np.ndarray] = []

    def collect(_: str | bytes, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            arrays.append(obj[()])

    handle.visititems(collect)
    if not arrays:
        raise ValueError(f"No numeric dataset in {handle.filename}")
    return arrays[0]


def load_mat_array(path: str | Path, key: str | None = None) -> np.ndarray:
    """Load one numeric array from classic MAT, v7.3 MAT, or plain HDF5."""
    path = Path(path)
    try:
        values = loadmat(path)
        if key is not None:
            array = values[key]
        else:
            array = next(v for k, v in values.items() if not k.startswith("__"))
        return np.asarray(array).squeeze()
    except (NotImplementedError, ValueError):
        with h5py.File(path, "r") as handle:
            if key is not None and key in handle:
                array = handle[key][()]
            else:
                array = _first_hdf_dataset(handle)
        return _matlab_axes(array)


@dataclass(frozen=True)
class StereoCalibration:
    camera_matrix_1: np.ndarray
    camera_matrix_2: np.ndarray
    radial_distortion_1: np.ndarray
    radial_distortion_2: np.ndarray
    tangential_distortion_1: np.ndarray
    tangential_distortion_2: np.ndarray
    rotation_2_from_1: np.ndarray
    translation_2_from_1: np.ndarray
    image_size: tuple[int, int]
    source: Path


def _read_hdf_array(group: h5py.Group, name: str) -> np.ndarray:
    return _matlab_axes(group[name][()])


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    """Extract numeric parameters from a MATLAB ``stereoParameters`` object.

    The public files store the object using MATLAB's private MCOS encoding.
    Numeric fields remain ordinary HDF5 datasets; this parser deliberately
    extracts those fields instead of depending on MATLAB.
    """
    path = Path(path)
    with h5py.File(path, "r") as handle:
        intrinsic_groups: list[h5py.Group] = []
        stereo_groups: list[h5py.Group] = []

        def collect(_: str | bytes, obj: Any) -> None:
            if not isinstance(obj, h5py.Group):
                return
            if "IntrinsicMatrix" in obj and "RadialDistortion" in obj:
                intrinsic_groups.append(obj)
            if "RotationOfCamera2" in obj and "TranslationOfCamera2" in obj:
                stereo_groups.append(obj)

        handle.visititems(collect)
        if len(intrinsic_groups) < 2 or not stereo_groups:
            raise ValueError(f"Could not decode stereo calibration in {path}")

        # MCOS writes CameraParameters1 and CameraParameters2 consecutively.
        intrinsic_groups.sort(key=lambda g: g.name)
        c1, c2 = intrinsic_groups[:2]
        stereo = sorted(stereo_groups, key=lambda g: g.name)[0]
        image_size_array = None
        for group in handle.values():
            if isinstance(group, h5py.Group):
                pass
        candidate_sizes: list[np.ndarray] = []

        def find_sizes(_: str | bytes, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.name.endswith("OriginalImageSize"):
                candidate_sizes.append(_matlab_axes(obj[()]))

        handle.visititems(find_sizes)
        if candidate_sizes:
            image_size_array = candidate_sizes[0].astype(int).ravel()
        image_size = (
            tuple(image_size_array.tolist())
            if image_size_array is not None and np.all(image_size_array > 0)
            else (480, 640)
        )

        return StereoCalibration(
            # MATLAB cameraParameters uses a row-vector convention.  Expose
            # conventional OpenCV/column-vector K and R to Python callers.
            camera_matrix_1=_read_hdf_array(c1, "IntrinsicMatrix").T,
            camera_matrix_2=_read_hdf_array(c2, "IntrinsicMatrix").T,
            radial_distortion_1=_read_hdf_array(c1, "RadialDistortion").ravel(),
            radial_distortion_2=_read_hdf_array(c2, "RadialDistortion").ravel(),
            tangential_distortion_1=_read_hdf_array(c1, "TangentialDistortion").ravel(),
            tangential_distortion_2=_read_hdf_array(c2, "TangentialDistortion").ravel(),
            rotation_2_from_1=_read_hdf_array(stereo, "RotationOfCamera2").T,
            translation_2_from_1=_read_hdf_array(stereo, "TranslationOfCamera2").ravel(),
            image_size=(int(image_size[0]), int(image_size[1])),
            source=path,
        )


STAGE_DIRS = {
    1: {
        "fringe_left": "input_fringe pattern_left",
        "fringe_right": "input_fringe pattern_right",
        "coarse_left": "input_coarse wrapped phase_left",
        "coarse_right": "input_coarse wrapped phase_right",
        "target_left": "ground truth_fringe order_left",
        "target_right": "ground truth_fringe order_right",
    },
    2: {
        "fringe_left": "input_fringe pattern_left",
        "fringe_right": "input_fringe pattern_right",
        "coarse_left": "input_coarse absolute phase_left",
        "coarse_right": "input_coarse absolute phase_right",
        "target_left": "ground truth_absolute phase_left",
        "target_right": "ground truth_absolute phase_right",
    },
}


class OfficialUDPRDataset(Dataset):
    """Paired left/right samples from one official section and one stage."""

    def __init__(
        self,
        root: str | Path,
        section: str = "3.2.1",
        stage: int = 1,
        include_output: bool = False,
        crop_size: int | None = None,
        normalize: bool = True,
    ):
        if stage not in STAGE_DIRS:
            raise ValueError("stage must be 1 or 2")
        root = Path(root)
        self.section_root = root / f"section {section}"
        self.data_root = self.section_root / f"data-stage{stage}"
        self.stage = stage
        self.crop_size = crop_size
        self.normalize = normalize
        self.include_output = include_output
        if not self.data_root.exists():
            raise FileNotFoundError(self.data_root)

        required = dict(STAGE_DIRS[stage])
        required.update({"mask_left": "mask_left", "mask_right": "mask_right"})
        if include_output:
            required.update({"output_left": "output_left", "output_right": "output_right"})
        self.directories = {k: self.data_root / v for k, v in required.items()}
        stem_sets = []
        for directory in self.directories.values():
            stems = {p.stem for p in directory.iterdir() if p.suffix.lower() in {".h5", ".mat"}}
            if not stems:
                raise FileNotFoundError(f"No samples in {directory}")
            stem_sets.append(stems)
        self.stems = sorted(set.intersection(*stem_sets), key=lambda s: tuple(int(x) for x in s.split("-")))
        if not self.stems:
            raise RuntimeError(f"No fully paired samples under {self.data_root}")

    def __len__(self) -> int:
        return len(self.stems)

    def _path(self, field: str, stem: str) -> Path:
        matches = list(self.directories[field].glob(stem + ".*"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {field} file for {stem}, found {matches}")
        return matches[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stem = self.stems[index]
        arrays = {
            field: load_mat_array(self._path(field, stem)).astype(np.float32)
            for field in self.directories
        }
        shape = arrays["fringe_left"].shape
        if any(a.shape != shape for a in arrays.values()):
            raise ValueError(f"Inconsistent shapes for sample {stem}")

        if self.crop_size is not None:
            size = self.crop_size
            if min(shape) < size:
                raise ValueError(f"Crop {size} exceeds sample shape {shape}")
            y = (shape[0] - size) // 2
            x = (shape[1] - size) // 2
            arrays = {k: v[y:y + size, x:x + size] for k, v in arrays.items()}

        if self.normalize:
            arrays["fringe_left"] /= 255.0
            arrays["fringe_right"] /= 255.0
            # Phase stays in radians: stage 1 reconstructs phi + 2*pi*K and
            # stage 2 constrains its residual in physical +/- pi units.

        result: dict[str, torch.Tensor | str] = {
            k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()
        }
        result["stem"] = stem
        return result

    def calibration(self, pair: str = "leftcamera_rightcamera") -> StereoCalibration:
        return load_stereo_calibration(self.section_root / "calibration parameter" / f"{pair}.mat")

    def fringe_order_bounds(self, view: str) -> tuple[np.ndarray, np.ndarray]:
        if view not in {"left", "right"}:
            raise ValueError("view must be left or right")
        root = self.section_root / "fringe order range"
        return (
            load_mat_array(root / f"{view}_fringe order min.mat"),
            load_mat_array(root / f"{view}_fringe order max.mat"),
        )
