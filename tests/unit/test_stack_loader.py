from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from nvap.config.types import VoxelSpacing
from nvap.io.stack_loader import load_dataset


def _write_rgb(path: Path, r: int, g: int, b: int) -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    iio.imwrite(path, arr)


def test_load_dataset_sorts_z_and_extracts_channels(tmp_path: Path) -> None:
    green_dir = tmp_path / "Segmented" / "Green"
    red_dir = tmp_path / "Segmented" / "Red"
    green_dir.mkdir(parents=True)
    red_dir.mkdir(parents=True)

    _write_rgb(green_dir / "sample_z002c1.png", r=0, g=40, b=0)
    _write_rgb(green_dir / "sample_z001c1.png", r=0, g=20, b=0)
    _write_rgb(red_dir / "sample_z001c2.png", r=10, g=0, b=0)
    _write_rgb(red_dir / "sample_z002c2.png", r=50, g=0, b=0)

    dataset = load_dataset(tmp_path, spacing=VoxelSpacing())
    assert dataset.green.z_indices == [1, 2]
    assert dataset.red.z_indices == [1, 2]
    assert dataset.green.data.shape == (2, 8, 8)
    assert dataset.red.data.shape == (2, 8, 8)
    assert float(dataset.green.data[0, 0, 0]) < float(dataset.green.data[1, 0, 0])
    assert float(dataset.red.data[0, 0, 0]) < float(dataset.red.data[1, 0, 0])
