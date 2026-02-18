from __future__ import annotations

import numpy as np

from nvap.config.types import ChannelVolume, PreprocessConfig, VoxelSpacing
from nvap.preprocess.enhancement import preprocess_channel


def test_preprocess_preserves_thin_branch_signal() -> None:
    arr = np.zeros((3, 16, 16), dtype=np.float32)
    arr[1, 8, 3:13] = 0.12  # thin microglia-like branch
    arr += 0.01 * np.random.default_rng(0).random(arr.shape, dtype=np.float32)

    channel = ChannelVolume("green", arr, [1, 2, 3], VoxelSpacing())
    cfg = PreprocessConfig(
        denoise_method="anisotropic",
        denoise_strength=0.01,
        preserve_branches=True,
    )
    out = preprocess_channel(channel, cfg)

    # Keep branch-like structure above low-intensity noise floor.
    branch_mean = float(out.data[1, 8, 3:13].mean())
    background_mean = float(out.data[1, 2:6, 2:6].mean())
    assert branch_mean > background_mean
