from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np
from skimage.filters import threshold_otsu

from nvap.config.types import ChannelVolume, DatasetVolume, PSFConfig, PreprocessConfig
from nvap.preprocess.enhancement import (
    postprocess_green_after_deconvolution,
    preprocess_dataset,
    suggest_green_threshold,
)
from nvap.preprocess.missing_slices import fill_channel_missing_slices
from nvap.preprocess.psf import deconvolve_volume
from nvap.preprocess.resample import prepare_mesh_dataset

logger = logging.getLogger(__name__)


def _shared_range(green: ChannelVolume, red: ChannelVolume) -> tuple[int, int]:
    start = max(min(green.z_indices), min(red.z_indices))
    end = min(max(green.z_indices), max(red.z_indices))
    if start > end:
        raise ValueError("No overlapping z range between channels.")
    return start, end


def fill_and_sync_dataset(dataset: DatasetVolume) -> DatasetVolume:
    logger.info("Filling missing slices and syncing channel z-ranges.")
    green = fill_channel_missing_slices(dataset.green)
    red = fill_channel_missing_slices(dataset.red)
    shared = _shared_range(green, red)
    logger.info(
        "Synced dataset: green_z=%d red_z=%d shared_range=%s",
        len(green.z_indices),
        len(red.z_indices),
        shared,
    )
    return DatasetVolume(green=green, red=red, shared_z_range=shared)


def apply_psf_to_dataset(
    dataset: DatasetVolume,
    config: PSFConfig,
    preprocess_config: PreprocessConfig | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> DatasetVolume:
    logger.info("Applying PSF pipeline to both channels.")
    green_data = deconvolve_volume(
        dataset.green.data,
        dataset.green.spacing,
        config,
        cancel_event=cancel_event,
        progress_callback=(
            (lambda current, total: progress_callback("green", current, total))
            if progress_callback is not None
            else None
        ),
    )
    red_data = deconvolve_volume(
        dataset.red.data,
        dataset.red.spacing,
        config,
        cancel_event=cancel_event,
        progress_callback=(
            (lambda current, total: progress_callback("red", current, total))
            if progress_callback is not None
            else None
        ),
    )
    green = ChannelVolume(
        name="green",
        data=green_data,
        z_indices=list(dataset.green.z_indices),
        spacing=dataset.green.spacing,
    )
    red = ChannelVolume(
        name="red",
        data=red_data,
        z_indices=list(dataset.red.z_indices),
        spacing=dataset.red.spacing,
    )
    out = DatasetVolume(green=green, red=red, shared_z_range=dataset.shared_z_range)
    if preprocess_config is not None:
        out = postprocess_green_after_deconvolution(out, preprocess_config)
    return out


def preprocess_for_deconvolution(
    dataset: DatasetVolume,
    preprocess_config: PreprocessConfig,
) -> DatasetVolume:
    logger.info("Applying preprocessing pipeline before deconvolution.")
    return preprocess_dataset(dataset, preprocess_config)


def prepare_dataset_for_mesh(
    dataset: DatasetVolume,
    preprocess_config: PreprocessConfig,
) -> DatasetVolume:
    return prepare_mesh_dataset(dataset, preprocess_config)


def default_threshold(volume: np.ndarray, fallback: float = 0.15) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or float(finite.max()) <= 0.0:
        return fallback
    try:
        value = float(threshold_otsu(finite))
    except ValueError:
        return fallback
    result = float(np.clip(value, 0.0, 1.0))
    logger.debug("Computed Otsu threshold=%.5f fallback=%.5f", result, fallback)
    return result


def default_green_threshold(volume: np.ndarray, fallback: float = 0.15) -> float:
    result = suggest_green_threshold(volume, fallback=fallback)
    logger.debug("Computed branch-aware green threshold=%.5f fallback=%.5f", result, fallback)
    return result
