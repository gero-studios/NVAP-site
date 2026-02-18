from __future__ import annotations

import logging

import numpy as np
import scipy.ndimage as ndi
from skimage.restoration import denoise_bilateral, denoise_nl_means

from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig

logger = logging.getLogger(__name__)


def _flatfield_background_correction(volume: np.ndarray, sigma_xy: float) -> np.ndarray:
    if sigma_xy <= 0:
        return volume.astype(np.float32, copy=True)
    background = ndi.gaussian_filter(
        volume.astype(np.float32, copy=False),
        sigma=(0.0, float(sigma_xy), float(sigma_xy)),
        mode="nearest",
    )
    corrected = volume.astype(np.float32, copy=False) - background
    corrected -= float(np.min(corrected))
    return corrected


def _contrast_normalize_per_slice(
    volume: np.ndarray, low_pct: float, high_pct: float
) -> np.ndarray:
    out = np.empty_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        plane = volume[z].astype(np.float32, copy=False)
        low = float(np.percentile(plane, low_pct))
        high = float(np.percentile(plane, high_pct))
        if high <= low:
            out[z] = np.zeros_like(plane, dtype=np.float32)
            continue
        norm = (plane - low) / (high - low)
        out[z] = np.clip(norm, 0.0, 1.0)
    return out


def _denoise_volume(
    volume: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> np.ndarray:
    method = config.denoise_method
    if method == "none":
        return volume.astype(np.float32, copy=True)

    if method == "anisotropic":
        # Fast anisotropic smoothing (z-smaller than xy) to reduce noise while preserving thin branches.
        sigma_xy = float(max(denoise_strength * 24.0, 0.15))
        sigma_z = sigma_xy * 0.55
        denoised = ndi.gaussian_filter(
            volume.astype(np.float32, copy=False),
            sigma=(sigma_z, sigma_xy, sigma_xy),
            mode="nearest",
        )
        return denoised.astype(np.float32, copy=False)

    if method == "non_local_means":
        patch_kw = dict(patch_size=3, patch_distance=4, fast_mode=True, channel_axis=None)
        h = float(max(denoise_strength * 0.9, 0.004))
        denoised = denoise_nl_means(
            volume.astype(np.float32, copy=False),
            h=h,
            sigma=0.0,
            preserve_range=True,
            **patch_kw,
        )
        return denoised.astype(np.float32, copy=False)

    if method == "bilateral":
        # Bilateral is applied slice-wise to preserve local microglia branch boundaries.
        out = np.empty_like(volume, dtype=np.float32)
        sigma_color = float(max(denoise_strength * 1.4, 0.01))
        sigma_spatial = 2.0
        for z in range(volume.shape[0]):
            out[z] = denoise_bilateral(
                volume[z].astype(np.float32, copy=False),
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                channel_axis=None,
            ).astype(np.float32, copy=False)
        return out

    raise ValueError(f"Unsupported denoise method: {method}")


def preprocess_channel(channel: ChannelVolume, config: PreprocessConfig) -> ChannelVolume:
    if not config.enabled:
        return ChannelVolume(
            name=channel.name,
            data=channel.data.astype(np.float32, copy=True),
            z_indices=list(channel.z_indices),
            spacing=channel.spacing,
        )

    denoise_strength = float(config.denoise_strength)
    if channel.name == "green":
        denoise_strength *= float(max(config.green_denoise_multiplier, 0.1))
    logger.info(
        "Preprocessing channel '%s' with method=%s denoise_strength=%.5f",
        channel.name,
        config.denoise_method,
        denoise_strength,
    )
    working = _flatfield_background_correction(channel.data, sigma_xy=config.flatfield_sigma_xy)
    normalized = _contrast_normalize_per_slice(
        working,
        low_pct=float(config.contrast_low_pct),
        high_pct=float(config.contrast_high_pct),
    )
    working = _denoise_volume(normalized, config, denoise_strength=denoise_strength)

    if config.preserve_branches and channel.name == "green":
        # Preserve faint branch-like structures by preventing denoising from suppressing
        # low-to-mid intensity filaments too aggressively.
        branch_floor = normalized * np.float32(0.78)
        working = np.maximum(working, branch_floor)
        working = np.clip(working * 1.02, 0.0, 1.0).astype(np.float32, copy=False)

    return ChannelVolume(
        name=channel.name,
        data=working.astype(np.float32, copy=False),
        z_indices=list(channel.z_indices),
        spacing=channel.spacing,
    )


def preprocess_dataset(dataset: DatasetVolume, config: PreprocessConfig) -> DatasetVolume:
    green = preprocess_channel(dataset.green, config)
    red = preprocess_channel(dataset.red, config)
    return DatasetVolume(green=green, red=red, shared_z_range=dataset.shared_z_range)
