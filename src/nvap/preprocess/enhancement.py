from __future__ import annotations

import logging

import numpy as np
import scipy.ndimage as ndi
from skimage.filters import meijering, threshold_otsu
from skimage.restoration import denoise_bilateral, denoise_nl_means

from nvap.config.types import ChannelVolume, DatasetVolume, PreprocessConfig
from nvap.preprocess.denoisers import estimate_noise_sigma, run_green_denoiser

logger = logging.getLogger(__name__)


def stage_illumination_correction(volume: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    if config.flatfield_sigma_xy <= 0:
        return np.asarray(volume, dtype=np.float32).copy()
    background = ndi.gaussian_filter(
        np.asarray(volume, dtype=np.float32),
        sigma=(0.0, float(config.flatfield_sigma_xy), float(config.flatfield_sigma_xy)),
        mode="nearest",
    )
    corrected = np.asarray(volume, dtype=np.float32) - background
    corrected -= float(np.min(corrected))
    return corrected.astype(np.float32, copy=False)


def stage_intensity_stabilization(volume: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    out = np.empty_like(volume, dtype=np.float32)
    for z in range(volume.shape[0]):
        plane = np.asarray(volume[z], dtype=np.float32)
        low = float(np.percentile(plane, float(config.contrast_low_pct)))
        high = float(np.percentile(plane, float(config.contrast_high_pct)))
        if high <= low:
            out[z] = np.zeros_like(plane, dtype=np.float32)
            continue
        out[z] = np.clip((plane - low) / (high - low), 0.0, 1.0)
    return out


def stage_noise_model_estimation(volume: np.ndarray, config: PreprocessConfig) -> dict[str, float | str]:
    sigma = estimate_noise_sigma(volume)
    if config.green_noise_model == "auto":
        noise_model = "poisson_gaussian" if float(np.max(volume)) > 0.2 else "gaussian"
    else:
        noise_model = config.green_noise_model
    return {"sigma": float(sigma), "noise_model": str(noise_model)}


def stage_branch_map_estimation(
    volume: np.ndarray,
    config: PreprocessConfig,
    channel_name: str,
) -> np.ndarray:
    if channel_name != "green" or not config.preserve_branches:
        return np.zeros_like(volume, dtype=np.float32)
    try:
        response = meijering(
            np.asarray(volume, dtype=np.float32),
            sigmas=[0.8, 1.2, 1.8],
            black_ridges=False,
            mode="reflect",
        )
        branch = np.asarray(response, dtype=np.float32)
    except Exception as exc:
        logger.debug("Branch map estimation failed, using zeros: %s", exc)
        return np.zeros_like(volume, dtype=np.float32)

    max_val = float(np.nanmax(branch)) if branch.size else 0.0
    if not np.isfinite(max_val) or max_val <= 0.0:
        return np.zeros_like(volume, dtype=np.float32)
    branch /= max_val
    branch = ndi.gaussian_filter(branch, sigma=(0.0, 0.8, 0.8), mode="nearest")
    return np.clip(branch, 0.0, 1.0).astype(np.float32, copy=False)


def _denoise_volume_default(
    volume: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> np.ndarray:
    method = config.denoise_method
    if method == "none":
        return np.asarray(volume, dtype=np.float32).copy()

    if method == "anisotropic":
        sigma_xy = float(max(denoise_strength * 24.0, 0.15))
        sigma_z = sigma_xy * 0.55
        return ndi.gaussian_filter(
            np.asarray(volume, dtype=np.float32),
            sigma=(sigma_z, sigma_xy, sigma_xy),
            mode="nearest",
        ).astype(np.float32, copy=False)

    if method == "non_local_means":
        patch_kw = dict(patch_size=3, patch_distance=4, fast_mode=True, channel_axis=None)
        h = float(max(denoise_strength * 0.9, 0.004))
        out = denoise_nl_means(
            np.asarray(volume, dtype=np.float32),
            h=h,
            sigma=0.0,
            preserve_range=True,
            **patch_kw,
        )
        return np.asarray(out, dtype=np.float32)

    if method == "bilateral":
        out = np.empty_like(volume, dtype=np.float32)
        sigma_color = float(max(denoise_strength * 1.4, 0.01))
        sigma_spatial = 2.0
        for z in range(volume.shape[0]):
            out[z] = np.asarray(
                denoise_bilateral(
                    np.asarray(volume[z], dtype=np.float32),
                    sigma_color=sigma_color,
                    sigma_spatial=sigma_spatial,
                    channel_axis=None,
                ),
                dtype=np.float32,
            )
        return out

    raise ValueError(f"Unsupported denoise method: {method}")


def stage_denoise_main(
    volume: np.ndarray,
    channel_name: str,
    config: PreprocessConfig,
    denoise_strength: float,
    branch_map: np.ndarray,
) -> tuple[np.ndarray, str]:
    if channel_name == "green":
        denoised, backend = run_green_denoiser(
            np.asarray(volume, dtype=np.float32),
            np.asarray(branch_map, dtype=np.float32),
            config=config,
            denoise_strength=denoise_strength,
        )
        return denoised, backend
    out = _denoise_volume_default(np.asarray(volume, dtype=np.float32), config, denoise_strength)
    return out.astype(np.float32, copy=False), config.denoise_method


def _attenuate_small_isolated_components(
    volume: np.ndarray,
    threshold: float,
    min_voxels: int,
    attenuation: float,
    exempt_mask: np.ndarray | None = None,
) -> np.ndarray:
    min_voxels = max(2, int(min_voxels))
    attenuation = float(np.clip(attenuation, 0.0, 1.0))
    threshold = float(np.clip(threshold, 0.0, 1.0))
    mask = np.asarray(volume, dtype=np.float32) >= threshold
    if not np.any(mask):
        return np.asarray(volume, dtype=np.float32).copy()

    labels, count = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count <= 0:
        return np.asarray(volume, dtype=np.float32).copy()

    sizes = np.bincount(labels.ravel())
    small_labels = np.where((sizes > 0) & (sizes < min_voxels))[0]
    small_labels = small_labels[small_labels != 0]
    if small_labels.size == 0:
        return np.asarray(volume, dtype=np.float32).copy()

    small_mask = np.isin(labels, small_labels)
    if exempt_mask is not None:
        small_mask &= ~np.asarray(exempt_mask, dtype=bool)
    out = np.asarray(volume, dtype=np.float32).copy()
    out[small_mask] *= attenuation
    return out


def stage_speckle_control(
    volume: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    channel_name: str,
) -> np.ndarray:
    if channel_name != "green":
        return np.asarray(volume, dtype=np.float32)
    arr = np.asarray(volume, dtype=np.float32)
    branch = np.asarray(branch_map, dtype=np.float32)

    speckle_threshold = float(max(0.08, np.quantile(arr, 0.82) * 0.75))
    # Preserve likely filaments: strong branch response and local contrast.
    local_mean = ndi.uniform_filter(arr, size=(1, 5, 5), mode="nearest")
    local_contrast = np.maximum(arr - local_mean, 0.0)
    contrast_gate = local_contrast >= np.quantile(local_contrast, 0.55)
    branch_candidate = (branch >= float(config.green_branch_protection) * 0.65) & contrast_gate
    labels, count = ndi.label(branch_candidate, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count > 0:
        sizes = np.bincount(labels.ravel())
        keep = np.where(sizes >= max(6, int(config.green_speckle_min_voxels)))[0]
        keep = keep[keep != 0]
        branch_exempt = np.isin(labels, keep) if keep.size > 0 else np.zeros_like(branch_candidate)
    else:
        branch_exempt = np.zeros_like(branch_candidate)

    out = _attenuate_small_isolated_components(
        arr,
        threshold=speckle_threshold,
        min_voxels=int(config.green_speckle_min_voxels),
        attenuation=float(config.green_speckle_attenuation),
        exempt_mask=branch_exempt,
    )
    # Secondary peak clamp for isolated bright spikes that survive component attenuation.
    peak_strength = np.maximum(arr - local_mean, 0.0)
    peak_cut = float(np.quantile(peak_strength, 0.97))
    peak_mask = (
        (arr >= ndi.maximum_filter(arr, size=(1, 3, 3), mode="nearest"))
        & (peak_strength >= peak_cut)
        & (~branch_exempt)
    )
    if np.any(peak_mask):
        clamp = float(np.quantile(out, 0.70) * 0.6)
        out[peak_mask] = np.minimum(out[peak_mask], clamp)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def preprocess_channel(channel: ChannelVolume, config: PreprocessConfig) -> ChannelVolume:
    if not config.enabled:
        return ChannelVolume(
            name=channel.name,
            data=np.asarray(channel.data, dtype=np.float32).copy(),
            z_indices=list(channel.z_indices),
            spacing=channel.spacing,
        )

    denoise_strength = float(config.denoise_strength)
    if channel.name == "green":
        denoise_strength *= float(max(config.green_denoise_multiplier, 0.1))
        denoise_strength *= float(max(config.green_pre_deconv_strength, 0.0))

    logger.info(
        "Preprocessing channel '%s' strategy=%s strength=%.5f",
        channel.name,
        config.green_denoise_strategy if channel.name == "green" else config.denoise_method,
        denoise_strength,
    )
    working = stage_illumination_correction(channel.data, config)
    stabilized = stage_intensity_stabilization(working, config)
    noise = stage_noise_model_estimation(stabilized, config)
    branch_map = stage_branch_map_estimation(stabilized, config, channel.name)
    denoised, backend = stage_denoise_main(
        stabilized,
        channel.name,
        config=config,
        denoise_strength=denoise_strength,
        branch_map=branch_map,
    )
    logger.info(
        "Denoise backend channel=%s backend=%s sigma=%.5f noise_model=%s",
        channel.name,
        backend,
        float(noise["sigma"]),
        str(noise["noise_model"]),
    )

    if config.preserve_branches and channel.name == "green":
        floor = (0.44 + (0.28 * float(config.green_branch_protection))) * branch_map
        branch_floor = stabilized * np.clip(floor, 0.15, 0.9)
        denoised = np.maximum(denoised, branch_floor)
        branch_mask = branch_map >= max(0.6, float(config.green_branch_protection) * 0.9)
        labels, count = ndi.label(branch_mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
        sizes = np.bincount(labels.ravel()) if count > 0 else np.array([], dtype=np.int64)
        large_labels = np.where(sizes >= max(6, int(config.green_speckle_min_voxels)))[0]
        large_labels = large_labels[large_labels != 0]
        branch_boost_mask = np.isin(labels, large_labels) if large_labels.size > 0 else np.zeros_like(branch_mask)
        if np.any(branch_boost_mask):
            boosted = stabilized * (0.72 + (0.18 * float(config.green_branch_protection)))
            denoised[branch_boost_mask] = np.maximum(
                denoised[branch_boost_mask],
                boosted[branch_boost_mask],
            )

    denoised = stage_speckle_control(denoised, branch_map, config, channel.name)
    return ChannelVolume(
        name=channel.name,
        data=np.asarray(denoised, dtype=np.float32),
        z_indices=list(channel.z_indices),
        spacing=channel.spacing,
    )


def preprocess_dataset(dataset: DatasetVolume, config: PreprocessConfig) -> DatasetVolume:
    green = preprocess_channel(dataset.green, config)
    red = preprocess_channel(dataset.red, config)
    return DatasetVolume(green=green, red=red, shared_z_range=dataset.shared_z_range)


def postprocess_green_after_deconvolution(
    dataset: DatasetVolume,
    config: PreprocessConfig,
) -> DatasetVolume:
    if not config.enabled or float(config.green_post_deconv_strength) <= 0.0:
        return dataset

    green = dataset.green
    green_data = np.clip(np.asarray(green.data, dtype=np.float32), 0.0, 1.0)
    branch_map = stage_branch_map_estimation(green_data, config, "green")
    strength = (
        float(config.denoise_strength)
        * float(max(config.green_denoise_multiplier, 0.1))
        * float(max(config.green_post_deconv_strength, 0.0))
    )
    denoised, backend = stage_denoise_main(
        green_data,
        "green",
        config=config,
        denoise_strength=strength,
        branch_map=branch_map,
    )
    logger.info("Post-deconvolution green denoise backend=%s strength=%.5f", backend, strength)
    denoised = stage_speckle_control(denoised, branch_map, config, "green")
    out_green = ChannelVolume(
        name="green",
        data=np.asarray(denoised, dtype=np.float32),
        z_indices=list(green.z_indices),
        spacing=green.spacing,
    )
    return DatasetVolume(green=out_green, red=dataset.red, shared_z_range=dataset.shared_z_range)


def suggest_green_threshold(volume: np.ndarray, fallback: float = 0.15) -> float:
    arr = np.asarray(volume, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(fallback)
    try:
        otsu = float(threshold_otsu(finite))
    except ValueError:
        otsu = float(fallback)
    branch_map = stage_branch_map_estimation(arr, PreprocessConfig(), "green")
    branch_vals = arr[branch_map > 0.35]
    if branch_vals.size == 0:
        return float(np.clip(otsu, 0.0, 1.0))
    floor = float(np.percentile(branch_vals, 35))
    result = min(otsu, floor * 1.2)
    return float(np.clip(max(0.03, result), 0.0, 1.0))
