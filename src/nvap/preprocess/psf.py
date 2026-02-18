from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Callable

import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndi
from scipy.signal import fftconvolve

from nvap.config.types import PSFConfig, VoxelSpacing

logger = logging.getLogger(__name__)


class OperationCanceledError(RuntimeError):
    """Raised when a long-running operation is canceled by the user."""


def build_gaussian_psf(
    spacing: VoxelSpacing,
    sigma_xy_um: float,
    sigma_z_um: float,
    truncate: float = 3.0,
) -> np.ndarray:
    sigma_x = max(sigma_xy_um / spacing.x_um, 1e-6)
    sigma_y = max(sigma_xy_um / spacing.y_um, 1e-6)
    sigma_z = max(sigma_z_um / spacing.z_um, 1e-6)

    rx = max(1, int(np.ceil(truncate * sigma_x)))
    ry = max(1, int(np.ceil(truncate * sigma_y)))
    rz = max(1, int(np.ceil(truncate * sigma_z)))

    z, y, x = np.mgrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]
    exponent = -0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2 + (z / sigma_z) ** 2)
    psf = np.exp(exponent).astype(np.float32)
    psf_sum = float(psf.sum())
    if psf_sum <= 0:
        raise ValueError("Invalid PSF kernel generated.")
    psf /= psf_sum
    logger.debug(
        "Built Gaussian PSF: shape=%s sigma_xy_um=%.4f sigma_z_um=%.4f",
        psf.shape,
        sigma_xy_um,
        sigma_z_um,
    )
    return psf


def _load_measured_psf_from_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if "psf" in data:
                arr = data["psf"]
            else:
                first_key = list(data.keys())[0]
                arr = data[first_key]
    else:
        arr = iio.imread(path)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Measured PSF must be 3D, got shape={arr.shape}")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("Measured PSF contains no positive signal.")
    arr /= total
    return arr


def _resolve_psf_kernel(config: PSFConfig, spacing: VoxelSpacing) -> np.ndarray:
    if config.use_measured_psf and config.measured_psf_path.strip():
        psf_path = Path(config.measured_psf_path).expanduser().resolve()
        if psf_path.exists():
            logger.info("Using measured PSF: %s", psf_path)
            return _load_measured_psf_from_file(psf_path)
        logger.warning("Measured PSF path not found: %s. Falling back to Gaussian PSF.", psf_path)

    return build_gaussian_psf(
        spacing=spacing,
        sigma_xy_um=config.sigma_xy_um,
        sigma_z_um=config.sigma_z_um,
    )


def deconvolve_volume(
    volume: np.ndarray,
    spacing: VoxelSpacing,
    config: PSFConfig,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError("volume must be 3D.")

    if not config.enabled or config.iterations <= 0:
        logger.info(
            "PSF deconvolution skipped (enabled=%s iterations=%d).",
            config.enabled,
            config.iterations,
        )
        return volume.astype(np.float32, copy=True)

    logger.info(
        "Running Richardson-Lucy: iterations=%d sigma_xy_um=%.4f sigma_z_um=%.4f volume_shape=%s",
        config.iterations,
        config.sigma_xy_um,
        config.sigma_z_um,
        volume.shape,
    )
    psf = _resolve_psf_kernel(config, spacing)

    image = volume.astype(np.float32, copy=False)
    estimate = np.maximum(image.copy(), 1e-6)
    psf_mirror = psf[::-1, ::-1, ::-1]
    eps = np.float32(1e-7)
    total = int(config.iterations)
    reg = float(max(config.regularization_lambda, 0.0))

    for idx in range(total):
        if cancel_event is not None and cancel_event.is_set():
            logger.info("PSF deconvolution canceled at iteration %d/%d.", idx, total)
            raise OperationCanceledError("PSF deconvolution canceled by user.")

        conv = fftconvolve(estimate, psf, mode="same")
        relative_blur = image / np.maximum(conv, eps)
        estimate *= fftconvolve(relative_blur, psf_mirror, mode="same")
        if reg > 0.0:
            smooth = ndi.gaussian_filter(estimate, sigma=(0.4, 0.45, 0.45), mode="nearest")
            estimate = ((1.0 - reg) * estimate) + (reg * smooth)
        np.maximum(estimate, 0.0, out=estimate)

        if progress_callback is not None:
            progress_callback(idx + 1, total)

    deconv = estimate.astype(np.float32, copy=False)
    np.maximum(deconv, 0.0, out=deconv)
    logger.debug("PSF deconvolution complete.")
    return deconv
