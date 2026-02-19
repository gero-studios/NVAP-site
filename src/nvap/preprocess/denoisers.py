from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.ndimage as ndi
from skimage.restoration import denoise_nl_means

from nvap.config.types import PreprocessConfig

logger = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """Raised when an optional denoising backend cannot run."""


def estimate_noise_sigma(volume: np.ndarray) -> float:
    """Robust sigma estimate from high-pass residuals."""
    arr = np.asarray(volume, dtype=np.float32)
    smooth = ndi.gaussian_filter(arr, sigma=(0.0, 1.2, 1.2), mode="nearest")
    residual = arr - smooth
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    sigma = mad / 0.6745 if mad > 0 else 0.0
    return float(max(sigma, 1.0e-4))


def anscombe_forward(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    return 2.0 * np.sqrt(np.clip(arr, 0.0, None) + 3.0 / 8.0)


def anscombe_inverse(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    return np.clip((arr * 0.5) ** 2 - 3.0 / 8.0, 0.0, None)


def denoise_legacy_anisotropic(volume: np.ndarray, denoise_strength: float) -> np.ndarray:
    sigma_xy = float(max(denoise_strength * 24.0, 0.15))
    sigma_z = sigma_xy * 0.55
    out = ndi.gaussian_filter(
        np.asarray(volume, dtype=np.float32),
        sigma=(sigma_z, sigma_xy, sigma_xy),
        mode="nearest",
    )
    return np.asarray(out, dtype=np.float32)


def _chunk_slices(depth: int, chunk_depth: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if depth <= chunk_depth or chunk_depth <= 0:
        return [(0, depth, 0, depth)]
    segments: list[tuple[int, int, int, int]] = []
    step = max(1, chunk_depth - (2 * max(0, overlap)))
    start = 0
    while start < depth:
        end = min(depth, start + chunk_depth)
        keep_start = start + (0 if start == 0 else overlap)
        keep_end = end - (0 if end == depth else overlap)
        segments.append((start, end, keep_start, keep_end))
        if end >= depth:
            break
        start += step
    return segments


def run_in_chunks(
    volume: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    chunk_depth: int,
    overlap: int,
) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    depth = int(arr.shape[0])
    pieces = _chunk_slices(depth, int(chunk_depth), int(overlap))
    if len(pieces) == 1:
        return np.asarray(fn(arr), dtype=np.float32)

    out = np.zeros_like(arr, dtype=np.float32)
    weights = np.zeros_like(arr, dtype=np.float32)
    for start, end, keep_start, keep_end in pieces:
        denoised = np.asarray(fn(arr[start:end]), dtype=np.float32)
        local_start = keep_start - start
        local_end = keep_end - start
        out[keep_start:keep_end] += denoised[local_start:local_end]
        weights[keep_start:keep_end] += 1.0
    weights = np.maximum(weights, 1.0)
    return out / weights


def denoise_classical_branch_aware(
    volume: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> np.ndarray:
    arr = np.asarray(volume, dtype=np.float32)
    branch = np.clip(np.asarray(branch_map, dtype=np.float32), 0.0, 1.0)

    noise_sigma = estimate_noise_sigma(arr)
    working = arr
    if config.green_apply_vst and config.green_noise_model != "gaussian":
        working = anscombe_forward(working)
        noise_sigma = estimate_noise_sigma(working)

    patch_kw = dict(
        patch_size=max(1, int(config.green_nlm_patch_size)),
        patch_distance=max(1, int(config.green_nlm_patch_distance)),
        fast_mode=True,
        channel_axis=None,
    )
    base_h = float(max(noise_sigma * float(config.green_nlm_h_factor), 0.002))
    strong_h = float(base_h * (1.0 + (0.9 * denoise_strength)))
    mild_h = float(base_h * max(0.35, 0.8 - (0.35 * denoise_strength)))

    strong = denoise_nl_means(working, h=strong_h, sigma=0.0, preserve_range=True, **patch_kw)
    mild = denoise_nl_means(working, h=mild_h, sigma=0.0, preserve_range=True, **patch_kw)
    blended = (branch * mild) + ((1.0 - branch) * strong)

    if config.green_apply_vst and config.green_noise_model != "gaussian":
        blended = anscombe_inverse(blended)

    post = ndi.gaussian_filter(
        np.asarray(blended, dtype=np.float32),
        sigma=(0.12, 0.3, 0.3),
        mode="nearest",
    )
    return np.clip(post, 0.0, 1.0).astype(np.float32, copy=False)


def _denoise_bm4d_impl(volume: np.ndarray, sigma: float) -> np.ndarray:
    try:
        import bm4d  # type: ignore
    except ImportError as exc:
        raise BackendUnavailableError("bm4d package not installed.") from exc

    arr = np.asarray(volume, dtype=np.float32)
    try:
        denoised = bm4d.bm4d(arr, sigma_psd=float(max(sigma, 1.0e-4)))
    except Exception as exc:
        raise BackendUnavailableError(f"bm4d backend failed: {exc}") from exc
    return np.asarray(denoised, dtype=np.float32)


def _denoise_noise2void_torch(volume: np.ndarray, model_path: str) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError("torch is not installed for noise2void backend.") from exc

    model_file = Path(model_path).expanduser().resolve()
    if not model_file.exists():
        raise BackendUnavailableError(f"noise2void model not found: {model_file}")

    try:
        model = torch.jit.load(str(model_file), map_location="cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
    except Exception as exc:
        raise BackendUnavailableError(f"Failed to load noise2void model: {exc}") from exc

    arr = np.asarray(volume, dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode():
        tensor = torch.from_numpy(arr[None, None]).to(device=device, dtype=torch.float32)
        output = model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        out = output.detach().to("cpu").numpy()[0, 0]
    return np.asarray(out, dtype=np.float32)


def run_green_denoiser(
    volume: np.ndarray,
    branch_map: np.ndarray,
    config: PreprocessConfig,
    denoise_strength: float,
) -> tuple[np.ndarray, str]:
    arr = np.asarray(volume, dtype=np.float32)
    strategy = config.green_denoise_strategy

    def run_classical(v: np.ndarray) -> np.ndarray:
        return denoise_classical_branch_aware(v, branch_map=np.clip(branch_map[: v.shape[0]], 0.0, 1.0), config=config, denoise_strength=denoise_strength)

    if strategy == "legacy_anisotropic":
        return denoise_legacy_anisotropic(arr, denoise_strength=denoise_strength), "legacy_anisotropic"

    if strategy == "classical_branch_aware":
        return run_classical(arr), "classical_branch_aware"

    if strategy == "noise2void":
        try:
            if not config.green_noise2void_model_path.strip():
                raise BackendUnavailableError("noise2void model path not configured.")
            denoised = _denoise_noise2void_torch(arr, config.green_noise2void_model_path)
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "noise2void"
        except BackendUnavailableError as exc:
            logger.warning("%s Falling back to classical_branch_aware.", exc)
            return run_classical(arr), "classical_branch_aware(fallback)"

    if strategy == "bm4d":
        try:
            sigma = estimate_noise_sigma(arr)
            if config.green_chunked_processing:
                denoised = run_in_chunks(
                    arr,
                    lambda chunk: _denoise_bm4d_impl(chunk, sigma=sigma),
                    chunk_depth=config.green_chunk_depth,
                    overlap=config.green_chunk_overlap,
                )
            else:
                denoised = _denoise_bm4d_impl(arr, sigma=sigma)
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "bm4d"
        except BackendUnavailableError as exc:
            logger.warning("%s Falling back to classical_branch_aware.", exc)
            return run_classical(arr), "classical_branch_aware(fallback)"

    # hybrid_auto
    try:
        try:
            import torch  # type: ignore
        except ImportError:
            torch = None  # type: ignore

        if (
            torch is not None
            and bool(torch.cuda.is_available())
            and config.green_noise2void_model_path.strip()
        ):
            denoised = _denoise_noise2void_torch(arr, config.green_noise2void_model_path)
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "noise2void"

        manageable_voxels = 512 * 512 * 160
        if arr.size <= manageable_voxels:
            sigma = estimate_noise_sigma(arr)
            denoised = _denoise_bm4d_impl(arr, sigma=sigma)
            return np.clip(denoised, 0.0, 1.0).astype(np.float32, copy=False), "bm4d"
    except BackendUnavailableError as exc:
        logger.info("Hybrid backend unavailable: %s", exc)

    return run_classical(arr), "classical_branch_aware"
