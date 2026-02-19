from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class VoxelSpacing:
    x_um: float = 0.331
    y_um: float = 0.331
    z_um: float = 0.4

    @property
    def voxel_volume_um3(self) -> float:
        return self.x_um * self.y_um * self.z_um


DEFAULT_SPACING = VoxelSpacing()


@dataclass
class ChannelVolume:
    name: Literal["green", "red"]
    data: np.ndarray
    z_indices: list[int]
    spacing: VoxelSpacing = DEFAULT_SPACING

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError("ChannelVolume.data must be 3D in (z, y, x) order.")
        if self.data.shape[0] != len(self.z_indices):
            raise ValueError("z_indices length must match data z dimension.")


@dataclass
class DatasetVolume:
    green: ChannelVolume
    red: ChannelVolume
    shared_z_range: tuple[int, int]


@dataclass(frozen=True)
class PSFConfig:
    enabled: bool = True
    sigma_xy_um: float = 0.4
    sigma_z_um: float = 0.8
    iterations: int = 10
    use_measured_psf: bool = True
    measured_psf_path: str = ""
    regularization_lambda: float = 0.003


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool = True
    flatfield_sigma_xy: float = 32.0
    contrast_low_pct: float = 0.5
    contrast_high_pct: float = 99.8
    denoise_method: Literal["anisotropic", "bilateral", "non_local_means", "none"] = "anisotropic"
    denoise_strength: float = 0.012
    green_denoise_multiplier: float = 2.8
    preserve_branches: bool = True
    green_denoise_strategy: Literal[
        "hybrid_auto",
        "classical_branch_aware",
        "bm4d",
        "noise2void",
        "legacy_anisotropic",
    ] = "hybrid_auto"
    green_noise_model: Literal["auto", "poisson_gaussian", "gaussian"] = "auto"
    green_branch_protection: float = 0.65
    green_nlm_patch_size: int = 3
    green_nlm_patch_distance: int = 4
    green_nlm_h_factor: float = 0.9
    green_apply_vst: bool = True
    green_pre_deconv_strength: float = 0.85
    green_post_deconv_strength: float = 0.45
    green_speckle_min_voxels: int = 10
    green_speckle_attenuation: float = 0.12
    green_noise2void_model_path: str = ""
    green_chunked_processing: bool = True
    green_chunk_depth: int = 48
    green_chunk_overlap: int = 4
    resample_for_mesh: bool = True
    mesh_target_z_um: float = 0.331


@dataclass(frozen=True)
class RenderConfig:
    threshold_green: float = 0.15
    threshold_red: float = 0.15
    opacity_green: float = 0.35
    opacity_red: float = 0.35
    iso_green: float = 0.25
    iso_red: float = 0.25
    offset_x_um: float = 0.0
    offset_y_um: float = 0.0
    offset_z_um: float = 0.0
    show_green: bool = True
    show_red: bool = True
    show_iso_green: bool = False
    show_iso_red: bool = False


@dataclass
class MetricsResult:
    channel: str
    voxel_count: int
    volume_um3: float
    component_count: int
    largest_component_voxels: int
    overlap_voxel_count: int = 0
    overlap_volume_um3: float = 0.0


@dataclass
class MetricsComputation:
    channel_results: list[MetricsResult] = field(default_factory=list)
    overlap_voxel_count: int = 0
    overlap_volume_um3: float = 0.0
