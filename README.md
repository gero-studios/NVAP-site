# NVAP

NVAP (NeuroVascular Analytics Program) is a desktop 3D viewer and analyzer for microscopy image stacks containing:

- Green channel: microglia
- Red channel: vasculature

It supports missing-slice interpolation, PSF-based deconvolution, 3D volume + isosurface rendering, basic metrics, and export.

## Units and spacing

Default voxel spacing is interpreted in micrometers:

- `x = 0.331 um`
- `y = 0.331 um`
- `z = 0.4 um`

## Expected input layout

Default auto-detection expects either:

- `Input/Segmented/Green/*.png`
- `Input/Segmented/Red/*.png`

or a selected root containing `Segmented/Green` and `Segmented/Red`.

NVAP also supports a single user-selected folder of RGB PNG slices when:

- filenames include `_z###` (for example `sample_z030.png`), and
- slice pixels are red/green-only (black background allowed; blue or mixed red+green pixels are ignored).

Filenames should include z-index and channel marker, for example:

- `..._z030c1.png` (green)
- `..._z030c2.png` (red)

## Run from source

### PowerShell quick start (Windows)

```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional denoise backends:

```powershell
# BM4D backend
python -m pip install -e ".[denoise_bm4d]"

# Torch / model backend
python -m pip install -e ".[denoise_torch]"
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Launch GUI:

```powershell
nvap
```

Launch GUI with verbose debug logs:

```powershell
nvap --debug
```

Optional headless smoke test:

```powershell
nvap --headless-smoke --input Input
```

Run tests:

```powershell
python -m pytest
```

## How to load dataset in NVAP

1. Click `Load Dataset`.
2. Choose a root folder that contains either:
- `Segmented/Green` and `Segmented/Red`, or
- `Input/Segmented/Green` and `Input/Segmented/Red`.
  You can also choose a single folder of RGB `_z###.png` slices that use only red/green pixels.
3. If auto-detection fails, NVAP prompts you to select Green and Red folders manually.
4. Wait for loading dialogs to complete:
- channel stack load
- missing z-slice interpolation
- PSF processing
- initial render + metrics
  During this step, NVAP shows `Elapsed` and estimated `ETA`.
5. Use threshold/opacity/isosurface controls in the left panel.
6. Check debug output in the `Debug Log` panel.

### PSF responsiveness note

- Richardson-Lucy on full `1024x1024xZ` stacks is expensive.
- In the UI, PSF and dataset load now run in background threads with an elapsed-time and ETA loading dialog.
- For interactive work, start with `Iterations = 1-3`, then increase if needed for final-quality output.
- `Apply PSF` now supports cancellation from the loading dialog.
- If canceled, NVAP keeps the previous rendered result unchanged.

### Branch-preserving preprocessing and mesh clarity

NVAP now applies a preprocessing chain before deconvolution:

- Flat-field/background correction
- Per-slice contrast normalization
- Branch-preserving denoise (anisotropic default, stronger on green channel)

Green channel denoising now supports strategies:

- `hybrid_auto` (default fallback chain: Noise2Void model -> BM4D -> classical branch-aware)
- `classical_branch_aware`
- `bm4d`
- `noise2void` (TorchScript model path optional in UI)
- `legacy_anisotropic` (compatibility baseline)

In the `Green Denoising` panel you can tune branch protection, NLM parameters, noise model, speckle controls,
and pre/post deconvolution denoise strength. You can preview a selected z-index and apply settings to full volume.

To better clean isosurfaces from anisotropic microscopy stacks, NVAP also resamples Z to near-isotropic spacing for mesh rendering.
Metrics remain computed on the processed (non-resampled) dataset.

For visualization, NVAP additionally applies a display-only Z squeeze (2/3 scale) so depth is less visually disproportionate.

For green microglia, default threshold initialization is biased lower to reduce the risk of cutting faint branches.
Threshold suggestion now uses a branch-aware green mode by default.

### Green denoise benchmark

CLI benchmark report:

```powershell
nvap --benchmark-denoise --input Input --output green_denoise_report.json
```

Override strategy/profile:

```powershell
nvap --benchmark-denoise --input Input --output report.json --green-denoise-strategy classical_branch_aware --green-denoise-profile low_snr
```

Script alternative:

```powershell
python scripts/benchmark_green_denoise.py --input Input --output report.json --strategy hybrid_auto
```

### Green denoise cookbook

- Faint branches, noisy stack: `hybrid_auto`, branch protection `0.7-0.8`, pre `0.9-1.1`, post `0.45-0.65`.
- High-SNR stacks: `classical_branch_aware`, branch protection `0.75+`, pre `0.5-0.7`, post `0.15-0.3`.
- Speckle-heavy output: increase `Speckle min voxels` and lower `Speckle attenuation` to suppress grains while retaining filaments.

### Measured PSF support

If you place a measured PSF file in your dataset root (or `Input/`) with one of these names, NVAP will automatically use it:

- `psf.npy`
- `psf.npz` (first array or key `psf`)
- `psf.tif` / `psf.tiff`

If no measured PSF is found, NVAP falls back to the Gaussian PSF configured by sigma XY/Z.

### Auto-save cache

- Processed PSF volumes are cached automatically in `.nvap_cache/`.
- Re-loading the same dataset with the same PSF settings reuses cache when available.
- This avoids re-running full PSF deconvolution and speeds up re-open/re-render workflows.

## Windows packaging

Use:

```powershell
.\scripts\build_windows.ps1
```

This builds a one-folder executable in `dist/NVAP/`.

## Current v1 capabilities

- Auto-detect channels with manual directory override fallback.
- Missing z-slice interpolation.
- Anisotropic Gaussian PSF + Richardson-Lucy deconvolution.
- 3D rendering per channel with independent controls.
- Basic metrics:
  - voxel count
  - physical volume
  - connected component count
  - largest component size
  - red-green overlap in shared z-range
- Export:
  - metrics CSV
  - snapshot PNG

## AI extension scaffold

NVAP exposes plugin discovery through the `nvap.plugins` entry point group and a `ChannelAnalyzerPlugin` protocol. v1 includes extension points only.
