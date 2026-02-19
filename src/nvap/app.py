from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path

from nvap.analysis.metrics import compute_metrics
from nvap.config.types import DEFAULT_SPACING, PSFConfig, PreprocessConfig, RenderConfig
from nvap.io.stack_loader import load_dataset
from nvap.pipeline import (
    apply_psf_to_dataset,
    default_green_threshold,
    default_threshold,
    fill_and_sync_dataset,
    preprocess_for_deconvolution,
)
from nvap.analysis.green_benchmark import run_green_denoise_benchmark

logger = logging.getLogger(__name__)


def configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)


def run_headless_smoke(input_path: str | Path) -> int:
    source = Path(input_path).resolve()
    logger.info("Headless smoke start: input=%s", source)
    dataset = load_dataset(source, spacing=DEFAULT_SPACING)
    dataset = fill_and_sync_dataset(dataset)
    preprocess_cfg = PreprocessConfig()
    dataset = preprocess_for_deconvolution(dataset, preprocess_cfg)
    # Keep smoke test runtime practical while still exercising PSF path.
    processed = apply_psf_to_dataset(dataset, PSFConfig(iterations=1), preprocess_config=preprocess_cfg)

    render = RenderConfig(
        threshold_green=default_green_threshold(processed.green.data),
        threshold_red=default_threshold(processed.red.data),
    )
    metrics = compute_metrics(processed, render)
    print(f"NVAP smoke run OK - source={source}")
    for item in metrics.channel_results:
        print(
            f"{item.channel}: voxels={item.voxel_count}, "
            f"volume_um3={item.volume_um3:.3f}, components={item.component_count}"
        )
    print(
        f"overlap: voxels={metrics.overlap_voxel_count}, "
        f"volume_um3={metrics.overlap_volume_um3:.3f}"
    )
    logger.info("Headless smoke complete.")
    return 0


def _apply_green_profile(config: PreprocessConfig, profile: str) -> PreprocessConfig:
    if profile == "low_snr":
        return replace(
            config,
            green_denoise_strategy="hybrid_auto",
            green_branch_protection=0.72,
            green_pre_deconv_strength=0.95,
            green_post_deconv_strength=0.55,
            green_nlm_h_factor=1.05,
        )
    if profile == "high_snr":
        return replace(
            config,
            green_denoise_strategy="classical_branch_aware",
            green_branch_protection=0.78,
            green_pre_deconv_strength=0.55,
            green_post_deconv_strength=0.20,
            green_nlm_h_factor=0.60,
        )
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="NVAP - NeuroVascular Analytics Program")
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="Run load/process/metrics pipeline without GUI.",
    )
    parser.add_argument(
        "--input",
        default="Input",
        help="Dataset root path (default: Input).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--benchmark-denoise",
        action="store_true",
        help="Run benchmark harness for green denoising strategies.",
    )
    parser.add_argument(
        "--output",
        default="green_denoise_benchmark_report.json",
        help="Output path for benchmark JSON report.",
    )
    parser.add_argument(
        "--green-denoise-strategy",
        choices=["hybrid_auto", "classical_branch_aware", "bm4d", "noise2void", "legacy_anisotropic"],
        default=None,
        help="Override green denoising strategy.",
    )
    parser.add_argument(
        "--green-denoise-profile",
        choices=["balanced", "low_snr", "high_snr"],
        default="balanced",
        help="Preset profile for green denoising defaults.",
    )
    args = parser.parse_args()
    configure_logging(args.debug)
    logger.info("NVAP startup (debug=%s)", args.debug)

    preprocess_cfg = _apply_green_profile(PreprocessConfig(), args.green_denoise_profile)
    if args.green_denoise_strategy:
        preprocess_cfg = replace(preprocess_cfg, green_denoise_strategy=args.green_denoise_strategy)

    if args.headless_smoke:
        return run_headless_smoke(args.input)
    if args.benchmark_denoise:
        report = run_green_denoise_benchmark(
            input_path=args.input,
            output_path=args.output,
            preprocess_config=preprocess_cfg,
        )
        print(f"Benchmark report written: {report}")
        return 0

    from PySide6.QtWidgets import QApplication

    from nvap.ui.main_window import MainWindow

    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
