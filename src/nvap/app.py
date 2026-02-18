from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nvap.analysis.metrics import compute_metrics
from nvap.config.types import DEFAULT_SPACING, PSFConfig, PreprocessConfig, RenderConfig
from nvap.io.stack_loader import load_dataset
from nvap.pipeline import (
    apply_psf_to_dataset,
    default_threshold,
    fill_and_sync_dataset,
    preprocess_for_deconvolution,
)

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
    dataset = preprocess_for_deconvolution(dataset, PreprocessConfig())
    # Keep smoke test runtime practical while still exercising PSF path.
    processed = apply_psf_to_dataset(dataset, PSFConfig(iterations=1))

    render = RenderConfig(
        threshold_green=default_threshold(processed.green.data),
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
    args = parser.parse_args()
    configure_logging(args.debug)
    logger.info("NVAP startup (debug=%s)", args.debug)

    if args.headless_smoke:
        return run_headless_smoke(args.input)

    from PySide6.QtWidgets import QApplication

    from nvap.ui.main_window import MainWindow

    app = QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
