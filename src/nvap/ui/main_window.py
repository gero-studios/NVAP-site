from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
)

from nvap.analysis.metrics import compute_metrics, metrics_to_csv_rows
from nvap.cache.processed_cache import (
    build_dataset_signature,
    build_processed_cache_key,
    load_processed_dataset,
    save_processed_dataset,
)
from nvap.config.types import (
    DEFAULT_SPACING,
    DatasetVolume,
    MetricsComputation,
    PSFConfig,
    PreprocessConfig,
    RenderConfig,
    VoxelSpacing,
)
from nvap.export.exporters import export_metrics_csv
from nvap.io.stack_loader import inspect_dataset_stats, load_dataset, resolve_channel_dirs
from nvap.pipeline import (
    apply_psf_to_dataset,
    default_threshold,
    fill_and_sync_dataset,
    prepare_dataset_for_mesh,
    preprocess_for_deconvolution,
)
from nvap.plugins.registry import discover_plugins
from nvap.render.vtk_scene import VTKScene
from nvap.ui.control_panel import ControlPanel

logger = logging.getLogger(__name__)


@dataclass
class _LoadTaskResult:
    raw_dataset: DatasetVolume
    processed_dataset: DatasetVolume
    visual_dataset: DatasetVolume
    threshold_green: float
    threshold_red: float


class _LogBridge(QObject):
    message = Signal(str)


class _ControlPanelLogHandler(logging.Handler):
    def __init__(self, emit_message: Callable[[str], None]) -> None:
        super().__init__()
        self._emit_message = emit_message

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self._emit_message(message)
        except Exception:
            # Never let logging errors break UI flow.
            pass


class _FunctionThread(QThread):
    result_ready = Signal(object)
    error_raised = Signal(str)

    def __init__(self, fn: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            self.result_ready.emit(result)
        except Exception:
            self.error_raised.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NVAP - NeuroVascular Analytics Program")
        self.resize(1540, 920)

        self.scene = VTKScene(self)
        self.controls = ControlPanel(self)
        splitter = QSplitter(self)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.scene.widget())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.spacing = DEFAULT_SPACING
        self.preprocess_config = PreprocessConfig()
        self.raw_dataset: DatasetVolume | None = None
        self.processed_dataset: DatasetVolume | None = None
        self.visual_dataset: DatasetVolume | None = None
        self.current_psf = self.controls.current_psf_config()
        self.current_render = self.controls.current_render_config()
        self.latest_metrics: MetricsComputation | None = None
        self.dataset_root: Path | None = None
        self._dataset_signature: str | None = None
        self._busy_dialog: QProgressDialog | None = None
        self._busy_start = 0.0
        self._busy_base_message = ""
        self._busy_title = ""
        self._busy_eta_total: float | None = None
        self._eta_scale_load = 1.0
        self._eta_scale_psf = 1.0
        self._display_z_scale = 2.0 / 3.0
        self._busy_timer = QTimer(self)
        self._busy_timer.setInterval(1000)
        self._busy_timer.timeout.connect(self._on_busy_tick)
        self._active_thread: _FunctionThread | None = None

        self._log_bridge = _LogBridge(self)
        self._log_bridge.message.connect(self.controls.append_debug_text)
        self._log_handler = _ControlPanelLogHandler(self._log_bridge.message.emit)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger("nvap").addHandler(self._log_handler)

        self.controls.load_requested.connect(self._on_load_requested)
        self.controls.apply_psf_requested.connect(self._on_apply_psf_requested)
        self.controls.psf_config_changed.connect(self._on_psf_config_changed)
        self.controls.render_config_changed.connect(self._on_render_config_changed)
        self.controls.export_metrics_requested.connect(self._on_export_metrics_requested)
        self.controls.export_snapshot_requested.connect(self._on_export_snapshot_requested)

        self._refresh_plugin_panel()
        self.statusBar().showMessage("Load a dataset to begin.")
        self._log_info("NVAP UI initialized.")

    def closeEvent(self, event) -> None:
        if self._active_thread is not None and self._active_thread.isRunning():
            self._log_info("Waiting briefly for active background task to finish before close.")
            self._active_thread.wait(2000)
        logging.getLogger("nvap").removeHandler(self._log_handler)
        super().closeEvent(event)

    def _log_info(self, message: str) -> None:
        logger.info(message)

    def _log_debug(self, message: str) -> None:
        logger.debug(message)

    def _display_spacing(self, spacing: VoxelSpacing) -> VoxelSpacing:
        # Visual-only Z squeeze for less depth exaggeration. Metrics stay in physical units.
        return VoxelSpacing(
            x_um=spacing.x_um,
            y_um=spacing.y_um,
            z_um=float(spacing.z_um) * self._display_z_scale,
        )

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        mins, secs = divmod(total, 60)
        return f"{mins:02d}:{secs:02d}"

    def _compose_busy_label(self, elapsed: float) -> str:
        lines = [self._busy_base_message, f"Elapsed: {self._format_seconds(elapsed)}"]
        if self._busy_eta_total is not None:
            remaining = max(0.0, self._busy_eta_total - elapsed)
            lines.append(f"ETA: {self._format_seconds(remaining)}")
        return "\n".join(lines)

    def _estimate_load_eta_seconds(
        self,
        root: Path,
        channel_overrides: dict[str, str | Path] | None,
        psf_cfg: PSFConfig,
    ) -> float | None:
        try:
            stats = inspect_dataset_stats(root, channel_overrides=channel_overrides)
        except Exception as exc:
            self._log_debug(f"ETA estimation unavailable: {exc}")
            return None

        # Heuristic model tuned from local benchmark runs on 1024x1024 microscopy stacks.
        load_seconds = 0.022 * (stats.green.slice_count + stats.red.slice_count)
        interpolate_seconds = 0.040 * stats.total_missing_slices
        preprocess_seconds = stats.total_full_voxels * 2.0e-8
        psf_seconds = 0.0
        if psf_cfg.enabled and psf_cfg.iterations > 0:
            psf_seconds = stats.total_full_voxels * psf_cfg.iterations * 8.0e-8
        threshold_seconds = stats.total_full_voxels * 1.2e-8
        resample_seconds = (
            stats.total_full_voxels * 7.0e-9 if self.preprocess_config.resample_for_mesh else 0.0
        )
        render_seconds = 4.0
        total = (
            load_seconds
            + interpolate_seconds
            + preprocess_seconds
            + psf_seconds
            + threshold_seconds
            + resample_seconds
            + render_seconds
        )
        total *= self._eta_scale_load
        total = max(5.0, total)
        self._log_info(
            "Estimated load ETA="
            f"{total:.1f}s (scale={self._eta_scale_load:.2f}, "
            f"slices={stats.green.slice_count}/{stats.red.slice_count}, "
            f"iterations={psf_cfg.iterations})"
        )
        return total

    def _estimate_psf_eta_seconds(
        self,
        dataset: DatasetVolume,
        psf_cfg: PSFConfig,
    ) -> float | None:
        if not psf_cfg.enabled or psf_cfg.iterations <= 0:
            return 2.0
        total_voxels = int(dataset.green.data.size + dataset.red.data.size)
        preprocess_seconds = total_voxels * 2.0e-8
        psf_seconds = total_voxels * psf_cfg.iterations * 8.0e-8
        threshold_seconds = total_voxels * 1.2e-8
        resample_seconds = total_voxels * 7.0e-9 if self.preprocess_config.resample_for_mesh else 0.0
        total = (preprocess_seconds + psf_seconds + threshold_seconds + resample_seconds + 2.0) * self._eta_scale_psf
        total = max(3.0, total)
        self._log_info(
            "Estimated PSF ETA="
            f"{total:.1f}s (scale={self._eta_scale_psf:.2f}, "
            f"iterations={psf_cfg.iterations})"
        )
        return total

    def _detect_measured_psf_path(self) -> str:
        if self.dataset_root is None:
            return ""
        candidates = [
            self.dataset_root / "psf.npy",
            self.dataset_root / "psf.npz",
            self.dataset_root / "psf.tif",
            self.dataset_root / "psf.tiff",
            self.dataset_root / "Input" / "psf.npy",
            self.dataset_root / "Input" / "psf.npz",
            self.dataset_root / "Input" / "psf.tif",
            self.dataset_root / "Input" / "psf.tiff",
        ]
        for path in candidates:
            if path.exists():
                self._log_info(f"Detected measured PSF file: {path}")
                return str(path.resolve())
        return ""

    def _effective_psf_config(self, config: PSFConfig) -> PSFConfig:
        if config.measured_psf_path.strip():
            return config
        detected = self._detect_measured_psf_path()
        if detected:
            return replace(config, measured_psf_path=detected, use_measured_psf=True)
        return config

    def _on_busy_tick(self) -> None:
        if self._busy_dialog is None:
            return
        elapsed = time.perf_counter() - self._busy_start
        self._busy_dialog.setLabelText(self._compose_busy_label(elapsed))

    def _show_busy(
        self,
        title: str,
        message: str,
        eta_total_seconds: float | None = None,
        allow_cancel: bool = False,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        if self._busy_dialog is None:
            self._busy_dialog = QProgressDialog(message, "", 0, 0, self)
            self._busy_dialog.setMinimumDuration(0)
            self._busy_dialog.setWindowModality(Qt.WindowModal)
        try:
            self._busy_dialog.canceled.disconnect()
        except Exception:
            pass
        if allow_cancel:
            self._busy_dialog.setCancelButtonText("Cancel")
            if on_cancel is not None:
                self._busy_dialog.canceled.connect(on_cancel)
        else:
            self._busy_dialog.setCancelButton(None)
        self._busy_title = title
        self._busy_base_message = message
        self._busy_eta_total = eta_total_seconds
        self._busy_start = time.perf_counter()
        self._busy_dialog.setWindowTitle(title)
        self._busy_dialog.setLabelText(self._compose_busy_label(0.0))
        self._busy_dialog.show()
        self._busy_timer.start()
        QApplication.processEvents()
        self._log_info(f"{title} - started")

    def _hide_busy(self) -> None:
        elapsed = time.perf_counter() - self._busy_start
        finished_title = self._busy_title
        self._busy_timer.stop()
        if self._busy_dialog is not None:
            self._busy_dialog.close()
            self._busy_dialog = None
        self._busy_eta_total = None
        self._busy_title = ""
        self._busy_base_message = ""
        QApplication.processEvents()
        if finished_title:
            self._log_info(f"{finished_title} - finished in {elapsed:.2f}s")

    @contextmanager
    def _busy(self, title: str, message: str):
        self._show_busy(title, message)
        try:
            yield
        finally:
            self._hide_busy()

    def _set_busy_message(self, message: str) -> None:
        self._busy_base_message = message
        if self._busy_dialog is not None:
            elapsed = time.perf_counter() - self._busy_start
            self._busy_dialog.setLabelText(self._compose_busy_label(elapsed))
            QApplication.processEvents()
        self._log_debug(message)

    def _start_background_task(
        self,
        title: str,
        message: str,
        fn: Callable[[], object],
        on_success: Callable[[object], None],
        error_title: str,
        success_status: str | None = None,
        eta_total_seconds: float | None = None,
        eta_kind: str | None = None,
        allow_cancel: bool = False,
        cancel_event: threading.Event | None = None,
        canceled_status: str | None = None,
    ) -> None:
        if self._active_thread is not None and self._active_thread.isRunning():
            QMessageBox.warning(self, "Task Running", "Another operation is still running.")
            return

        if allow_cancel and cancel_event is None:
            cancel_event = threading.Event()

        def on_cancel_request() -> None:
            if cancel_event is not None:
                cancel_event.set()
            self._set_busy_message("Cancel requested. Finishing current iteration...")
            self._log_info("Cancellation requested by user.")

        self.controls.setEnabled(False)
        self._show_busy(
            title,
            message,
            eta_total_seconds=eta_total_seconds,
            allow_cancel=allow_cancel,
            on_cancel=on_cancel_request if allow_cancel else None,
        )
        thread = _FunctionThread(fn, self)
        self._active_thread = thread

        def cleanup() -> None:
            elapsed = time.perf_counter() - self._busy_start
            if eta_total_seconds is not None and eta_total_seconds > 0:
                ratio = elapsed / eta_total_seconds
                if eta_kind == "load":
                    scale = self._eta_scale_load
                    scale = max(0.5, min(2.5, (0.8 * scale) + (0.2 * ratio)))
                    self._eta_scale_load = scale
                    self._log_debug(
                        f"Load ETA calibration updated: ratio={ratio:.2f}, scale={self._eta_scale_load:.2f}"
                    )
                elif eta_kind == "psf":
                    scale = self._eta_scale_psf
                    scale = max(0.5, min(2.5, (0.8 * scale) + (0.2 * ratio)))
                    self._eta_scale_psf = scale
                    self._log_debug(
                        f"PSF ETA calibration updated: ratio={ratio:.2f}, scale={self._eta_scale_psf:.2f}"
                    )
            self.controls.setEnabled(True)
            self._hide_busy()
            if self._active_thread is thread:
                self._active_thread = None
            thread.deleteLater()

        def handle_success(result: object) -> None:
            try:
                on_success(result)
                if success_status:
                    self.statusBar().showMessage(success_status, 5000)
            except Exception as exc:
                self._show_error(error_title, str(exc))
            finally:
                cleanup()

        def handle_error(error_text: str) -> None:
            logger.error("Background task error:\n%s", error_text)
            if "OperationCanceledError" in error_text:
                if canceled_status:
                    self.statusBar().showMessage(canceled_status, 5000)
                self._log_info("Background task canceled.")
                cleanup()
                return
            concise = error_text.strip().splitlines()[-1] if error_text.strip() else "Unknown error"
            self._show_error(error_title, concise)
            cleanup()

        thread.result_ready.connect(handle_success)
        thread.error_raised.connect(handle_error)
        thread.start()

    def _refresh_plugin_panel(self) -> None:
        self._log_debug("Discovering plugins from entry point group 'nvap.plugins'.")
        plugins = discover_plugins()
        if not plugins:
            self.controls.set_plugin_text("No plugins discovered in 'nvap.plugins'.")
            self._log_info("No plugins discovered.")
            return
        lines = []
        for plugin in plugins:
            if plugin.status == "loaded":
                lines.append(f"- {plugin.plugin_id} ({plugin.target_channel}) loaded")
            else:
                lines.append(f"- {plugin.plugin_id} error: {plugin.error}")
        self._log_info(f"Discovered {len(plugins)} plugin descriptor(s).")
        self.controls.set_plugin_text("\n".join(lines))

    def _on_load_requested(self) -> None:
        base = self.dataset_root or (Path.cwd() / "Input")
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Root",
            str(base),
        )
        if not selected:
            return
        self.dataset_root = Path(selected)
        self._log_info(f"Dataset root selected: {self.dataset_root}")

        channel_overrides: dict[str, str | Path] | None = None
        channel_dirs: dict[str, Path]
        try:
            channel_dirs = resolve_channel_dirs(self.dataset_root)
        except FileNotFoundError:
            self._log_info("Auto-detection failed; requesting manual channel mapping.")
            QMessageBox.information(
                self,
                "Manual Channel Mapping",
                (
                    "Auto-detection failed.\n\n"
                    "Select Green channel folder, then Red channel folder."
                ),
            )
            green_dir = QFileDialog.getExistingDirectory(self, "Select Green Channel Folder", str(self.dataset_root))
            if not green_dir:
                return
            red_dir = QFileDialog.getExistingDirectory(self, "Select Red Channel Folder", str(self.dataset_root))
            if not red_dir:
                return
            channel_overrides = {"green": green_dir, "red": red_dir}
            channel_dirs = resolve_channel_dirs(self.dataset_root, channel_overrides=channel_overrides)

        self._dataset_signature = build_dataset_signature(channel_dirs)
        self._log_debug(f"Dataset signature set: {self._dataset_signature}")

        root = self.dataset_root
        psf_cfg = self._effective_psf_config(self.current_psf)
        self.current_psf = psf_cfg
        eta_seconds = self._estimate_load_eta_seconds(root, channel_overrides, psf_cfg)
        self._start_background_task(
            title="Load Dataset",
            message="Loading stacks, preprocessing, and running PSF...",
            fn=lambda: self._background_load_dataset(root, channel_overrides, psf_cfg),
            on_success=self._on_load_task_success,
            error_title="Dataset load failed",
            success_status=f"Loaded dataset: {self.dataset_root}",
            eta_total_seconds=eta_seconds,
            eta_kind="load",
        )

    def _background_load_dataset(
        self,
        root: Path,
        channel_overrides: dict[str, str | Path] | None,
        psf_cfg: PSFConfig,
    ) -> _LoadTaskResult:
        dataset = load_dataset(root, spacing=self.spacing, channel_overrides=channel_overrides)
        raw_dataset = fill_and_sync_dataset(dataset)
        processed_dataset = self._get_processed_dataset_with_cache(
            raw_dataset,
            psf_cfg,
            self.preprocess_config,
            self._dataset_signature,
            cancel_event=None,
        )
        visual_dataset = prepare_dataset_for_mesh(processed_dataset, self.preprocess_config)
        threshold_green = default_threshold(processed_dataset.green.data)
        threshold_red = default_threshold(processed_dataset.red.data)
        if self.preprocess_config.preserve_branches:
            threshold_green = max(0.02, threshold_green * 0.88)
        return _LoadTaskResult(
            raw_dataset=raw_dataset,
            processed_dataset=processed_dataset,
            visual_dataset=visual_dataset,
            threshold_green=threshold_green,
            threshold_red=threshold_red,
        )

    def _on_load_task_success(self, result: object) -> None:
        if not isinstance(result, _LoadTaskResult):
            raise TypeError("Invalid load task result payload.")
        self.raw_dataset = result.raw_dataset
        self.processed_dataset = result.processed_dataset
        self.visual_dataset = result.visual_dataset
        self._set_busy_message("Uploading green channel to VTK...")
        self.scene.set_channel_data(
            channel="green",
            volume=self.visual_dataset.green.data,
            spacing=self._display_spacing(self.visual_dataset.green.spacing),
        )
        self._set_busy_message("Uploading red channel to VTK...")
        self.scene.set_channel_data(
            channel="red",
            volume=self.visual_dataset.red.data,
            spacing=self._display_spacing(self.visual_dataset.red.spacing),
        )
        self._set_busy_message("Applying initial thresholds and computing metrics...")
        self.controls.set_threshold_defaults(result.threshold_green, result.threshold_red)
        self.scene.apply_render_config(self.current_render)
        self._refresh_metrics()
        self._log_info("Dataset load and initial render completed.")

    def _on_psf_task_success(self, result: object) -> None:
        if not isinstance(result, DatasetVolume):
            raise TypeError("Invalid PSF task result payload.")
        self.processed_dataset = result
        self.visual_dataset = prepare_dataset_for_mesh(self.processed_dataset, self.preprocess_config)
        self._set_busy_message("Uploading green channel to VTK...")
        self.scene.set_channel_data(
            channel="green",
            volume=self.visual_dataset.green.data,
            spacing=self._display_spacing(self.visual_dataset.green.spacing),
        )
        self._set_busy_message("Uploading red channel to VTK...")
        self.scene.set_channel_data(
            channel="red",
            volume=self.visual_dataset.red.data,
            spacing=self._display_spacing(self.visual_dataset.red.spacing),
        )
        self._set_busy_message("Refreshing render + metrics...")
        self.scene.apply_render_config(self.current_render)
        self._refresh_metrics()
        self._log_info("PSF processing applied and scene refreshed.")

    def _get_processed_dataset_with_cache(
        self,
        raw_dataset: DatasetVolume,
        psf_cfg: PSFConfig,
        preprocess_cfg: PreprocessConfig,
        dataset_signature: str | None,
        cancel_event: threading.Event | None,
    ) -> DatasetVolume:
        preprocessed_dataset = preprocess_for_deconvolution(raw_dataset, preprocess_cfg)
        cache_key = None
        if dataset_signature:
            cache_key = build_processed_cache_key(
                dataset_signature,
                self.spacing,
                psf_cfg,
                preprocess_config=preprocess_cfg,
            )
            cached = load_processed_dataset(cache_key, self.spacing)
            if cached is not None:
                self._log_info("Using cached processed dataset.")
                return cached

        processed = apply_psf_to_dataset(
            preprocessed_dataset,
            psf_cfg,
            cancel_event=cancel_event,
        )
        if cache_key is not None and (cancel_event is None or not cancel_event.is_set()):
            save_processed_dataset(cache_key, processed)
        return processed

    def _load_dataset_with_manual_fallback(self, root: Path):
        # Kept for compatibility with older code paths.
        try:
            self._log_debug("Attempting dataset auto-detection.")
            return load_dataset(root, spacing=self.spacing)
        except FileNotFoundError:
            self._log_info("Auto-detection failed; requesting manual channel mapping.")
            QMessageBox.information(
                self,
                "Manual Channel Mapping",
                (
                    "Auto-detection failed.\n\n"
                    "Select Green channel folder, then Red channel folder."
                ),
            )
            green_dir = QFileDialog.getExistingDirectory(self, "Select Green Channel Folder", str(root))
            if not green_dir:
                raise RuntimeError("Green channel directory selection canceled.")
            red_dir = QFileDialog.getExistingDirectory(self, "Select Red Channel Folder", str(root))
            if not red_dir:
                raise RuntimeError("Red channel directory selection canceled.")
            return load_dataset(
                root,
                spacing=self.spacing,
                channel_overrides={"green": green_dir, "red": red_dir},
            )

    def _on_psf_config_changed(self, config: PSFConfig) -> None:
        self.current_psf = config
        if config.iterations >= 8:
            self.statusBar().showMessage(
                "High RL iterations can take several minutes on large stacks.",
                4000,
            )

    def _on_apply_psf_requested(self) -> None:
        if self.raw_dataset is None:
            self._show_error("No dataset", "Load a dataset before applying PSF.")
            return
        raw_dataset = self.raw_dataset
        psf_cfg = self._effective_psf_config(self.current_psf)
        self.current_psf = psf_cfg
        dataset_signature = self._dataset_signature
        cancel_event = threading.Event()
        eta_seconds = self._estimate_psf_eta_seconds(raw_dataset, psf_cfg)
        self._start_background_task(
            title="Apply PSF",
            message=(
                f"Running Richardson-Lucy (iterations={psf_cfg.iterations})...\n"
                "This can take several minutes for large volumes."
            ),
            fn=lambda: self._get_processed_dataset_with_cache(
                raw_dataset,
                psf_cfg,
                self.preprocess_config,
                dataset_signature,
                cancel_event=cancel_event,
            ),
            on_success=self._on_psf_task_success,
            error_title="PSF processing failed",
            success_status="PSF processing complete.",
            eta_total_seconds=eta_seconds,
            eta_kind="psf",
            allow_cancel=True,
            cancel_event=cancel_event,
            canceled_status="PSF canceled. Previous rendering kept.",
        )

    def _apply_psf_and_refresh(self, update_thresholds: bool) -> None:
        # Kept for compatibility with older code paths.
        assert self.raw_dataset is not None
        self._set_busy_message("Applying PSF deconvolution to channels...")
        psf_cfg = self._effective_psf_config(self.current_psf)
        preprocessed = preprocess_for_deconvolution(self.raw_dataset, self.preprocess_config)
        self.processed_dataset = apply_psf_to_dataset(preprocessed, psf_cfg)
        self.visual_dataset = prepare_dataset_for_mesh(self.processed_dataset, self.preprocess_config)
        self._log_debug(
            f"Processed dataset shapes - green={self.processed_dataset.green.data.shape}, "
            f"red={self.processed_dataset.red.data.shape}"
        )

        self._set_busy_message("Uploading green channel to VTK...")
        self.scene.set_channel_data(
            channel="green",
            volume=self.visual_dataset.green.data,
            spacing=self._display_spacing(self.visual_dataset.green.spacing),
        )
        self._set_busy_message("Uploading red channel to VTK...")
        self.scene.set_channel_data(
            channel="red",
            volume=self.visual_dataset.red.data,
            spacing=self._display_spacing(self.visual_dataset.red.spacing),
        )

        if update_thresholds:
            self._set_busy_message("Computing initial Otsu thresholds...")
            tg = default_threshold(self.processed_dataset.green.data)
            tr = default_threshold(self.processed_dataset.red.data)
            self.controls.set_threshold_defaults(tg, tr)
            self._log_info(f"Default thresholds set: green={tg:.4f}, red={tr:.4f}")

        self._set_busy_message("Refreshing render + metrics...")
        self.scene.apply_render_config(self.current_render)
        self._refresh_metrics()

    def _on_render_config_changed(self, config: RenderConfig) -> None:
        self.current_render = config
        if self.processed_dataset is None:
            return
        self.scene.apply_render_config(config)
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        if self.processed_dataset is None:
            return
        self.latest_metrics = compute_metrics(self.processed_dataset, self.current_render)
        lines = []
        for item in self.latest_metrics.channel_results:
            lines.append(
                (
                    f"{item.channel}: voxels={item.voxel_count}, "
                    f"volume_um3={item.volume_um3:.3f}, "
                    f"components={item.component_count}, "
                    f"largest_component={item.largest_component_voxels}"
                )
            )
        lines.append(
            f"overlap: voxels={self.latest_metrics.overlap_voxel_count}, "
            f"volume_um3={self.latest_metrics.overlap_volume_um3:.3f}"
        )
        self.controls.set_metrics_text("\n".join(lines))
        self._log_debug("Metrics updated.")

    def _on_export_metrics_requested(self) -> None:
        if self.latest_metrics is None:
            self._show_error("No metrics", "Compute metrics before exporting.")
            return
        start = str((self.dataset_root or Path.cwd()) / "metrics.csv")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Metrics CSV",
            start,
            "CSV files (*.csv)",
        )
        if not file_path:
            return
        try:
            with self._busy("Export Metrics", "Writing CSV..."):
                rows = metrics_to_csv_rows(self.latest_metrics)
                out = export_metrics_csv(rows, file_path)
            self.statusBar().showMessage(f"Metrics exported to {out}", 5000)
            self._log_info(f"Metrics exported: {out}")
        except Exception as exc:
            self._log_info(f"Metrics export failed: {exc}")
            self._show_error("Export failed", str(exc))

    def _on_export_snapshot_requested(self) -> None:
        start = str((self.dataset_root or Path.cwd()) / "snapshot.png")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Snapshot PNG",
            start,
            "PNG files (*.png)",
        )
        if not file_path:
            return
        try:
            with self._busy("Export Snapshot", "Rendering and writing PNG..."):
                out = self.scene.capture_snapshot(file_path)
            self.statusBar().showMessage(f"Snapshot exported to {out}", 5000)
            self._log_info(f"Snapshot exported: {out}")
        except Exception as exc:
            self._log_info(f"Snapshot export failed: {exc}")
            self._show_error("Snapshot export failed", str(exc))

    def _show_error(self, title: str, details: str) -> None:
        logger.error("%s: %s", title, details)
        QMessageBox.critical(self, title, details)
