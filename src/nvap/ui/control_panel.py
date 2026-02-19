from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

from nvap.config.types import PSFConfig, PreprocessConfig, RenderConfig


class ControlPanel(QWidget):
    load_requested = Signal()
    apply_psf_requested = Signal()
    render_config_changed = Signal(object)
    psf_config_changed = Signal(object)
    preprocess_config_changed = Signal(object)
    preview_green_denoise_requested = Signal()
    apply_green_denoise_requested = Signal()
    export_metrics_requested = Signal()
    export_snapshot_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        load_btn = QPushButton("Load Dataset")
        load_btn.clicked.connect(self.load_requested.emit)
        root.addWidget(load_btn)

        self.render_group = self._build_render_group()
        root.addWidget(self.render_group)

        self.psf_group = self._build_psf_group()
        root.addWidget(self.psf_group)

        self.preprocess_group = self._build_preprocess_group()
        root.addWidget(self.preprocess_group)

        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)
        csv_btn = QPushButton("Metrics CSV")
        csv_btn.clicked.connect(self.export_metrics_requested.emit)
        png_btn = QPushButton("Snapshot PNG")
        png_btn.clicked.connect(self.export_snapshot_requested.emit)
        export_layout.addWidget(csv_btn)
        export_layout.addWidget(png_btn)
        root.addWidget(export_group)

        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumHeight(170)
        metrics_layout.addWidget(self.metrics_text)
        root.addWidget(metrics_group)

        plugin_group = QGroupBox("AI Plugins (Reserved)")
        plugin_layout = QVBoxLayout(plugin_group)
        self.plugin_text = QTextEdit()
        self.plugin_text.setReadOnly(True)
        self.plugin_text.setMinimumHeight(90)
        plugin_layout.addWidget(self.plugin_text)
        root.addWidget(plugin_group)

        debug_group = QGroupBox("Debug Log")
        debug_layout = QVBoxLayout(debug_group)
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setMinimumHeight(140)
        debug_layout.addWidget(self.debug_text)
        root.addWidget(debug_group)

        root.addStretch(1)
        self._emit_render_config()
        self._emit_psf_config()
        self._emit_preprocess_config()

    def _build_render_group(self) -> QGroupBox:
        group = QGroupBox("Rendering")
        form = QFormLayout(group)

        self.show_green = QCheckBox("Show green")
        self.show_green.setChecked(True)
        self.show_green.stateChanged.connect(self._emit_render_config)
        form.addRow(self.show_green)

        self.show_red = QCheckBox("Show red")
        self.show_red.setChecked(True)
        self.show_red.stateChanged.connect(self._emit_render_config)
        form.addRow(self.show_red)

        self.show_iso_green = QCheckBox("Show green isosurface")
        self.show_iso_green.stateChanged.connect(self._emit_render_config)
        form.addRow(self.show_iso_green)

        self.show_iso_red = QCheckBox("Show red isosurface")
        self.show_iso_red.stateChanged.connect(self._emit_render_config)
        form.addRow(self.show_iso_red)

        self.threshold_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_green.valueChanged.connect(self._emit_render_config)
        form.addRow("Threshold green", self.threshold_green)

        self.threshold_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.15)
        self.threshold_red.valueChanged.connect(self._emit_render_config)
        form.addRow("Threshold red", self.threshold_red)

        self.opacity_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.35)
        self.opacity_green.valueChanged.connect(self._emit_render_config)
        form.addRow("Opacity green", self.opacity_green)

        self.opacity_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.35)
        self.opacity_red.valueChanged.connect(self._emit_render_config)
        form.addRow("Opacity red", self.opacity_red)

        self.iso_green = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.25)
        self.iso_green.valueChanged.connect(self._emit_render_config)
        form.addRow("Iso green", self.iso_green)

        self.iso_red = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.25)
        self.iso_red.valueChanged.connect(self._emit_render_config)
        form.addRow("Iso red", self.iso_red)

        self.offset_x = self._make_offset_spinbox()
        self.offset_x.valueChanged.connect(self._emit_render_config)
        form.addRow("Green offset X (um)", self.offset_x)

        self.offset_y = self._make_offset_spinbox()
        self.offset_y.valueChanged.connect(self._emit_render_config)
        form.addRow("Green offset Y (um)", self.offset_y)

        self.offset_z = self._make_offset_spinbox()
        self.offset_z.valueChanged.connect(self._emit_render_config)
        form.addRow("Green offset Z (um)", self.offset_z)
        return group

    def _build_psf_group(self) -> QGroupBox:
        group = QGroupBox("PSF Deconvolution")
        form = QFormLayout(group)

        self.psf_enabled = QCheckBox("Enabled")
        self.psf_enabled.setChecked(True)
        self.psf_enabled.stateChanged.connect(self._emit_psf_config)
        form.addRow(self.psf_enabled)

        self.sigma_xy = self._make_unit_spinbox(0.05, 5.0, 0.05, 0.4)
        self.sigma_xy.valueChanged.connect(self._emit_psf_config)
        form.addRow("Sigma XY (um)", self.sigma_xy)

        self.sigma_z = self._make_unit_spinbox(0.05, 8.0, 0.05, 0.8)
        self.sigma_z.valueChanged.connect(self._emit_psf_config)
        form.addRow("Sigma Z (um)", self.sigma_z)

        self.iterations = QSpinBox()
        self.iterations.setRange(0, 100)
        self.iterations.setValue(10)
        self.iterations.valueChanged.connect(self._emit_psf_config)
        form.addRow("Iterations", self.iterations)

        apply_btn = QPushButton("Apply PSF")
        apply_btn.clicked.connect(self.apply_psf_requested.emit)
        form.addRow(apply_btn)
        return group

    def _build_preprocess_group(self) -> QGroupBox:
        group = QGroupBox("Green Denoising")
        form = QFormLayout(group)

        self.green_strategy = QComboBox()
        self.green_strategy.addItems(
            [
                "hybrid_auto",
                "classical_branch_aware",
                "bm4d",
                "noise2void",
                "legacy_anisotropic",
            ]
        )
        self.green_strategy.currentTextChanged.connect(self._emit_preprocess_config)
        form.addRow("Strategy", self.green_strategy)

        self.green_noise_model = QComboBox()
        self.green_noise_model.addItems(["auto", "poisson_gaussian", "gaussian"])
        self.green_noise_model.currentTextChanged.connect(self._emit_preprocess_config)
        form.addRow("Noise model", self.green_noise_model)

        self.green_branch_protection = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.65)
        self.green_branch_protection.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("Branch protection", self.green_branch_protection)

        self.green_nlm_patch_size = QSpinBox()
        self.green_nlm_patch_size.setRange(1, 9)
        self.green_nlm_patch_size.setSingleStep(1)
        self.green_nlm_patch_size.setValue(3)
        self.green_nlm_patch_size.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("NLM patch size", self.green_nlm_patch_size)

        self.green_nlm_patch_distance = QSpinBox()
        self.green_nlm_patch_distance.setRange(1, 12)
        self.green_nlm_patch_distance.setValue(4)
        self.green_nlm_patch_distance.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("NLM patch dist", self.green_nlm_patch_distance)

        self.green_nlm_h_factor = self._make_unit_spinbox(0.1, 3.0, 0.05, 0.9)
        self.green_nlm_h_factor.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("NLM h factor", self.green_nlm_h_factor)

        self.green_apply_vst = QCheckBox("Apply VST (Anscombe)")
        self.green_apply_vst.setChecked(True)
        self.green_apply_vst.stateChanged.connect(self._emit_preprocess_config)
        form.addRow(self.green_apply_vst)

        self.green_pre_deconv_strength = self._make_unit_spinbox(0.0, 2.0, 0.05, 0.85)
        self.green_pre_deconv_strength.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("Pre-deconv strength", self.green_pre_deconv_strength)

        self.green_post_deconv_strength = self._make_unit_spinbox(0.0, 2.0, 0.05, 0.45)
        self.green_post_deconv_strength.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("Post-deconv strength", self.green_post_deconv_strength)

        self.green_speckle_min_voxels = QSpinBox()
        self.green_speckle_min_voxels.setRange(2, 500)
        self.green_speckle_min_voxels.setValue(10)
        self.green_speckle_min_voxels.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("Speckle min voxels", self.green_speckle_min_voxels)

        self.green_speckle_attenuation = self._make_unit_spinbox(0.0, 1.0, 0.01, 0.12)
        self.green_speckle_attenuation.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("Speckle attenuation", self.green_speckle_attenuation)

        self.green_noise2void_model_path = QLineEdit()
        self.green_noise2void_model_path.setPlaceholderText("Optional TorchScript model path")
        self.green_noise2void_model_path.editingFinished.connect(self._emit_preprocess_config)
        form.addRow("N2V model path", self.green_noise2void_model_path)

        self.green_chunked_processing = QCheckBox("Chunked processing")
        self.green_chunked_processing.setChecked(True)
        self.green_chunked_processing.stateChanged.connect(self._emit_preprocess_config)
        form.addRow(self.green_chunked_processing)

        self.green_chunk_depth = QSpinBox()
        self.green_chunk_depth.setRange(8, 512)
        self.green_chunk_depth.setValue(48)
        self.green_chunk_depth.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("Chunk depth", self.green_chunk_depth)

        self.green_chunk_overlap = QSpinBox()
        self.green_chunk_overlap.setRange(0, 64)
        self.green_chunk_overlap.setValue(4)
        self.green_chunk_overlap.valueChanged.connect(self._emit_preprocess_config)
        form.addRow("Chunk overlap", self.green_chunk_overlap)

        self.preview_z_index = QSpinBox()
        self.preview_z_index.setRange(0, 100000)
        self.preview_z_index.setValue(0)
        form.addRow("Preview z index", self.preview_z_index)

        row = QHBoxLayout()
        preview_btn = QPushButton("Preview Green Denoise")
        preview_btn.clicked.connect(self.preview_green_denoise_requested.emit)
        apply_btn = QPushButton("Apply to Full Volume")
        apply_btn.clicked.connect(self.apply_green_denoise_requested.emit)
        row.addWidget(preview_btn)
        row.addWidget(apply_btn)
        form.addRow(row)
        return group

    @staticmethod
    def _make_unit_spinbox(minimum: float, maximum: float, step: float, value: float):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    @staticmethod
    def _make_offset_spinbox():
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setSingleStep(0.1)
        spin.setDecimals(3)
        spin.setValue(0.0)
        return spin

    def _emit_render_config(self) -> None:
        self.render_config_changed.emit(self.current_render_config())

    def _emit_psf_config(self) -> None:
        self.psf_config_changed.emit(self.current_psf_config())

    def _emit_preprocess_config(self) -> None:
        self.preprocess_config_changed.emit(self.current_preprocess_config())

    def current_render_config(self) -> RenderConfig:
        return RenderConfig(
            threshold_green=float(self.threshold_green.value()),
            threshold_red=float(self.threshold_red.value()),
            opacity_green=float(self.opacity_green.value()),
            opacity_red=float(self.opacity_red.value()),
            iso_green=float(self.iso_green.value()),
            iso_red=float(self.iso_red.value()),
            offset_x_um=float(self.offset_x.value()),
            offset_y_um=float(self.offset_y.value()),
            offset_z_um=float(self.offset_z.value()),
            show_green=self.show_green.isChecked(),
            show_red=self.show_red.isChecked(),
            show_iso_green=self.show_iso_green.isChecked(),
            show_iso_red=self.show_iso_red.isChecked(),
        )

    def current_psf_config(self) -> PSFConfig:
        return PSFConfig(
            enabled=self.psf_enabled.isChecked(),
            sigma_xy_um=float(self.sigma_xy.value()),
            sigma_z_um=float(self.sigma_z.value()),
            iterations=int(self.iterations.value()),
        )

    def current_preprocess_config(self) -> PreprocessConfig:
        patch_size = int(self.green_nlm_patch_size.value())
        if patch_size % 2 == 0:
            patch_size += 1
        return PreprocessConfig(
            green_denoise_strategy=str(self.green_strategy.currentText()),
            green_noise_model=str(self.green_noise_model.currentText()),
            green_branch_protection=float(self.green_branch_protection.value()),
            green_nlm_patch_size=int(patch_size),
            green_nlm_patch_distance=int(self.green_nlm_patch_distance.value()),
            green_nlm_h_factor=float(self.green_nlm_h_factor.value()),
            green_apply_vst=self.green_apply_vst.isChecked(),
            green_pre_deconv_strength=float(self.green_pre_deconv_strength.value()),
            green_post_deconv_strength=float(self.green_post_deconv_strength.value()),
            green_speckle_min_voxels=int(self.green_speckle_min_voxels.value()),
            green_speckle_attenuation=float(self.green_speckle_attenuation.value()),
            green_noise2void_model_path=self.green_noise2void_model_path.text().strip(),
            green_chunked_processing=self.green_chunked_processing.isChecked(),
            green_chunk_depth=int(self.green_chunk_depth.value()),
            green_chunk_overlap=int(self.green_chunk_overlap.value()),
        )

    def current_preview_z_index(self) -> int:
        return int(self.preview_z_index.value())

    def set_threshold_defaults(self, green: float, red: float) -> None:
        self.threshold_green.blockSignals(True)
        self.threshold_red.blockSignals(True)
        self.iso_green.blockSignals(True)
        self.iso_red.blockSignals(True)
        self.threshold_green.setValue(green)
        self.threshold_red.setValue(red)
        self.iso_green.setValue(min(1.0, max(0.0, green + 0.02)))
        self.iso_red.setValue(min(1.0, max(0.0, red + 0.05)))
        self.threshold_green.blockSignals(False)
        self.threshold_red.blockSignals(False)
        self.iso_green.blockSignals(False)
        self.iso_red.blockSignals(False)
        self._emit_render_config()

    def set_metrics_text(self, text: str) -> None:
        self.metrics_text.setPlainText(text)

    def set_plugin_text(self, text: str) -> None:
        self.plugin_text.setPlainText(text)

    def append_debug_text(self, text: str) -> None:
        self.debug_text.append(text)

    def clear_debug_text(self) -> None:
        self.debug_text.clear()
