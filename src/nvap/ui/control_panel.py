from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

from nvap.config.types import PSFConfig, RenderConfig


class ControlPanel(QWidget):
    load_requested = Signal()
    apply_psf_requested = Signal()
    render_config_changed = Signal(object)
    psf_config_changed = Signal(object)
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
