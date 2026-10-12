"""Graphical Configuration Utility for KRunner Semantic Search.

Provides a native Qt6/KDE desktop interface for managing watched indexing
directories with real-time indexing progress percentage (%xx), selecting
hardware acceleration (CUDA/CPU), adjusting search thresholds, and managing
the official EmbeddingGemma 2 GGUF model.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from krunner_semantic.config import (
    PRESET_EXCLUDES,
    Config,
    detect_device,
    load_config,
    save_config,
)
from krunner_semantic.storage import StorageManager

logger = logging.getLogger(__name__)

OFFICIAL_MODEL_URL = (
    "https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/"
    "embeddinggemma-300M-Q8_0.gguf"
)
DEFAULT_MODEL_NAME = "embeddinggemma-300M-Q8_0.gguf"


class ModelDownloadWorker(QThread):
    """Background worker thread to download the EmbeddingGemma GGUF model with progress reporting."""

    progress_changed = pyqtSignal(int, str)
    download_finished = pyqtSignal(bool, str)

    def __init__(self, target_path: Path, url: str = OFFICIAL_MODEL_URL):
        super().__init__()
        self.target_path = target_path
        self.url = url
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import urllib.request

        try:
            self.target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.target_path.with_suffix(".part")

            req = urllib.request.Request(
                self.url,
                headers={"User-Agent": "krunner-semantic-search/0.1.0"},
            )

            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                block_size = 1024 * 512  # 512 KB

                with open(temp_path, "wb") as out_file:
                    while not self._cancelled:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        out_file.write(buffer)
                        downloaded += len(buffer)
                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            mb_down = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            self.progress_changed.emit(
                                percent, f"{percent}% ({mb_down:.1f} MB / {mb_total:.1f} MB)"
                            )

            if self._cancelled:
                if temp_path.exists():
                    temp_path.unlink()
                self.download_finished.emit(False, "Download was cancelled.")
                return

            temp_path.replace(self.target_path)
            self.download_finished.emit(
                True, f"Model successfully downloaded to {self.target_path}"
            )

        except Exception as e:
            self.download_finished.emit(False, str(e))


class TaskProgressWidget(QFrame):
    """Stacked progress widget representing an individual active or completed indexing task."""

    dismiss_requested = pyqtSignal(str)  # emits task_id when user dismisses

    def __init__(
        self,
        task_id: str,
        title: str,
        tooltip_path: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.task_id = task_id
        self.title_text = title
        self.is_completed = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        if tooltip_path:
            self.setToolTip(tooltip_path)

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(self.title_text)
        self.title_label.setStyleSheet("font-weight: bold;")

        self.dismiss_btn = QPushButton("✕")
        self.dismiss_btn.setToolTip("Dismiss this task")
        self.dismiss_btn.setFixedSize(22, 22)
        self.dismiss_btn.setVisible(False)
        self.dismiss_btn.clicked.connect(lambda: self.dismiss_requested.emit(self.task_id))

        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.dismiss_btn)
        layout.addLayout(header_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(14)
        layout.addWidget(self.progress_bar)

        self.detail_label = QLabel("Starting inspection...")
        self.detail_label.setStyleSheet("color: palette(highlight); font-size: 11px;")
        layout.addWidget(self.detail_label)

    def update_progress(self, percent: int, detail: str):
        self.progress_bar.setValue(percent)
        self.detail_label.setText(detail)

    def set_finished(self, success: bool, message: str):
        self.is_completed = True
        self.dismiss_btn.setVisible(True)
        if success:
            self.progress_bar.setValue(100)
            self.title_label.setText(f"✓ {self.title_text} — Finished")
            self.title_label.setStyleSheet("font-weight: bold; color: #2e7d32;")
            first_line = message.split("\n\n")[0] if "\n\n" in message else message
            self.detail_label.setText(first_line)
            self.detail_label.setStyleSheet("color: #2e7d32; font-size: 11px; font-weight: bold;")
        else:
            self.title_label.setText(f"⚠ {self.title_text} — Failed")
            self.title_label.setStyleSheet("font-weight: bold; color: #d32f2f;")
            self.detail_label.setText(f"Error: {message}")
            self.detail_label.setStyleSheet("color: #d32f2f; font-size: 11px;")


class ReindexWorker(QThread):
    """Background thread performing directory indexing with granular progress signals."""

    progress_updated = pyqtSignal(str, int, str)  # task_id, percent (0-100), status text
    reindex_finished = pyqtSignal(str, bool, str)  # task_id, success, message

    def __init__(
        self,
        config: Config,
        target_dir: Optional[Path] = None,
        force: bool = False,
        task_id: str = "",
    ):
        super().__init__()
        self.config = config
        self.target_dir = target_dir
        self.force = force
        self.task_id = task_id
        self._start_time = 0.0

    def run(self):
        try:
            from krunner_semantic.embedder import create_embedder
            from krunner_semantic.indexer import SemanticIndexer

            self._start_time = time.time()
            display_name = self.target_dir.name if self.target_dir else "All Watched Folders"
            self.progress_updated.emit(
                self.task_id, 0, f"Initializing Embedding Model for '{display_name}'..."
            )

            storage = StorageManager(self.config.db_path)
            embedder = create_embedder(
                model_path=self.config.model_path,
                device=self.config.effective_device,
                fallback_to_mock=True,
            )
            indexer = SemanticIndexer(self.config, storage, embedder)

            self.progress_updated.emit(
                self.task_id, 0, f"Scanning candidate files in '{display_name}'..."
            )

            def _callback(current: int, total: int, fname: str):
                if total > 0:
                    percent = min(100, int((current / total) * 100))
                    text = f"{percent}% — Indexing: {fname} ({current}/{total})"
                    self.progress_updated.emit(self.task_id, percent, text)

            if self.target_dir:
                stats = indexer.index_directory(
                    self.target_dir, force=self.force, progress_callback=_callback
                )
            else:
                stats = indexer.scan_all(force=self.force, progress_callback=_callback)

            elapsed = time.time() - self._start_time

            total_scanned = stats.get("scanned", 0)
            total_indexed = stats.get("indexed", 0)
            total_skipped = stats.get("skipped", 0)
            total_deleted = stats.get("deleted", 0)

            msg = (
                f"Indexing '{display_name}' completed in {elapsed:.1f}s!\n\n"
                f"• Scanned: {total_scanned} files\n"
                f"• Indexed/Updated: {total_indexed} files\n"
                f"• Unchanged (Skipped): {total_skipped} files\n"
                f"• Cleaned (Deleted): {total_deleted} files"
            )
            self.reindex_finished.emit(self.task_id, True, msg)

        except Exception as e:
            self.reindex_finished.emit(self.task_id, False, str(e))


class SettingsDialog(QDialog):
    """Native KDE/Qt6 Settings Window for KRunner Semantic Search (English First)."""

    def __init__(
        self,
        config: Optional[Config] = None,
        parent: Optional[QWidget] = None,
        config_path: Optional[Path] = None,
    ):
        super().__init__(parent)
        self.config_path = config_path
        self.config = config or load_config(config_path)
        self.download_worker: Optional[ModelDownloadWorker] = None
        self.workers: dict[str, ReindexWorker] = {}
        self.task_widgets: dict[str, TaskProgressWidget] = {}
        self._task_counter: int = 0

        self.setWindowTitle("Semantic Search Settings — KDE KRunner")
        self.setMinimumSize(740, 580)
        self.resize(780, 620)
        self.setWindowIcon(QIcon.fromTheme("system-search", QIcon.fromTheme("preferences-system-search")))

        self._build_ui()
        self._load_values()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        self.tab_widget = QTabWidget()

        # Tab 1: Folders & Indexing
        tab_folders = self._build_folders_tab()
        self.tab_widget.addTab(tab_folders, QIcon.fromTheme("folder"), "Folders && Indexing")

        # Tab 2: Exclusion Rules
        tab_exclusions = self._build_exclusions_tab()
        self.tab_widget.addTab(tab_exclusions, QIcon.fromTheme("dialog-cancel", QIcon.fromTheme("edit-delete")), "Exclusion Rules")

        # Tab 3: Model & Hardware
        tab_model = self._build_model_tab()
        self.tab_widget.addTab(tab_model, QIcon.fromTheme("preferences-desktop-display", QIcon.fromTheme("cpu")), "Model && Hardware")

        # Tab 4: Search Tuning
        tab_search = self._build_search_tab()
        self.tab_widget.addTab(tab_search, QIcon.fromTheme("system-search", QIcon.fromTheme("preferences-system-search")), "Search Tuning")

        main_layout.addWidget(self.tab_widget)

        # Bottom Action Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setIcon(QIcon.fromTheme("document-save"))
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._on_save)

        btn_box.addWidget(self.close_btn)
        btn_box.addWidget(self.save_btn)
        main_layout.addLayout(btn_box)

        self._refresh_stats()

    def _build_folders_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Watched Directories Group
        folder_group = QGroupBox("Watched Directories")
        folder_layout = QVBoxLayout(folder_group)

        self.dir_list = QListWidget()
        self.dir_list.setMinimumHeight(130)
        folder_layout.addWidget(self.dir_list)

        dir_btn_layout = QHBoxLayout()
        self.add_dir_btn = QPushButton("Add Folder...")
        self.add_dir_btn.setIcon(QIcon.fromTheme("list-add"))
        self.add_dir_btn.clicked.connect(self._on_add_directory)

        self.remove_dir_btn = QPushButton("Remove Selected")
        self.remove_dir_btn.setIcon(QIcon.fromTheme("list-remove"))
        self.remove_dir_btn.clicked.connect(self._on_remove_directory)

        self.index_dir_btn = QPushButton("Index Selected")
        self.index_dir_btn.setIcon(QIcon.fromTheme("system-search"))
        self.index_dir_btn.clicked.connect(self._on_index_selected_directory)

        dir_btn_layout.addWidget(self.add_dir_btn)
        dir_btn_layout.addWidget(self.remove_dir_btn)
        dir_btn_layout.addWidget(self.index_dir_btn)
        dir_btn_layout.addStretch()
        folder_layout.addLayout(dir_btn_layout)

        self.hidden_files_cb = QCheckBox("Index hidden files and directories (names starting with '.')")
        folder_layout.addWidget(self.hidden_files_cb)
        layout.addWidget(folder_group)

        # Indexing Controls & Live Status Group
        indexing_group = QGroupBox("Indexing Controls && Live Status")
        indexing_layout = QVBoxLayout(indexing_group)

        status_header_box = QHBoxLayout()
        self.stats_label = QLabel("Loading database status...")
        self.stats_label.setStyleSheet("font-weight: bold;")

        self.clear_tasks_btn = QPushButton("Clear Finished")
        self.clear_tasks_btn.setIcon(QIcon.fromTheme("edit-clear"))
        self.clear_tasks_btn.setVisible(False)
        self.clear_tasks_btn.clicked.connect(self._on_clear_finished_tasks)

        self.reindex_btn = QPushButton("Update All Folders")
        self.reindex_btn.setIcon(QIcon.fromTheme("view-refresh"))
        self.reindex_btn.clicked.connect(lambda: self._start_indexing(target_dir=None, force=False))

        self.reindex_force_btn = QPushButton("Force Re-index All")
        self.reindex_force_btn.setIcon(QIcon.fromTheme("system-reboot"))
        self.reindex_force_btn.clicked.connect(lambda: self._start_indexing(target_dir=None, force=True))

        status_header_box.addWidget(self.stats_label)
        status_header_box.addStretch()
        status_header_box.addWidget(self.clear_tasks_btn)
        status_header_box.addWidget(self.reindex_btn)
        status_header_box.addWidget(self.reindex_force_btn)
        indexing_layout.addLayout(status_header_box)

        # Scrollable container for stacked multi-task progress items
        self.tasks_scroll = QScrollArea()
        self.tasks_scroll.setWidgetResizable(True)
        self.tasks_scroll.setMinimumHeight(110)
        self.tasks_scroll.setVisible(False)
        self.tasks_scroll.setFrameShape(QFrame.Shape.StyledPanel)

        self.tasks_container = QWidget()
        self.tasks_layout = QVBoxLayout(self.tasks_container)
        self.tasks_layout.setContentsMargins(4, 4, 4, 4)
        self.tasks_layout.setSpacing(6)
        self.tasks_layout.addStretch()

        self.tasks_scroll.setWidget(self.tasks_container)
        indexing_layout.addWidget(self.tasks_scroll)

        layout.addWidget(indexing_group)
        return widget

    def _build_exclusions_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Predefined Exclusion Presets
        preset_group = QGroupBox("Predefined Exclusion Presets (Checked = Ignored)")
        preset_layout = QVBoxLayout(preset_group)

        preset_grid = QGridLayout()
        preset_grid.setHorizontalSpacing(24)
        preset_grid.setVerticalSpacing(8)
        self.preset_checkboxes: dict[str, QCheckBox] = {}
        row, col = 0, 0
        for key, info in PRESET_EXCLUDES.items():
            cb = QCheckBox(info["label"])
            self.preset_checkboxes[key] = cb
            preset_grid.addWidget(cb, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        preset_layout.addLayout(preset_grid)
        layout.addWidget(preset_group)

        # Custom Regular Expression Exclusion Patterns
        custom_group = QGroupBox("Custom Regular Expression Exclusion Patterns")
        custom_layout = QVBoxLayout(custom_group)

        tip_label = QLabel(
            "💡 Checked patterns are excluded from indexing. "
            "Patterns match file or directory names and paths (e.g. \\.tmp$, secret_.*, or (^|/)cache(/|$))."
        )
        tip_label.setStyleSheet("color: #666; font-size: 11px;")
        tip_label.setWordWrap(True)
        custom_layout.addWidget(tip_label)

        self.custom_exclude_list = QListWidget()
        self.custom_exclude_list.setMinimumHeight(170)
        custom_layout.addWidget(self.custom_exclude_list)

        pattern_btn_layout = QHBoxLayout()
        self.new_pattern_edit = QLineEdit()
        self.new_pattern_edit.setPlaceholderText(r"Enter regex pattern (e.g. \.tmp$ or secret_.*)")
        self.new_pattern_edit.returnPressed.connect(self._on_add_custom_pattern)

        self.add_pattern_btn = QPushButton("Add Pattern")
        self.add_pattern_btn.setIcon(QIcon.fromTheme("list-add"))
        self.add_pattern_btn.clicked.connect(self._on_add_custom_pattern)

        self.remove_pattern_btn = QPushButton("Remove Selected")
        self.remove_pattern_btn.setIcon(QIcon.fromTheme("list-remove"))
        self.remove_pattern_btn.clicked.connect(self._on_remove_custom_pattern)

        pattern_btn_layout.addWidget(self.new_pattern_edit)
        pattern_btn_layout.addWidget(self.add_pattern_btn)
        pattern_btn_layout.addWidget(self.remove_pattern_btn)
        custom_layout.addLayout(pattern_btn_layout)

        layout.addWidget(custom_group)
        return widget

    def _build_model_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Hardware selection
        hw_group = QGroupBox("Inference Device")
        hw_layout = QFormLayout(hw_group)

        self.device_combo = QComboBox()
        detected = detect_device()
        self.device_combo.addItem(f"Auto (Recommended — Detected: {detected.upper()})", "auto")
        self.device_combo.addItem("CUDA GPU (NVIDIA Hardware Acceleration)", "cuda")
        self.device_combo.addItem("CPU (Pure CPU Mode)", "cpu")
        hw_layout.addRow("Compute Device:", self.device_combo)
        layout.addWidget(hw_group)

        # Embedding Model settings
        model_group = QGroupBox("Embedding Model (EmbeddingGemma 2)")
        model_layout = QFormLayout(model_group)

        model_file_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.textChanged.connect(self._check_model_status)
        self.browse_model_btn = QPushButton("Browse...")
        self.browse_model_btn.setIcon(QIcon.fromTheme("document-open"))
        self.browse_model_btn.clicked.connect(self._on_browse_model)

        model_file_layout.addWidget(self.model_path_edit)
        model_file_layout.addWidget(self.browse_model_btn)
        model_layout.addRow("Model File (.gguf):", model_file_layout)

        download_box = QHBoxLayout()
        self.model_status_label = QLabel()
        self.download_model_btn = QPushButton("Download EmbeddingGemma 2 (333 MB)")
        self.download_model_btn.setIcon(QIcon.fromTheme("download"))
        self.download_model_btn.clicked.connect(self._on_download_model)

        self.download_progress = QProgressBar()
        self.download_progress.setVisible(False)

        download_box.addWidget(self.model_status_label)
        download_box.addWidget(self.download_model_btn)
        download_box.addWidget(self.download_progress)
        download_box.addStretch()
        model_layout.addRow("Model Status:", download_box)

        layout.addWidget(model_group)

        # Architecture Specifications Card
        info_group = QGroupBox("Architecture Specifications")
        info_layout = QVBoxLayout(info_group)
        info_text = QLabel(
            "• <b>Model:</b> Google EmbeddingGemma 300M (Q8_0 quantized GGUF)<br>"
            "• <b>Vector Dimension:</b> 768 dimensions (L2 normalized for cosine similarity)<br>"
            "• <b>Storage Engine:</b> SQLite with sqlite-vec extension (Cosine distance KNN)<br>"
            "• <b>Hardware Offload:</b> Fully accelerated via cuBLAS and CUDA v12 runtime"
        )
        info_text.setTextFormat(Qt.TextFormat.RichText)
        info_layout.addWidget(info_text)
        layout.addWidget(info_group)

        layout.addStretch()
        return widget

    def _build_search_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Search Matching Parameters
        search_group = QGroupBox("Search Matching Parameters")
        search_layout = QFormLayout(search_group)

        self.max_results_spin = QSpinBox()
        self.max_results_spin.setRange(1, 25)
        self.max_results_spin.setValue(5)
        search_layout.addRow("Maximum Match Results:", self.max_results_spin)

        threshold_box = QHBoxLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(10, 90)
        self.threshold_slider.setValue(55)
        self.threshold_val_label = QLabel("0.55")
        self.threshold_val_label.setStyleSheet("font-weight: bold; min-width: 36px;")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_val_label.setText(f"{v / 100:.2f}")
        )
        threshold_box.addWidget(self.threshold_slider)
        threshold_box.addWidget(self.threshold_val_label)
        search_layout.addRow("Similarity Threshold:", threshold_box)

        thresh_hint = QLabel("Lower threshold = broader fuzzy matches. Higher threshold = stricter semantic relevance.")
        thresh_hint.setStyleSheet("color: #666; font-size: 11px;")
        search_layout.addRow("", thresh_hint)

        layout.addWidget(search_group)

        # KRunner Usage Guide
        guide_group = QGroupBox("How to Search in KDE Plasma KRunner")
        guide_layout = QVBoxLayout(guide_group)
        guide_text = QLabel(
            "• Open KRunner with <b>Alt + Space</b> or <b>Alt + F2</b>.<br>"
            "• Type <b>sem: &lt;query&gt;</b> (e.g. <code>sem: machine learning notes</code> or <code>sem: invoice 2026</code>).<br>"
            "• Direct queries also match semantically when natural language patterns are used.<br>"
            "• Press <b>Enter</b> to instantly open the matched document or file."
        )
        guide_text.setTextFormat(Qt.TextFormat.RichText)
        guide_layout.addWidget(guide_text)
        layout.addWidget(guide_group)

        layout.addStretch()
        return widget

    def _load_values(self):
        """Populate UI widgets with configuration values."""
        self.dir_list.clear()
        for d in self.config.watch_directories:
            item = QListWidgetItem(str(d))
            item.setIcon(QIcon.fromTheme("folder"))
            self.dir_list.addItem(item)

        # Hidden files
        self.hidden_files_cb.setChecked(self.config.index_hidden_files)

        # Device
        idx = self.device_combo.findData(self.config.device)
        if idx >= 0:
            self.device_combo.setCurrentIndex(idx)

        # Model Path
        self.model_path_edit.setText(str(self.config.model_path))

        # Exclusion Presets
        for key, cb in self.preset_checkboxes.items():
            cb.setChecked(key in self.config.enabled_presets)

        # Custom Excludes
        self.custom_exclude_list.clear()
        for item_data in self.config.custom_excludes:
            pat = item_data.get("pattern", "")
            if not pat:
                continue
            item = QListWidgetItem(pat)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            state = Qt.CheckState.Checked if item_data.get("enabled", True) else Qt.CheckState.Unchecked
            item.setCheckState(state)
            self.custom_exclude_list.addItem(item)

        # Search tuning
        self.max_results_spin.setValue(self.config.max_results)
        slider_val = int(self.config.similarity_threshold * 100)
        self.threshold_slider.setValue(slider_val)
        self.threshold_val_label.setText(f"{self.config.similarity_threshold:.2f}")

        # Check model status with freshly populated model path
        self._check_model_status()

    def _check_model_status(self):
        raw_text = self.model_path_edit.text().strip()
        if not raw_text:
            self.model_status_label.setText("⚠ No model file specified")
            self.model_status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self.download_model_btn.setText("Download EmbeddingGemma 2 (333 MB)")
            return

        p = Path(raw_text).expanduser().resolve()
        if p.exists() and p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            self.model_status_label.setText(f"✓ Installed ({p.name}, {size_mb:.0f} MB)")
            self.model_status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self.download_model_btn.setText("Re-download Model")
        else:
            self.model_status_label.setText("⚠ Model missing (Mock fallback active)")
            self.model_status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
            self.download_model_btn.setText("Download EmbeddingGemma 2 (333 MB)")

    def _on_add_custom_pattern(self):
        pat = self.new_pattern_edit.text().strip()
        if not pat:
            return
        import re
        try:
            re.compile(pat)
        except re.error as err:
            QMessageBox.warning(self, "Invalid Regex", f"The regular expression pattern is invalid:\n{err}")
            return

        for i in range(self.custom_exclude_list.count()):
            if self.custom_exclude_list.item(i).text() == pat:
                self.new_pattern_edit.clear()
                return

        item = QListWidgetItem(pat)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self.custom_exclude_list.addItem(item)
        self.new_pattern_edit.clear()

    def _on_remove_custom_pattern(self):
        row = self.custom_exclude_list.currentRow()
        if row >= 0:
            self.custom_exclude_list.takeItem(row)

    def _on_add_directory(self):
        dir_selected = QFileDialog.getExistingDirectory(
            self, "Select Folder to Index", str(Path.home())
        )
        if dir_selected:
            path_obj = Path(dir_selected).resolve()
            for i in range(self.dir_list.count()):
                if self.dir_list.item(i).text() == str(path_obj):
                    return
            item = QListWidgetItem(str(path_obj))
            item.setIcon(QIcon.fromTheme("folder"))
            self.dir_list.addItem(item)
            # Prompt user to index newly added folder
            reply = QMessageBox.question(
                self,
                "Folder Added",
                f"Folder '{path_obj.name}' added to watch list.\n"
                "Would you like to start indexing it right now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._on_save(quiet=True)
                self._start_indexing(target_dir=path_obj, force=False)

    def _on_index_selected_directory(self):
        item = self.dir_list.currentItem()
        if not item:
            QMessageBox.information(
                self, "No Folder Selected", "Please select a folder from the list first."
            )
            return
        folder_path = Path(item.text())
        self._on_save(quiet=True)
        self._start_indexing(target_dir=folder_path, force=False)

    def _on_remove_directory(self):
        current_row = self.dir_list.currentRow()
        if current_row >= 0:
            removed_item = self.dir_list.takeItem(current_row)
            if removed_item:
                p = Path(removed_item.text())
                # Ask to purge deleted folder from index database
                reply = QMessageBox.question(
                    self,
                    "Remove Folder",
                    f"Remove '{p.name}' from index list?\n"
                    "Its indexed records will be purged from the search database.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._on_save(quiet=True)
                    self._start_indexing(target_dir=None, force=False)

    def _on_browse_model(self):
        start_dir = str(self.config.model_path.parent if self.config.model_path else Path.home())
        file_selected, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Embedding Model", start_dir, "GGUF Model Files (*.gguf);;All Files (*)"
        )
        if file_selected:
            self.model_path_edit.setText(file_selected)
            self._check_model_status()

    def _on_download_model(self):
        models_dir = Path.home() / ".local" / "share" / "krunner-semantic" / "models"
        target_model = models_dir / DEFAULT_MODEL_NAME

        if target_model.exists():
            resp = QMessageBox.question(
                self,
                "Model Already Exists",
                f"Model file already exists ({target_model.name}).\nDo you want to re-download it?",
            )
            if resp != QMessageBox.StandardButton.Yes:
                self.model_path_edit.setText(str(target_model))
                self._check_model_status()
                return

        self.download_model_btn.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self.model_status_label.setText("Downloading official model from Hugging Face...")

        self.download_worker = ModelDownloadWorker(target_model)
        self.download_worker.progress_changed.connect(
            lambda pct, text: (self.download_progress.setValue(pct), self.model_status_label.setText(text))
        )
        self.download_worker.download_finished.connect(self._on_download_finished)
        self.download_worker.start()

    def _on_download_finished(self, success: bool, message: str):
        self.download_model_btn.setEnabled(True)
        self.download_progress.setVisible(False)

        if success:
            models_dir = Path.home() / ".local" / "share" / "krunner-semantic" / "models"
            target_model = models_dir / DEFAULT_MODEL_NAME
            self.model_path_edit.setText(str(target_model))
            self._check_model_status()
            QMessageBox.information(
                self, "Download Complete", "EmbeddingGemma 2 GGUF model is downloaded and ready for GPU/CPU inference!"
            )
        else:
            self._check_model_status()
            QMessageBox.critical(self, "Download Failed", f"Could not download model: {message}")

    def _refresh_stats(self):
        try:
            storage = StorageManager(self.config.db_path)
            f_count = storage.count_files()
            c_count = storage.count_chunks()
            self.stats_label.setText(f"Status: {f_count} files, {c_count} chunks indexed")
        except Exception:
            self.stats_label.setText("Status: Database not initialized yet.")

    def _start_indexing(self, target_dir: Optional[Path] = None, force: bool = False):
        """Start background indexing worker with its own dedicated stacked progress widget."""
        self._on_save(quiet=True)

        # Check if identical task is already running
        for w in self.workers.values():
            if w.isRunning() and w.target_dir == target_dir and w.force == force:
                name = target_dir.name if target_dir else "All Folders"
                QMessageBox.information(
                    self,
                    "Task Already Running",
                    f"An indexing task for '{name}' is already in progress.",
                )
                return

        self._task_counter += 1
        task_id = f"task_{self._task_counter}_{int(time.time())}"

        if target_dir:
            title = f"Folder: {target_dir.name}"
            tooltip = str(target_dir.resolve())
        else:
            title = f"All Folders{' (Force)' if force else ''}"
            tooltip = "Scanning all configured watch folders"

        task_widget = TaskProgressWidget(task_id, title=title, tooltip_path=tooltip)
        task_widget.dismiss_requested.connect(self._on_dismiss_task)

        # Insert before bottom stretch in tasks_layout
        self.tasks_layout.insertWidget(self.tasks_layout.count() - 1, task_widget)
        self.tasks_scroll.setVisible(True)

        worker = ReindexWorker(self.config, target_dir=target_dir, force=force, task_id=task_id)
        worker.progress_updated.connect(self._on_task_progress)
        worker.reindex_finished.connect(self._on_task_finished)

        self.workers[task_id] = worker
        self.task_widgets[task_id] = task_widget
        worker.start()

    def _on_task_progress(self, task_id: str, percent: int, detail: str):
        widget = self.task_widgets.get(task_id)
        if widget:
            widget.update_progress(percent, detail)

    def _on_task_finished(self, task_id: str, success: bool, msg: str):
        widget = self.task_widgets.get(task_id)
        if widget:
            widget.set_finished(success, msg)
        self._refresh_stats()
        self._update_clear_button_visibility()

    def _on_dismiss_task(self, task_id: str):
        if task_id in self.workers:
            w = self.workers.pop(task_id)
            if w.isRunning():
                w.terminate()
        widget = self.task_widgets.pop(task_id, None)
        if widget:
            self.tasks_layout.removeWidget(widget)
            widget.deleteLater()

        # Hide scroll area if no tasks remain
        if len(self.task_widgets) == 0:
            self.tasks_scroll.setVisible(False)
        self._update_clear_button_visibility()

    def _on_clear_finished_tasks(self):
        finished_ids = [
            tid for tid, widget in self.task_widgets.items() if widget.is_completed
        ]
        for tid in finished_ids:
            self._on_dismiss_task(tid)

    def _update_clear_button_visibility(self):
        has_completed = any(w.is_completed for w in self.task_widgets.values())
        self.clear_tasks_btn.setVisible(has_completed)

    def _on_save(self, quiet: bool = False):
        """Save form data into config.toml."""
        new_dirs = [Path(self.dir_list.item(i).text()) for i in range(self.dir_list.count())]
        self.config.watch_directories = new_dirs
        self.config.index_hidden_files = self.hidden_files_cb.isChecked()
        self.config.device = self.device_combo.currentData()
        self.config.model_path = Path(self.model_path_edit.text().strip())
        self.config.max_results = self.max_results_spin.value()
        self.config.similarity_threshold = self.threshold_slider.value() / 100.0

        # Save enabled presets
        self.config.enabled_presets = [
            k for k, cb in self.preset_checkboxes.items() if cb.isChecked()
        ]

        # Save custom exclusion patterns
        custom_items = []
        for i in range(self.custom_exclude_list.count()):
            item = self.custom_exclude_list.item(i)
            custom_items.append({
                "pattern": item.text(),
                "enabled": item.checkState() == Qt.CheckState.Checked,
            })
        self.config.custom_excludes = custom_items

        save_config(self.config, self.config_path)
        self._check_model_status()

        if not quiet:
            QMessageBox.information(self, "Saved", "Settings saved successfully!")
            self.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("krunner-semantic-search")
    app.setDesktopFileName("krunner-semantic-config")

    dialog = SettingsDialog()
    dialog.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
