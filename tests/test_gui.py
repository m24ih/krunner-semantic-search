"""Unit tests for PyQt6 Settings GUI and stacked TaskProgressWidget."""

from pathlib import Path
import sys
import tempfile

import pytest
from PyQt6.QtWidgets import QApplication

from krunner_semantic.config import Config
from krunner_semantic.gui import SettingsDialog, TaskProgressWidget


@pytest.fixture(scope="session")
def qapp():
    """Ensure a singleton QApplication exists for widget testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_task_progress_widget_lifecycle(qapp):
    """Test individual progress widget updates, finished state, and dismissal."""
    widget = TaskProgressWidget(task_id="test_task_1", title="Indexing: ~/Documents")

    # Initial state
    assert widget.task_id == "test_task_1"
    assert widget.title_text == "Indexing: ~/Documents"
    assert widget.progress_bar.value() == 0
    assert not widget.is_completed
    assert widget.dismiss_btn.isHidden()

    # Progress update
    widget.update_progress(75, "75% — Indexing: memo.txt (3/4)")
    assert widget.progress_bar.value() == 75
    assert "memo.txt" in widget.detail_label.text()

    # Finished state (success)
    dismiss_signal_emitted = []
    widget.dismiss_requested.connect(lambda tid: dismiss_signal_emitted.append(tid))

    widget.set_finished(True, "Indexing completed in 1.2s!\n\n• Scanned: 4 files")
    assert widget.is_completed
    assert widget.progress_bar.value() == 100
    assert "Finished" in widget.title_label.text()
    assert not widget.dismiss_btn.isHidden()

    # Trigger dismissal
    widget.dismiss_btn.click()
    assert dismiss_signal_emitted == ["test_task_1"]


def test_task_progress_widget_failure(qapp):
    """Test widget display when a task fails."""
    widget = TaskProgressWidget(task_id="task_fail", title="Indexing: Broken Folder")
    widget.set_finished(False, "Permission denied")
    assert widget.is_completed
    assert "Failed" in widget.title_label.text()
    assert "Permission denied" in widget.detail_label.text()
    assert not widget.dismiss_btn.isHidden()


def test_settings_dialog_stacked_tasks_concurrency(qapp):
    """Verify that multiple indexing tasks stack vertically and update independently."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = Config(
            db_path=tmp_path / "index.db",
            watch_directories=[tmp_path / "folder_a", tmp_path / "folder_b"],
        )

        dialog = SettingsDialog(config=cfg)

        # Initially tasks container is hidden
        assert dialog.tasks_scroll.isHidden()
        assert len(dialog.task_widgets) == 0

        # Simulate adding Task 1 (Folder A at 90%)
        task_1_id = "task_folder_a"
        widget_1 = TaskProgressWidget(task_1_id, title="Folder: folder_a")
        widget_1.dismiss_requested.connect(dialog._on_dismiss_task)
        dialog.tasks_layout.insertWidget(dialog.tasks_layout.count() - 1, widget_1)
        dialog.tasks_scroll.setVisible(True)
        dialog.task_widgets[task_1_id] = widget_1

        # Simulate adding Task 2 (Folder B at 68%) - stacked beneath Task 1
        task_2_id = "task_folder_b"
        widget_2 = TaskProgressWidget(task_2_id, title="Folder: folder_b")
        widget_2.dismiss_requested.connect(dialog._on_dismiss_task)
        dialog.tasks_layout.insertWidget(dialog.tasks_layout.count() - 1, widget_2)
        dialog.task_widgets[task_2_id] = widget_2

        # Both widgets are registered and stacked
        assert not dialog.tasks_scroll.isHidden()
        assert len(dialog.task_widgets) == 2

        # Update Task 1 to 90%
        dialog._on_task_progress(task_1_id, 90, "90% — doc1.pdf")
        # Update Task 2 to 68%
        dialog._on_task_progress(task_2_id, 68, "68% — notes.md")

        # Independent progress check: Task 1 must remain 90%, Task 2 must remain 68%
        assert widget_1.progress_bar.value() == 90
        assert widget_2.progress_bar.value() == 68

        # Advance Task 1 to 91%
        dialog._on_task_progress(task_1_id, 91, "91% — doc2.pdf")
        assert widget_1.progress_bar.value() == 91
        assert widget_2.progress_bar.value() == 68  # Task 2 untouched!

        # Complete Task 1
        dialog._on_task_finished(task_1_id, True, "Completed")
        assert widget_1.is_completed
        assert not widget_2.is_completed
        assert not dialog.clear_tasks_btn.isHidden()

        # Clear finished tasks removes Task 1 while leaving Task 2 active
        dialog._on_clear_finished_tasks()
        assert task_1_id not in dialog.task_widgets
        assert task_2_id in dialog.task_widgets
        assert len(dialog.task_widgets) == 1

        # Dismiss Task 2
        dialog._on_dismiss_task(task_2_id)
        assert len(dialog.task_widgets) == 0
        assert dialog.tasks_scroll.isHidden()
        assert dialog.clear_tasks_btn.isHidden()


def test_gui_model_status_and_exclusions(qapp):
    """Test model status recognition upon launch, hidden files, and regex pattern management."""
    from PyQt6.QtCore import Qt

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fake_model = tmp_path / "embeddinggemma-test.gguf"
        fake_model.write_bytes(b"GGUF_TEST_BYTES" * 100)

        cfg = Config(
            db_path=tmp_path / "index.db",
            model_path=fake_model,
            index_hidden_files=True,
            enabled_presets=["git", "node_modules"],
            custom_excludes=[{"pattern": r"\.tmp$", "enabled": True}],
        )

        dialog = SettingsDialog(config=cfg, config_path=tmp_path / "config.toml")

        # 1. Model status must immediately show Installed and NOT missing
        assert "Installed" in dialog.model_status_label.text()
        assert "missing" not in dialog.model_status_label.text().lower()

        # 2. Hidden files checkbox must be checked
        assert dialog.hidden_files_cb.isChecked()

        # 3. Presets must reflect config
        assert dialog.preset_checkboxes["git"].isChecked()
        assert dialog.preset_checkboxes["node_modules"].isChecked()
        assert not dialog.preset_checkboxes["python_venv"].isChecked()

        # 4. Custom excludes list populated with 1 item
        assert dialog.custom_exclude_list.count() == 1
        item = dialog.custom_exclude_list.item(0)
        assert item.text() == r"\.tmp$"
        assert item.checkState() == Qt.CheckState.Checked

        # 5. Add a new valid pattern
        dialog.new_pattern_edit.setText(r"secret_.*")
        dialog._on_add_custom_pattern()
        assert dialog.custom_exclude_list.count() == 2

        # 6. Save form data and verify updated config
        dialog.hidden_files_cb.setChecked(False)
        dialog._on_save(quiet=True)
        assert dialog.config.index_hidden_files is False
        assert len(dialog.config.custom_excludes) == 2
