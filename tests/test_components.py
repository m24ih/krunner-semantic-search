"""Comprehensive unit tests for KRunner Semantic Search storage, config, and D-Bus runner."""

from pathlib import Path
import tempfile

from krunner_semantic.config import Config, detect_device, load_config, save_config
from krunner_semantic.daemon import SemanticRunner, get_file_icon
from krunner_semantic.storage import StorageManager, VECTOR_DIMENSION


def test_detect_device():
    device = detect_device()
    assert device in ("cuda", "cpu")


def test_config_defaults_and_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "config.toml"
        cfg = load_config(cfg_path)
        assert cfg_path.exists()
        assert cfg.device == "auto"
        assert cfg.max_results == 5
        assert cfg.similarity_threshold == 0.55
        assert cfg.effective_device in ("cuda", "cpu")
        assert cfg.index_hidden_files is False
        assert len(cfg.enabled_presets) > 0
        assert len(cfg.active_exclude_patterns) > 0

        # Mutate and save
        cfg.max_results = 10
        cfg.similarity_threshold = 0.70
        cfg.index_hidden_files = True
        cfg.enabled_presets = ["git"]
        cfg.custom_excludes = [{"pattern": r"\.tmp$", "enabled": True}]
        save_config(cfg, cfg_path)

        reloaded = load_config(cfg_path)
        assert reloaded.max_results == 10
        assert reloaded.similarity_threshold == 0.70
        assert reloaded.index_hidden_files is True
        assert reloaded.enabled_presets == ["git"]
        assert len(reloaded.custom_excludes) == 1
        assert reloaded.custom_excludes[0]["pattern"] == r"\.tmp$"


def test_storage_manager_crud_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_storage.db"
        with StorageManager(db_file) as storage:
            # 1. Initially empty
            assert storage.count_files() == 0
            assert storage.count_chunks() == 0

            # 2. Upsert files
            file1 = Path(tmpdir) / "doc1.txt"
            file1.write_text("Alpha content")
            fid1 = storage.upsert_file(file1, mtime=100.0, file_size=13)
            assert fid1 == 1

            file2 = Path(tmpdir) / "doc2.md"
            file2.write_text("Beta content")
            fid2 = storage.upsert_file(file2, mtime=200.0, file_size=12)
            assert fid2 == 2

            # 3. Insert chunks
            vec1 = [0.0] * VECTOR_DIMENSION
            vec1[0] = 1.0  # Unit vector along index 0

            vec2 = [0.0] * VECTOR_DIMENSION
            vec2[1] = 1.0  # Unit vector along index 1 (orthogonal)

            storage.insert_chunks(fid1, [(0, "Chunk for doc1", vec1)])
            storage.insert_chunks(fid2, [(0, "Chunk for doc2", vec2)])

            assert storage.count_files() == 2
            assert storage.count_chunks() == 2

            # 4. Search matching doc1
            matches = storage.search(vec1, limit=5, threshold=0.55)
            assert len(matches) == 1
            assert matches[0]["file_path"] == str(file1.resolve())
            assert matches[0]["score"] >= 0.99
            assert matches[0]["text"] == "Chunk for doc1"

            # 5. Search matching both with threshold 0.0
            matches_all = storage.search(vec1, limit=5, threshold=0.0)
            assert len(matches_all) == 2

            # 6. Delete file1
            deleted = storage.delete_file(file1)
            assert deleted is True
            assert storage.count_files() == 1
            assert storage.count_chunks() == 1

            # doc1 should no longer be found
            assert len(storage.search(vec1, limit=5, threshold=0.55)) == 0


def test_get_file_icon():
    assert get_file_icon("notes.md") == "text-markdown"
    assert get_file_icon("paper.pdf") == "application-pdf"
    assert get_file_icon("main.py") == "text-x-python"
    assert get_file_icon("photo.jpg") == "image-x-generic"


def test_semantic_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "runner.db"
        storage = StorageManager(db_file)
        cfg = Config(db_path=db_file, max_results=5, similarity_threshold=0.5)

        doc = Path(tmpdir) / "thesis.pdf"
        doc.write_text("Quantum computing thesis")
        fid = storage.upsert_file(doc, mtime=300.0, file_size=24)

        target_vec = [0.0] * VECTOR_DIMENSION
        target_vec[5] = 1.0
        storage.insert_chunks(fid, [(0, "Quantum entanglement section", target_vec)])

        class MockEmbedder:
            def embed_query(self, query: str):
                v = [0.0] * VECTOR_DIMENSION
                if "quantum" in query.lower():
                    v[5] = 1.0
                return v

        runner = SemanticRunner(cfg, storage, MockEmbedder())

        # Test actions
        actions = runner.actions()
        assert len(actions) == 3
        assert actions[0][0] == "open"
        assert actions[1][0] == "open_folder"

        # Test prefix matching
        matches = runner.match("sem: quantum mechanics")
        assert len(matches) == 1
        m_id, text, icon, mtype, score, props = matches[0]
        assert text == "thesis.pdf"
        assert icon == "application-pdf"
        assert mtype == 100
        assert score >= 0.99
        assert "urls" in props

        # Test non-match
        assert len(runner.match("sem: biology")) == 0

        # Test short query
        assert len(runner.match("q")) == 0
