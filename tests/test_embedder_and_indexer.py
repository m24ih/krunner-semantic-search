"""Unit tests for embedder, extractors, and indexer modules."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import pytest
from PIL import Image

from krunner_semantic.config import Config
from krunner_semantic.embedder import (
    BaseEmbedder,
    LocalGemmaEmbedder,
    MockEmbedder,
    create_embedder,
    normalize_l2,
)
from krunner_semantic.extractors import (
    FileExtractorRegistry,
    ImageExtractor,
    PdfExtractor,
    TextExtractor,
    chunk_text,
    default_registry,
    extract_file,
    can_extract_file,
)
from krunner_semantic.indexer import SemanticIndexer
from krunner_semantic.storage import StorageManager


class TestEmbedder:
    """Tests for embedding functionality and normalization."""

    def test_normalize_l2_unit_length(self):
        v = [3.0, 4.0]
        norm = normalize_l2(v)
        assert abs(norm[0] - 0.6) < 1e-6
        assert abs(norm[1] - 0.8) < 1e-6
        assert abs(math.sqrt(sum(x * x for x in norm)) - 1.0) < 1e-6

    def test_normalize_l2_zero_vector(self):
        v = [0.0, 0.0, 0.0]
        norm = normalize_l2(v)
        assert norm == [0.0, 0.0, 0.0]

    def test_mock_embedder_deterministic(self):
        embedder = MockEmbedder(dimension=768)
        assert embedder.dimension == 768
        vec1 = embedder.embed_text("test sentence")
        vec2 = embedder.embed_text("test sentence")
        vec3 = embedder.embed_text("another text")

        assert len(vec1) == 768
        assert vec1 == vec2
        assert vec1 != vec3

        # Check unit L2 norm
        norm = math.sqrt(sum(x * x for x in vec1))
        assert abs(norm - 1.0) < 1e-6

    def test_mock_embedder_batch(self):
        embedder = MockEmbedder(dimension=768)
        texts = ["first text", "second text", "third text"]
        batch = embedder.embed_batch(texts)
        assert len(batch) == 3
        for v in batch:
            assert len(v) == 768
            assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6

    def test_local_gemma_fallback_to_mock(self):
        embedder = LocalGemmaEmbedder(model_path="nonexistent_model.gguf", fallback_to_mock=True)
        assert embedder.is_mock
        vec = embedder.embed_text("sample")
        assert len(vec) == 768
        assert abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-6


class TestExtractors:
    """Tests for file content extractors and chunking."""

    def test_chunk_text_basic(self):
        text = "Hello world! This is a test. Another sentence follows."
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 35
            assert c.strip()

    def test_chunk_text_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t ") == []

    def test_text_extractor(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def test():\n    return 42\n", encoding="utf-8")
        extracted = extract_file(f)
        assert extracted is not None
        assert "def test():" in extracted

    def test_image_extractor(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (640, 480), color="green")
        img.save(img_path)

        meta = extract_file(img_path)
        assert meta is not None
        assert "640x480" in meta
        assert "PNG" in meta
        assert "RGB" in meta

    def test_unsupported_file(self, tmp_path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\x03")
        assert not can_extract_file(f)
        assert extract_file(f) is None


class TestSemanticIndexer:
    """Tests for directory scanning, change detection, and indexing."""

    @pytest.fixture
    def test_env(self, tmp_path):
        db_path = tmp_path / "test_index.db"
        docs_dir = tmp_path / "documents"
        docs_dir.mkdir()

        doc1 = docs_dir / "plasma.md"
        doc1.write_text("# KDE Plasma\nFast semantic launcher integration.")

        doc2 = docs_dir / "notes.txt"
        doc2.write_text("Remember to buy milk and groceries.")

        cfg = Config(
            device="cpu",
            model_path=tmp_path / "dummy.gguf",
            db_path=db_path,
            watch_directories=[docs_dir],
            max_results=5,
            similarity_threshold=0.3,
        )

        storage = StorageManager(db_path)
        embedder = MockEmbedder(768)
        indexer = SemanticIndexer(config=cfg, storage=storage, embedder=embedder)

        return indexer, storage, docs_dir, doc1, doc2

    def test_index_file(self, test_env):
        indexer, storage, docs_dir, doc1, doc2 = test_env
        assert indexer.index_file(doc1) is True
        assert storage.count_files() == 1
        assert storage.count_chunks() == 1

        file_info = storage.get_file(doc1)
        assert file_info is not None
        assert file_info["file_path"] == str(doc1.resolve())

    def test_index_directory_and_incremental(self, test_env):
        indexer, storage, docs_dir, doc1, doc2 = test_env

        # Initial scan
        stats = indexer.index_directory(docs_dir)
        assert stats["scanned"] == 2
        assert stats["indexed"] == 2
        assert stats["skipped"] == 0
        assert storage.count_files() == 2

        # Second scan: files unchanged -> should be skipped
        stats2 = indexer.index_directory(docs_dir)
        assert stats2["scanned"] == 2
        assert stats2["indexed"] == 0
        assert stats2["skipped"] == 2

    def test_deleted_file_purge(self, test_env):
        indexer, storage, docs_dir, doc1, doc2 = test_env

        indexer.index_directory(docs_dir)
        assert storage.count_files() == 2

        # Delete doc2
        doc2.unlink()

        # Re-scan: doc2 should be removed from database
        stats = indexer.index_directory(docs_dir)
        assert stats["deleted"] == 1
        assert storage.count_files() == 1
        assert storage.get_file(doc2) is None

    def test_get_status(self, test_env):
        indexer, storage, docs_dir, doc1, doc2 = test_env
        indexer.index_directory(docs_dir)
        status = indexer.get_status()
        assert status["total_files"] == 2
        assert status["total_chunks"] == 2
        assert status["embedder_dimension"] == 768

    def test_hidden_files_and_regex_exclusions(self, test_env):
        import re
        indexer, storage, docs_dir, doc1, doc2 = test_env

        hidden_file = docs_dir / ".hidden_note.txt"
        hidden_file.write_text("Secret dotfile content")

        node_modules = docs_dir / "node_modules"
        node_modules.mkdir(exist_ok=True)
        (node_modules / "pkg.txt").write_text("Should be excluded")

        # Default: hidden files disabled, node_modules excluded
        indexer.config.index_hidden_files = False
        indexer._compiled_exclude_regexes = [
            re.compile(pat, re.IGNORECASE) for pat in indexer.config.active_exclude_patterns
        ]
        stats = indexer.index_directory(docs_dir, force=True)
        assert stats["indexed"] == 2
        assert storage.get_file(hidden_file) is None
        assert storage.get_file(node_modules / "pkg.txt") is None

        # Enable hidden files: .hidden_note.txt is now indexed, node_modules remains excluded
        indexer.config.index_hidden_files = True
        stats_hidden = indexer.index_directory(docs_dir, force=True)
        assert stats_hidden["indexed"] == 3
        assert storage.get_file(hidden_file) is not None
        assert storage.get_file(node_modules / "pkg.txt") is None
