"""KRunner Semantic Search.

Fast, local, privacy-first semantic file search plugin for KDE Plasma KRunner
powered by EmbeddingGemma.
"""

from __future__ import annotations

from krunner_semantic.config import Config, load_config
from krunner_semantic.embedder import (
    BaseEmbedder,
    LocalGemmaEmbedder,
    MockEmbedder,
    create_embedder,
    normalize_l2,
)
from krunner_semantic.extractors import (
    BaseExtractor,
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "BaseEmbedder",
    "LocalGemmaEmbedder",
    "MockEmbedder",
    "create_embedder",
    "normalize_l2",
    "BaseExtractor",
    "TextExtractor",
    "PdfExtractor",
    "ImageExtractor",
    "FileExtractorRegistry",
    "default_registry",
    "chunk_text",
    "extract_file",
    "can_extract_file",
    "SemanticIndexer",
    "StorageManager",
]
