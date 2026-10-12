"""Text embedding module for KRunner Semantic Search.

Provides abstract interface BaseEmbedder along with LocalGemmaEmbedder (powered by
llama-cpp-python for local GGUF models) and MockEmbedder for development/testing.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import math
import random
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

def _setup_cuda_env() -> None:
    """Preload NVIDIA CUDA/cuBLAS runtime shared libraries from site-packages if present."""
    import ctypes
    import glob
    import os
    import sys

    search_dirs = [
        f"{sys.prefix}/lib/python*/site-packages/nvidia/*/lib",
        f"{sys.base_prefix}/lib/python*/site-packages/nvidia/*/lib",
        os.path.expanduser("~/.local/lib/python*/site-packages/nvidia/*/lib"),
    ]
    for pattern in search_dirs:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                if path not in os.environ.get("LD_LIBRARY_PATH", ""):
                    os.environ["LD_LIBRARY_PATH"] = f"{path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                for so_file in sorted(glob.glob(f"{path}/*.so*")):
                    try:
                        ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                    except Exception:
                        pass


_setup_cuda_env()

# Optional llama_cpp import
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    llama_cpp = None
    LLAMA_CPP_AVAILABLE = False


def normalize_l2(vector: Sequence[float]) -> list[float]:
    """Normalize a vector to unit L2 length (magnitude 1.0).

    This is critical for cosine distance indexing and querying in sqlite-vec.
    If the vector has zero or near-zero magnitude, returns a zero vector of the same dimension.

    Args:
        vector: Input sequence of float numbers.

    Returns:
        A list of floats representing the L2-normalized vector.
    """
    squared_sum = sum(float(x) * float(x) for x in vector)
    norm = math.sqrt(squared_sum)
    if norm < 1e-12:
        return [0.0] * len(vector)
    return [float(x) / norm for x in vector]


class BaseEmbedder(abc.ABC):
    """Abstract base class for all semantic embedding providers."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Vector embedding dimension (e.g., 768 for EmbeddingGemma)."""
        pass

    @abc.abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized embedding vector for a single text string.

        Args:
            text: Input text string.

        Returns:
            A list of floats with length equal to self.dimension and unit L2 norm.
        """
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for a batch of text strings.

        Args:
            texts: List of input text strings.

        Returns:
            A list of lists of floats, each with unit L2 norm.
        """
        pass


    @property
    def is_mock(self) -> bool:
        """Return True if this embedder is running in mock mode."""
        return False


class MockEmbedder(BaseEmbedder):
    """Deterministic mock embedder for development, testing, and fallback.

    Generates reproducible 768-dimensional unit-norm vectors by hashing the input
    text with SHA-256 and seeding a pseudorandom Gaussian distribution.
    """

    def __init__(self, dimension: int = 768):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_mock(self) -> bool:
        return True

    def embed_text(self, text: str) -> list[float]:
        """Generate a deterministic 768-dimensional unit vector for text.

        Uses token-averaged hashing so documents containing the query words
        produce positive cosine similarities, allowing KRunner test searches
        to function realistically before the full GGUF model is downloaded.
        """
        import re

        words = re.findall(r"\w+", text.lower()) if text else []
        if not words:
            seed = hashlib.sha256(b"__empty__").digest()
            rng = random.Random(seed)
            return normalize_l2([rng.gauss(0.0, 1.0) for _ in range(self._dim)])

        accum = [0.0] * self._dim
        for word in set(words):
            word_seed = hashlib.sha256(word.encode("utf-8")).digest()
            rng = random.Random(word_seed)
            for i in range(self._dim):
                accum[i] += rng.gauss(0.0, 1.0)

        return normalize_l2(accum)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic unit vectors for a list of texts."""
        return [self.embed_text(t) for t in texts]


class LocalGemmaEmbedder(BaseEmbedder):
    """Local GGUF model embedder powered by llama-cpp-python.

    Optimized for EmbeddingGemma 2 or similar 768-dim GGUF embedding models.
    Supports GPU offloading (CUDA) with automatic CPU fallback, and graceful
    MockEmbedder fallback when the model file is missing or llama-cpp is unavailable.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "auto",
        dimension: int = 768,
        n_ctx: int = 2048,
        fallback_to_mock: bool = True,
        verbose: bool = False,
    ):
        """Initialize the local Gemma embedder.

        Args:
            model_path: Path to GGUF model file on disk.
            device: Compute device target: 'auto', 'cuda', or 'cpu'.
            dimension: Expected vector dimension (default: 768).
            n_ctx: Context window size in tokens (default: 2048).
            fallback_to_mock: If True, falls back to MockEmbedder if model file
                              is not found or llama-cpp is not installed.
            verbose: Enable verbose logging from llama.cpp.
        """
        self._dim = dimension
        self._device = device.lower().strip()
        self._n_ctx = n_ctx
        self._verbose = verbose
        self._model_path = Path(model_path).expanduser().resolve() if model_path else None
        self._llm = None
        self._fallback_embedder: MockEmbedder | None = None

        # Check availability of llama-cpp-python
        if not LLAMA_CPP_AVAILABLE:
            if fallback_to_mock:
                logger.warning(
                    "llama-cpp-python library is not installed. "
                    "Falling back to MockEmbedder for development/testing."
                )
                self._fallback_embedder = MockEmbedder(dimension=self._dim)
                return
            raise ImportError(
                "llama-cpp-python is required to run LocalGemmaEmbedder. "
                "Install it via `pip install llama-cpp-python`."
            )

        # Check existence of model file
        if self._model_path is None or not self._model_path.is_file():
            if fallback_to_mock:
                logger.warning(
                    "Model file not found at '%s'. "
                    "Falling back to MockEmbedder for development/testing.",
                    self._model_path,
                )
                self._fallback_embedder = MockEmbedder(dimension=self._dim)
                return
            raise FileNotFoundError(
                f"Model file not found at: {self._model_path}"
            )

        # Attempt model initialization with GPU/CPU fallback
        self._load_model()

    def _load_model(self) -> None:
        """Load the GGUF model with CUDA offload and fallback to CPU on error."""
        assert llama_cpp is not None
        assert self._model_path is not None

        wants_cuda = self._device in ("cuda", "auto")

        if wants_cuda:
            try:
                logger.info(
                    "Attempting to load GGUF model '%s' with CUDA (n_gpu_layers=-1)...",
                    self._model_path.name,
                )
                self._llm = llama_cpp.Llama(
                    model_path=str(self._model_path),
                    embedding=True,
                    n_gpu_layers=-1,
                    n_ctx=self._n_ctx,
                    verbose=self._verbose,
                )
                logger.info("Successfully loaded GGUF model with CUDA acceleration.")
                return
            except Exception as e:
                logger.warning(
                    "CUDA initialization failed (%s). Falling back to CPU (n_gpu_layers=0).",
                    e,
                )

        # CPU fallback or explicit CPU request
        try:
            logger.info(
                "Loading GGUF model '%s' on CPU (n_gpu_layers=0)...",
                self._model_path.name,
            )
            self._llm = llama_cpp.Llama(
                model_path=str(self._model_path),
                embedding=True,
                n_gpu_layers=0,
                n_ctx=self._n_ctx,
                verbose=self._verbose,
            )
            logger.info("Successfully loaded GGUF model on CPU.")
        except Exception as e:
            logger.error("Failed to load model on CPU: %s", e)
            if self._fallback_embedder is None:
                logger.warning("Defaulting to MockEmbedder after model loading failure.")
                self._fallback_embedder = MockEmbedder(dimension=self._dim)
            else:
                raise

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def is_mock(self) -> bool:
        """Returns True if the embedder is currently running in mock mode."""
        return self._fallback_embedder is not None

    def embed_text(self, text: str) -> list[float]:
        """Generate a normalized 768-dim embedding vector for text."""
        if self._fallback_embedder is not None:
            return self._fallback_embedder.embed_text(text)

        assert self._llm is not None
        cleaned = text.strip() if text else ""
        if not cleaned:
            # Empty text: generate normalized zero or mock padding
            return [0.0] * self._dim

        # Try embed() API from llama_cpp.Llama
        try:
            raw = self._llm.embed(cleaned, truncate=True)
            if isinstance(raw, list):
                if len(raw) > 0 and isinstance(raw[0], (int, float)):
                    vec = raw
                elif len(raw) > 0 and isinstance(raw[0], list):
                    vec = raw[0]
                else:
                    vec = [0.0] * self._dim
            else:
                vec = list(raw)
        except Exception:
            # Fallback to create_embedding API
            resp = self._llm.create_embedding(cleaned)
            vec = resp["data"][0]["embedding"]

        # Ensure vector length matches dimension
        if len(vec) > self._dim:
            vec = vec[:self._dim]
        elif len(vec) < self._dim:
            vec = vec + [0.0] * (self._dim - len(vec))

        return normalize_l2(vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embedding vectors for a list of texts."""
        if self._fallback_embedder is not None:
            return self._fallback_embedder.embed_batch(texts)

        if not texts:
            return []

        # Batch embedding with chunking to prevent memory exhaustion
        results: list[list[float]] = []
        batch_chunk_size = 16

        for i in range(0, len(texts), batch_chunk_size):
            chunk_texts = texts[i : i + batch_chunk_size]
            cleaned_texts = [t.strip() if t else "" for t in chunk_texts]

            try:
                assert self._llm is not None
                raw_batch = self._llm.embed(cleaned_texts, truncate=True)
                if isinstance(raw_batch, list) and len(raw_batch) > 0 and isinstance(raw_batch[0], list):
                    for raw_vec in raw_batch:
                        results.append(normalize_l2(raw_vec[:self._dim]))
                else:
                    # In case embed didn't return a 2D list
                    for t in cleaned_texts:
                        results.append(self.embed_text(t))
            except Exception as e:
                logger.debug("Batch embedding call failed (%s), falling back to sequential: ", e)
                for t in cleaned_texts:
                    results.append(self.embed_text(t))

        return results


def create_embedder(
    model_path: str | Path | Any = None,
    device: str = "auto",
    fallback_to_mock: bool = True,
) -> BaseEmbedder:
    """Convenience factory function to create an embedder instance.

    Args:
        model_path: Optional path to the GGUF model file, or a Config instance.
        device: Device string ('auto', 'cuda', 'cpu').
        fallback_to_mock: Whether to fall back to MockEmbedder if model is missing.

    Returns:
        An instance of BaseEmbedder (LocalGemmaEmbedder or MockEmbedder).
    """
    if model_path is not None and hasattr(model_path, "model_path") and hasattr(model_path, "device"):
        actual_model_path = model_path.model_path
        actual_device = model_path.device
    else:
        actual_model_path = model_path
        actual_device = device

    return LocalGemmaEmbedder(
        model_path=actual_model_path,
        device=actual_device,
        fallback_to_mock=fallback_to_mock,
    )

