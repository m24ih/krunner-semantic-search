"""File content extraction and text chunking module for KRunner Semantic Search.

Extracts text content from plain text/code files, PDFs (via pypdfium2), and images
(via Pillow metadata and EXIF) for embedding and semantic search indexing.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# Optional pypdfium2 import
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    pdfium = None
    PDFIUM_AVAILABLE = False

# Optional Pillow import
try:
    from PIL import ExifTags, Image
    PILLOW_AVAILABLE = True
except ImportError:
    Image = None
    ExifTags = None
    PILLOW_AVAILABLE = False


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping segments suitable for semantic embedding.

    Uses a sliding window approach with intelligent boundary detection (preferring
    paragraph breaks, newlines, sentence ends, or spaces over mid-word cuts).

    Args:
        text: Input text string to chunk.
        chunk_size: Target maximum character length for each chunk (default: 500).
        overlap: Character overlap between consecutive chunks (default: 100).

    Returns:
        A list of non-empty string chunks.
    """
    if not text or not text.strip():
        return []

    cleaned_text = text.strip()
    text_len = len(cleaned_text)

    # Sanity checks for chunk_size and overlap
    if chunk_size <= 0:
        chunk_size = 500
    if overlap < 0:
        overlap = 0
    if overlap >= chunk_size:
        overlap = chunk_size // 2

    # If entire text fits in one chunk, return it directly
    if text_len <= chunk_size:
        return [cleaned_text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If not at the end of text, find the best natural breakpoint
        if end < text_len:
            # Look backwards from 'end' within the overlap zone for natural breaks
            search_start = max(start + step // 2, end - overlap)
            segment = cleaned_text[search_start:end]

            best_break = -1
            # Priority 1: Double newline (paragraph break)
            p_break = segment.rfind("\n\n")
            if p_break != -1:
                best_break = search_start + p_break + 2
            else:
                # Priority 2: Single newline
                nl_break = segment.rfind("\n")
                if nl_break != -1:
                    best_break = search_start + nl_break + 1
                else:
                    # Priority 3: Sentence ending (. , ! , ? followed by space or newline)
                    for sep in (". ", "! ", "? "):
                        s_idx = segment.rfind(sep)
                        if s_idx != -1:
                            best_break = search_start + s_idx + len(sep)
                            break
                    if best_break == -1:
                        # Priority 4: Whitespace
                        sp_idx = segment.rfind(" ")
                        if sp_idx != -1:
                            best_break = search_start + sp_idx + 1

            if best_break > start:
                end = best_break

        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward
        if end >= text_len:
            break

        start = max(start + 1, end - overlap)

    return chunks


class BaseExtractor(abc.ABC):
    """Abstract base class for all file content extractors."""

    @property
    @abc.abstractmethod
    def supported_extensions(self) -> set[str]:
        """Set of supported lowercase file extensions including leading dot."""
        pass

    @abc.abstractmethod
    def extract(self, file_path: Path) -> str | None:
        """Extract textual content or descriptive metadata from the given file.

        Args:
            file_path: Path to the target file.

        Returns:
            Extracted text as a string, or None if extraction is unsupported/failed.
        """
        pass


class TextExtractor(BaseExtractor):
    """Extracts text from plain text and source code files."""

    SUPPORTED_EXTENSIONS = {
        # Documentation & Plain Text
        ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".log",
        # Source Code
        ".py", ".pyi", ".sh", ".bash", ".zsh", ".fish",
        ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".cxx",
        ".rs", ".go", ".java", ".kt", ".kts", ".scala",
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        # Data & Config formats
        ".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml",
        ".csv", ".tsv", ".ini", ".cfg", ".conf",
        ".desktop", ".service", ".sql",
    }

    def __init__(self, max_file_size_bytes: int = 10 * 1024 * 1024):
        """Initialize TextExtractor with a file size threshold (default: 10MB)."""
        self._max_size = max_file_size_bytes

    @property
    def supported_extensions(self) -> set[str]:
        return self.SUPPORTED_EXTENSIONS

    def extract(self, file_path: Path) -> str | None:
        """Read and decode text file contents with UTF-8 and fallback encodings."""
        try:
            size = file_path.stat().st_size
            if size > self._max_size:
                logger.warning(
                    "Skipping file '%s': size (%d bytes) exceeds max limit (%d bytes)",
                    file_path, size, self._max_size,
                )
                return None

            # First attempt: standard UTF-8
            try:
                content = file_path.read_text(encoding="utf-8")
                return content if content.strip() else None
            except UnicodeDecodeError:
                pass

            # Second attempt: UTF-8 with surrogate escape or Latin-1
            try:
                content = file_path.read_text(encoding="latin-1")
                return content if content.strip() else None
            except Exception as e:
                logger.debug("Failed reading file '%s' with fallback encoding: %s", file_path, e)
                return None

        except Exception as e:
            logger.warning("Error reading text file '%s': %s", file_path, e)
            return None


class PdfExtractor(BaseExtractor):
    """Extracts text content from PDF documents using pypdfium2."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    @property
    def supported_extensions(self) -> set[str]:
        return self.SUPPORTED_EXTENSIONS

    def extract(self, file_path: Path) -> str | None:
        """Extract text from all pages in a PDF file."""
        if not PDFIUM_AVAILABLE or pdfium is None:
            logger.warning("pypdfium2 is not installed; skipping PDF extraction for '%s'", file_path)
            return None

        try:
            pdf = pdfium.PdfDocument(str(file_path))
            page_texts: list[str] = []

            for page_index, page in enumerate(pdf):
                try:
                    textpage = page.get_textpage()
                    text = textpage.get_text_range()
                    if text and text.strip():
                        page_texts.append(text.strip())
                except Exception as page_err:
                    logger.debug("Failed extracting page %d of '%s': %s", page_index, file_path, page_err)

            pdf.close()

            if not page_texts:
                return None

            return "\n\n".join(page_texts)

        except Exception as e:
            logger.warning("Failed to extract PDF text from '%s': %s", file_path, e)
            return None


class ImageExtractor(BaseExtractor):
    """Extracts metadata and EXIF description from image files using Pillow."""

    SUPPORTED_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif",
    }

    @property
    def supported_extensions(self) -> set[str]:
        return self.SUPPORTED_EXTENSIONS

    def extract(self, file_path: Path) -> str | None:
        """Generate descriptive metadata text from an image for semantic search."""
        if not PILLOW_AVAILABLE or Image is None:
            logger.warning("Pillow is not installed; skipping image extraction for '%s'", file_path)
            return None

        try:
            with Image.open(str(file_path)) as img:
                fmt = img.format or file_path.suffix.lstrip(".").upper()
                width, height = img.size
                mode = img.mode

                # Clean filename as search context
                name_words = file_path.stem.replace("_", " ").replace("-", " ")
                lines = [
                    f"Image file: {file_path.name}",
                    f"Title: {name_words}",
                    f"Format: {fmt}, Dimensions: {width}x{height} pixels, Color mode: {mode}",
                ]

                # Extract EXIF if available
                exif_data = img.getexif()
                if exif_data and ExifTags is not None:
                    interesting_tags = {
                        "Make": "Camera Make",
                        "Model": "Camera Model",
                        "LensModel": "Lens",
                        "DateTime": "Date",
                        "DateTimeOriginal": "Date Taken",
                        "ImageDescription": "Description",
                        "UserComment": "Comment",
                        "Software": "Software",
                        "Artist": "Artist",
                        "Copyright": "Copyright",
                    }

                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if tag_name in interesting_tags and value:
                            val_str = str(value).strip()
                            if val_str:
                                lines.append(f"{interesting_tags[tag_name]}: {val_str}")

                return "\n".join(lines)

        except Exception as e:
            logger.warning("Failed to extract metadata from image '%s': %s", file_path, e)
            return None


class FileExtractorRegistry:
    """Registry managing extractors mapped to file extensions."""

    def __init__(self):
        self._extractors: list[BaseExtractor] = []
        self._ext_map: dict[str, BaseExtractor] = {}

    def register(self, extractor: BaseExtractor) -> None:
        """Register an extractor for its supported extensions."""
        self._extractors.append(extractor)
        for ext in extractor.supported_extensions:
            self._ext_map[ext.lower()] = extractor

    def get_extractor(self, file_path: Path) -> BaseExtractor | None:
        """Return the registered extractor for a given file path, if any."""
        suffix = file_path.suffix.lower()
        return self._ext_map.get(suffix)

    def can_extract(self, file_path: Path) -> bool:
        """Check whether an extractor is registered for the file extension."""
        return file_path.suffix.lower() in self._ext_map

    def extract(self, file_path: Path) -> str | None:
        """Extract content using the appropriate registered extractor."""
        extractor = self.get_extractor(file_path)
        if extractor is None:
            return None
        return extractor.extract(file_path)

    @property
    def supported_extensions(self) -> set[str]:
        """Return set of all supported file extensions."""
        return set(self._ext_map.keys())


# Global default registry instance
default_registry = FileExtractorRegistry()
default_registry.register(TextExtractor())
default_registry.register(PdfExtractor())
default_registry.register(ImageExtractor())


def extract_file(file_path: Path | str) -> str | None:
    """Convenience function to extract text or metadata from a file."""
    p = Path(file_path)
    return default_registry.extract(p)


def can_extract_file(file_path: Path | str) -> bool:
    """Check if the given file extension can be extracted."""
    p = Path(file_path)
    return default_registry.can_extract(p)
