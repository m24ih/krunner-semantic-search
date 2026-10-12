"""Background indexing service and CLI tool for KRunner Semantic Search.

Discovers, extracts, chunks, embeds, and indexes documents into SQLite-Vec.
Monitors filesystem changes based on mtime and file size, and purges deleted files.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from krunner_semantic.config import Config, load_config
from krunner_semantic.embedder import BaseEmbedder, create_embedder
from krunner_semantic.extractors import (
    FileExtractorRegistry,
    chunk_text,
    default_registry,
)
from krunner_semantic.storage import StorageManager

logger = logging.getLogger(__name__)

# Default directory names to ignore during recursive scanning
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".cache",
    ".local",
    ".config",
    "build",
    "dist",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

# Maximum file size to index (50 MB by default)
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


class SemanticIndexer:
    """Manages document discovery, content extraction, embedding, and indexing."""

    def __init__(
        self,
        config: Config | None = None,
        storage: StorageManager | None = None,
        embedder: BaseEmbedder | None = None,
        registry: FileExtractorRegistry | None = None,
    ) -> None:
        """Initialize the semantic indexer.

        Args:
            config: Optional configuration instance. If omitted, loaded from disk.
            storage: Optional StorageManager. If omitted, initialized with config.db_path.
            embedder: Optional BaseEmbedder. If omitted, initialized with config settings.
            registry: Optional FileExtractorRegistry. If omitted, uses default_registry.
        """
        self.config = config or load_config()
        self.storage = storage or StorageManager(self.config.db_path)
        self.embedder = embedder or create_embedder(
            model_path=self.config.model_path,
            device=self.config.effective_device,
            fallback_to_mock=True,
        )
        self.registry = registry or default_registry

        # Compile active exclusion patterns from configuration
        self._compiled_exclude_regexes: list[re.Pattern] = []
        for pat in self.config.active_exclude_patterns:
            try:
                self._compiled_exclude_regexes.append(re.compile(pat, re.IGNORECASE))
            except re.error as err:
                logger.warning("Invalid exclusion regex pattern '%s': %s", pat, err)

    def is_excluded(self, path_or_name: Path | str) -> bool:
        """Check if a file/dir name or path matches any active exclusion regex pattern."""
        str_val = str(path_or_name)
        for regex in self._compiled_exclude_regexes:
            if regex.search(str_val):
                return True
        return False

    def index_file(self, path: Path | str) -> bool:
        """Extract, chunk, embed, and index a single file.

        Args:
            path: Path to the target file.

        Returns:
            True if the file was successfully indexed, False otherwise.
        """
        file_path = Path(path).expanduser().resolve()

        if not file_path.is_file():
            logger.debug("Skipping '%s': not a regular file.", file_path)
            return False

        # Check if the file type is supported
        if not self.registry.can_extract(file_path):
            logger.debug("Skipping '%s': unsupported extension '%s'.", file_path, file_path.suffix)
            return False

        try:
            stat = file_path.stat()
            file_size = stat.st_size
            mtime = stat.st_mtime

            if file_size == 0:
                logger.debug("Skipping empty file: '%s'.", file_path)
                return False

            if file_size > MAX_FILE_SIZE_BYTES:
                logger.warning(
                    "Skipping '%s': file size (%d MB) exceeds limit (%d MB).",
                    file_path,
                    file_size // (1024 * 1024),
                    MAX_FILE_SIZE_BYTES // (1024 * 1024),
                )
                return False

            # Extract text / metadata
            text_content = self.registry.extract(file_path)
            if not text_content or not text_content.strip():
                logger.debug("No extractable text found in '%s'.", file_path)
                # If it was previously indexed, remove it from index
                self.storage.delete_file(file_path)
                return False

            # Segment into sliding window chunks
            chunks = chunk_text(text_content, chunk_size=500, overlap=100)
            if not chunks:
                self.storage.delete_file(file_path)
                return False

            # Generate normalized vector embeddings
            embeddings = self.embedder.embed_batch(chunks)

            # Prepare tuples for storage: (chunk_index, text_content, embedding)
            chunk_tuples = [
                (idx, chunk, emb)
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]

            # Upsert file metadata and insert vector chunks
            file_id = self.storage.upsert_file(file_path, mtime, file_size)
            self.storage.insert_chunks(file_id, chunk_tuples)

            logger.info("Indexed '%s': %d chunks (file_id=%d).", file_path.name, len(chunks), file_id)
            return True

        except Exception as exc:
            logger.error("Failed to index file '%s': %s", file_path, exc, exc_info=True)
            return False

    def index_directory(
        self,
        dir_path: Path | str,
        force: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> dict[str, int]:
        """Recursively scan and index a directory.

        Detects modified or new files, re-indexes if mtime or size changed (or force=True),
        and removes deleted files from the database.

        Args:
            dir_path: Root directory to scan.
            force: If True, re-index all matching files regardless of mtime/size.
            progress_callback: Optional callback(current, total, filename).

        Returns:
            Dictionary containing scan statistics:
            {"scanned": int, "indexed": int, "skipped": int, "deleted": int, "errors": int}
        """
        root_dir = Path(dir_path).expanduser().resolve()
        stats = {"scanned": 0, "indexed": 0, "skipped": 0, "deleted": 0, "errors": 0}

        if not root_dir.is_dir():
            logger.warning("Directory does not exist: '%s'.", root_dir)
            return stats

        logger.info("Scanning directory: '%s' (force=%s)...", root_dir, force)

        # 1. Discover all candidate files
        candidate_paths: list[Path] = []
        for current_root, dirs, files in os.walk(root_dir, topdown=True):
            filtered_dirs = []
            for d in dirs:
                if not self.config.index_hidden_files and d.startswith("."):
                    continue
                d_full = Path(current_root) / d
                if self.is_excluded(d) or self.is_excluded(d_full):
                    continue
                filtered_dirs.append(d)
            dirs[:] = filtered_dirs

            for fname in files:
                if not self.config.index_hidden_files and fname.startswith("."):
                    continue
                fpath = Path(current_root) / fname
                if self.is_excluded(fname) or self.is_excluded(fpath):
                    continue
                if self.registry.can_extract(fpath):
                    candidate_paths.append(fpath)

        total_candidates = len(candidate_paths)
        current_files_on_disk: set[str] = set()

        for idx, fpath in enumerate(candidate_paths):
            resolved_str = str(fpath.resolve())
            current_files_on_disk.add(resolved_str)
            stats["scanned"] += 1

            if progress_callback:
                try:
                    progress_callback(idx + 1, total_candidates, fpath.name)
                except Exception:
                    pass

            try:
                fstat = fpath.stat()
                file_mtime = fstat.st_mtime
                file_size = fstat.st_size

                # Check if file has changed since last index
                if not force:
                    existing = self.storage.get_file(resolved_str)
                    if (
                        existing is not None
                        and existing["mtime"] == file_mtime
                        and existing["file_size"] == file_size
                    ):
                        stats["skipped"] += 1
                        continue

                # Index the modified or new file
                success = self.index_file(fpath)
                if success:
                    stats["indexed"] += 1
                else:
                    stats["errors"] += 1

            except Exception as exc:
                logger.debug("Error processing '%s': %s", fpath, exc)
                stats["errors"] += 1

        # Check for files previously indexed under root_dir that no longer exist
        root_prefix = str(root_dir)
        all_indexed = self.storage.get_all_files()

        for record in all_indexed:
            record_path_str = record["file_path"]
            if record_path_str.startswith(root_prefix):
                if record_path_str not in current_files_on_disk:
                    logger.info("Removing deleted file from index: '%s'", record_path_str)
                    self.storage.delete_file(record_path_str)
                    stats["deleted"] += 1

        logger.info(
            "Finished scanning '%s': scanned=%d, indexed=%d, skipped=%d, deleted=%d, errors=%d",
            root_dir.name,
            stats["scanned"],
            stats["indexed"],
            stats["skipped"],
            stats["deleted"],
            stats["errors"],
        )
        return stats

    def scan_all(
        self,
        force: bool = False,
        progress_callback: Optional[Any] = None,
    ) -> dict[str, int]:
        """Scan and index all directories listed in config.watch_directories.

        Args:
            force: If True, re-index all files unconditionally.
            progress_callback: Optional callback(current, total, filename).

        Returns:
            Cumulative statistics dictionary across all watch directories.
        """
        total_stats = {"scanned": 0, "indexed": 0, "skipped": 0, "deleted": 0, "errors": 0}

        # Count total candidate files first across all watched directories
        all_candidates: list[tuple[Path, Path]] = []
        for watch_dir in self.config.watch_directories:
            resolved_dir = watch_dir.expanduser().resolve()
            if not resolved_dir.is_dir():
                continue
            for current_root, dirs, files in os.walk(resolved_dir, topdown=True):
                filtered_dirs = []
                for d in dirs:
                    if not self.config.index_hidden_files and d.startswith("."):
                        continue
                    d_full = Path(current_root) / d
                    if self.is_excluded(d) or self.is_excluded(d_full):
                        continue
                    filtered_dirs.append(d)
                dirs[:] = filtered_dirs

                for fname in files:
                    if not self.config.index_hidden_files and fname.startswith("."):
                        continue
                    fpath = Path(current_root) / fname
                    if self.is_excluded(fname) or self.is_excluded(fpath):
                        continue
                    if self.registry.can_extract(fpath):
                        all_candidates.append((resolved_dir, fpath))

        total_files = len(all_candidates)
        processed = 0

        for watch_dir in self.config.watch_directories:
            resolved_dir = watch_dir.expanduser().resolve()
            if not resolved_dir.exists():
                logger.debug("Watch directory does not exist: '%s'. Skipping.", resolved_dir)
                continue

            def _callback(curr_in_dir: int, total_in_dir: int, fname: str):
                nonlocal processed
                if progress_callback:
                    progress_callback(processed + curr_in_dir, total_files, fname)

            dir_stats = self.index_directory(resolved_dir, force=force, progress_callback=_callback)
            processed += dir_stats.get("scanned", 0)
            for k in total_stats:
                total_stats[k] += dir_stats.get(k, 0)

        return total_stats

    def get_status(self) -> dict[str, Any]:
        """Return operational and metrics status of the indexer."""
        total_files = self.storage.count_files()
        total_chunks = self.storage.count_chunks()
        db_size_bytes = 0
        if self.config.db_path.exists():
            db_size_bytes = self.config.db_path.stat().st_size

        return {
            "total_files": total_files,
            "total_chunks": total_chunks,
            "db_path": str(self.config.db_path),
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
            "watch_directories": [str(d) for d in self.config.watch_directories],
            "effective_device": self.config.effective_device,
            "embedder_dimension": self.embedder.dimension,
            "is_mock_embedder": getattr(self.embedder, "is_mock", False),
        }


def main() -> int:
    """CLI entrypoint for semantic file indexing."""
    parser = argparse.ArgumentParser(
        prog="krunner-semantic-indexer",
        description="Index local files and documents for KRunner Semantic Search.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and index all configured watch directories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing of all files, even if unchanged.",
    )
    parser.add_argument(
        "--file",
        type=str,
        metavar="PATH",
        help="Index a single file.",
    )
    parser.add_argument(
        "--directory",
        type=str,
        metavar="PATH",
        help="Index a specific directory recursively.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show indexed file and chunk statistics.",
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to custom config.toml file.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    indexer = SemanticIndexer(config=config)

    start_time = time.time()

    # Handle --status
    if args.status:
        status = indexer.get_status()
        print("=== KRunner Semantic Search Index Status ===")
        print(f"  Indexed Files:      {status['total_files']}")
        print(f"  Indexed Chunks:     {status['total_chunks']}")
        print(f"  Database File:      {status['db_path']} ({status['db_size_mb']} MB)")
        print(f"  Compute Device:     {status['effective_device']}")
        print(f"  Embedder Dimension: {status['embedder_dimension']}")
        print(f"  Mock Embedder Mode: {'Yes' if status['is_mock_embedder'] else 'No'}")
        print("  Watch Directories:")
        for wd in status["watch_directories"]:
            print(f"    - {wd}")
        return 0

    # Handle --file
    if args.file:
        target_file = Path(args.file)
        print(f"Indexing single file: {target_file.resolve()}")
        success = indexer.index_file(target_file)
        elapsed = round(time.time() - start_time, 2)
        if success:
            print(f"Successfully indexed file in {elapsed}s.")
            return 0
        else:
            print(f"Failed or skipped indexing file (took {elapsed}s).")
            return 1

    # Handle --directory
    if args.directory:
        target_dir = Path(args.directory)
        print(f"Indexing directory: {target_dir.resolve()} (force={args.force})")
        stats = indexer.index_directory(target_dir, force=args.force)
        elapsed = round(time.time() - start_time, 2)
        print(f"Directory index complete in {elapsed}s:")
        print(f"  Scanned: {stats['scanned']}, Indexed: {stats['indexed']}, Skipped: {stats['skipped']}, Deleted: {stats['deleted']}, Errors: {stats['errors']}")
        return 0

    # Handle --scan (or default if no other action provided)
    if args.scan or len(sys.argv) == 1:
        print(f"Scanning all watch directories (force={args.force})...")
        total_stats = indexer.scan_all(force=args.force)
        elapsed = round(time.time() - start_time, 2)
        print(f"Scan complete in {elapsed}s:")
        print(f"  Scanned: {total_stats['scanned']}, Indexed: {total_stats['indexed']}, Skipped: {total_stats['skipped']}, Deleted: {total_stats['deleted']}, Errors: {total_stats['errors']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
