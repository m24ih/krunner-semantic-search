"""Storage manager for document metadata and vector embeddings using SQLite and sqlite-vec."""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Optional

import sqlite_vec

logger = logging.getLogger(__name__)

VECTOR_DIMENSION = 768

# Process-wide write lock to ensure thread safety across concurrent indexing workers
_DB_WRITE_LOCK = threading.RLock()


class StorageManager:
    """Manages SQLite storage for files, text chunks, and vector embeddings.

    Uses `sqlite-vec` virtual table (`vec0`) for fast nearest-neighbor (KNN)
    cosine similarity searches.
    """

    def __init__(self, db_path: Path | str) -> None:
        """Initialize and open the vector SQLite database.

        Args:
            db_path: Path to the SQLite database file. Parent directories
                will be created if they do not exist.
        """
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
        )
        self.conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Performance and integrity pragmas
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")

        self.init_db()

    def init_db(self) -> None:
        """Create the necessary tables and indexes if they do not exist."""
        with _DB_WRITE_LOCK, self.conn:
            # Files table: tracks indexed files and their metadata
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    mtime REAL NOT NULL,
                    file_size INTEGER NOT NULL,
                    last_indexed REAL NOT NULL
                );
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_files_path ON files(file_path);
            """)

            # Chunks table: stores document chunks linked to files
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
                );
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
            """)

            # Vector virtual table using sqlite-vec
            # Uses cosine distance metric (distance = 1.0 - cosine_similarity)
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[{VECTOR_DIMENSION}] distance_metric=cosine
                );
            """)

    def upsert_file(self, file_path: str | Path, mtime: float, file_size: int) -> int:
        """Insert or update a file record.

        Args:
            file_path: Absolute or relative path to the file.
            mtime: Last modification timestamp of the file.
            file_size: Size of the file in bytes.

        Returns:
            The integer file_id in the `files` table.
        """
        norm_path = str(Path(file_path).expanduser().resolve())
        now = time.time()

        with _DB_WRITE_LOCK, self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO files (file_path, mtime, file_size, last_indexed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    mtime = excluded.mtime,
                    file_size = excluded.file_size,
                    last_indexed = excluded.last_indexed
                RETURNING id;
                """,
                (norm_path, float(mtime), int(file_size), now),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError(f"Failed to upsert file: {norm_path}")
            return int(row["id"])

    def insert_chunks(
        self,
        file_id: int,
        chunks: list[tuple[int, str, list[float]]],
    ) -> None:
        """Insert text chunks and their 768-dimensional embeddings for a file.

        Existing chunks and embeddings for the given `file_id` are replaced
        to maintain index consistency.

        Args:
            file_id: The ID of the file from `files` table.
            chunks: A list of tuples `(chunk_index, text_content, embedding)`
                where `embedding` is a 768-element list of floats.
        """
        if not chunks:
            return

        with _DB_WRITE_LOCK, self.conn:
            # First remove existing vector entries and chunks for this file
            self.conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)",
                (file_id,),
            )
            self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

            for chunk_index, text_content, embedding in chunks:
                if len(embedding) != VECTOR_DIMENSION:
                    raise ValueError(
                        f"Expected embedding dimension {VECTOR_DIMENSION}, got {len(embedding)}"
                    )

                cursor = self.conn.execute(
                    """
                    INSERT INTO chunks (file_id, chunk_index, text_content)
                    VALUES (?, ?, ?)
                    RETURNING id;
                    """,
                    (file_id, int(chunk_index), text_content),
                )
                chunk_id = int(cursor.fetchone()["id"])

                serialized_vector = sqlite_vec.serialize_float32(embedding)
                self.conn.execute(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialized_vector),
                )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        threshold: float = 0.55,
        deduplicate_files: bool = True,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform a hybrid semantic (vector KNN) and lexical (filename match) search.

        Args:
            query_vector: 768-dimensional float embedding vector of the search query.
            limit: Maximum number of results to return.
            threshold: Minimum cosine similarity threshold (range 0.0 to 1.0).
            deduplicate_files: If True, returns only the best matching chunk per file.
            query_text: Optional raw text query for lexical filename matching.

        Returns:
            A list of dictionary records matching the search query, sorted by
            similarity score descending.
        """
        if len(query_vector) != VECTOR_DIMENSION:
            raise ValueError(
                f"Expected query vector dimension {VECTOR_DIMENSION}, got {len(query_vector)}"
            )

        if limit <= 0:
            return []

        results: list[dict[str, Any]] = []
        seen_files: set[str] = set()

        # 1. Lexical Filename Matching (Instant Exact/Partial File Name Hit)
        if query_text and len(query_text.strip()) >= 2:
            clean_token = query_text.strip().lower()
            try:
                lex_cursor = self.conn.execute(
                    """
                    SELECT
                        f.id AS file_id,
                        f.file_path,
                        f.mtime,
                        f.file_size,
                        COALESCE(c.id, 0) AS chunk_id,
                        COALESCE(c.chunk_index, 0) AS chunk_index,
                        COALESCE(c.text_content, f.file_path) AS text_content
                    FROM files f
                    LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0
                    WHERE LOWER(f.file_path) LIKE ?
                    LIMIT ?
                    """,
                    (f"%{clean_token}%", limit),
                )
                for row in lex_cursor.fetchall():
                    f_path = row["file_path"]
                    if f_path not in seen_files:
                        seen_files.add(f_path)
                        # High relevance score for exact filename hits
                        results.append({
                            "chunk_id": int(row["chunk_id"]),
                            "file_id": int(row["file_id"]),
                            "file_path": f_path,
                            "chunk_index": int(row["chunk_index"]),
                            "text": row["text_content"],
                            "score": 0.95,
                            "distance": 0.05,
                            "mtime": float(row["mtime"]),
                            "file_size": int(row["file_size"]),
                        })
            except Exception as lex_exc:
                logger.warning("Lexical filename matching error: %s", lex_exc)

        # 2. Vector KNN Semantic Search
        fetch_k = max(limit * 4, 30) if deduplicate_files else limit
        serialized_q = sqlite_vec.serialize_float32(query_vector)

        cursor = self.conn.execute(
            """
            SELECT
                v.chunk_id,
                v.distance,
                c.file_id,
                c.chunk_index,
                c.text_content,
                f.file_path,
                f.mtime,
                f.file_size
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
            """,
            (serialized_q, fetch_k),
        )

        for row in cursor.fetchall():
            distance = float(row["distance"])
            similarity = max(0.0, min(1.0, 1.0 - distance))

            if similarity < threshold:
                continue

            file_path = row["file_path"]

            if deduplicate_files:
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)

            results.append({
                "chunk_id": int(row["chunk_id"]),
                "file_id": int(row["file_id"]),
                "file_path": file_path,
                "chunk_index": int(row["chunk_index"]),
                "text": row["text_content"],
                "score": round(similarity, 4),
                "distance": round(distance, 4),
                "mtime": float(row["mtime"]),
                "file_size": int(row["file_size"]),
            })

            if len(results) >= limit * 2:
                break

        # Sort combined results descending by score
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def delete_file(self, file_path: str | Path) -> bool:
        """Delete a file and all its associated chunks and vector embeddings.

        Args:
            file_path: Path of the file to remove.

        Returns:
            True if the file was deleted, False if it was not found.
        """
        norm_path = str(Path(file_path).expanduser().resolve())

        with _DB_WRITE_LOCK, self.conn:
            row = self.conn.execute(
                "SELECT id FROM files WHERE file_path = ?",
                (norm_path,),
            ).fetchone()

            if not row:
                return False

            file_id = int(row["id"])

            # Delete embeddings from vec0 table
            self.conn.execute(
                "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)",
                (file_id,),
            )

            # Deleting from files cascades to chunks
            self.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            return True

    def get_file(self, file_path: str | Path) -> Optional[dict[str, Any]]:
        """Retrieve indexed metadata for a specific file.

        Args:
            file_path: Path of the file.

        Returns:
            Dictionary with file info or None if not indexed.
        """
        norm_path = str(Path(file_path).expanduser().resolve())
        row = self.conn.execute(
            "SELECT id, file_path, mtime, file_size, last_indexed FROM files WHERE file_path = ?",
            (norm_path,),
        ).fetchone()

        if not row:
            return None

        return {
            "id": int(row["id"]),
            "file_path": row["file_path"],
            "mtime": float(row["mtime"]),
            "file_size": int(row["file_size"]),
            "last_indexed": float(row["last_indexed"]),
        }

    def get_all_files(self) -> list[dict[str, Any]]:
        """Retrieve all currently indexed files.

        Returns:
            List of file dictionaries.
        """
        rows = self.conn.execute(
            "SELECT id, file_path, mtime, file_size, last_indexed FROM files ORDER BY id ASC"
        ).fetchall()

        return [
            {
                "id": int(row["id"]),
                "file_path": row["file_path"],
                "mtime": float(row["mtime"]),
                "file_size": int(row["file_size"]),
                "last_indexed": float(row["last_indexed"]),
            }
            for row in rows
        ]

    def count_files(self) -> int:
        """Return the total number of indexed files."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM files").fetchone()
        return int(row["cnt"]) if row else 0

    def count_chunks(self) -> int:
        """Return the total number of indexed chunks."""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM chunks").fetchone()
        return int(row["cnt"]) if row else 0

    def close(self) -> None:
        """Close the database connection."""
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> StorageManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
