"""
Persistent document index for fixed document collections.

Solves the "same question takes same time every time" problem by:
1. Parsing documents ONCE and storing in SQLite (survives restarts)
2. Caching answers so repeated/similar questions return instantly
3. Pre-building folder summaries for fast relevance matching
4. Detecting folder changes via file hash to auto-reindex only when needed

This is the core optimization that makes the MCP server fast for
company documentation that rarely changes.
"""

import hashlib
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported document extensions (mirrors fs.py)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"
})

# Maximum document content to store per file (4MB chars ~ enough for any doc)
MAX_CONTENT_CHARS = 4_000_000

# Maximum total context to send to LLM in a single call
MAX_LLM_CONTEXT_CHARS = 800_000


class DocumentIndex:
    """
    Persistent document index backed by SQLite.

    Provides:
    - One-time folder indexing (parse all docs, store in DB)
    - Instant document retrieval from cache
    - Keyword-based relevance scoring for query-document matching
    - Answer caching with normalized query matching
    - Folder hash-based change detection for auto-reindex
    """

    def __init__(self, db_path: str = ".cache/doc_index.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create database tables if they don't exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS folders (
                folder_path TEXT PRIMARY KEY,
                folder_hash TEXT NOT NULL,
                indexed_at  REAL NOT NULL,
                doc_count   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                folder_path TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                file_name   TEXT NOT NULL,
                file_hash   TEXT NOT NULL,
                content     TEXT NOT NULL,
                summary     TEXT NOT NULL DEFAULT '',
                char_count  INTEGER DEFAULT 0,
                indexed_at  REAL NOT NULL,
                UNIQUE(folder_path, file_path)
            );

            CREATE TABLE IF NOT EXISTS answer_cache (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                query_normalized TEXT NOT NULL,
                folder_path      TEXT NOT NULL,
                answer           TEXT NOT NULL,
                created_at       REAL NOT NULL,
                hit_count        INTEGER DEFAULT 0,
                UNIQUE(query_normalized, folder_path)
            );

            CREATE INDEX IF NOT EXISTS idx_docs_folder
                ON documents(folder_path);
            CREATE INDEX IF NOT EXISTS idx_cache_query
                ON answer_cache(query_normalized, folder_path);
        """)
        self._conn.commit()
        logger.info(f"Document index initialized at {self.db_path}")

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DocumentIndex not initialized. Call initialize() first.")
        return self._conn

    # -------------------------------------------------------------------------
    # Folder hashing & change detection
    # -------------------------------------------------------------------------

    def _compute_folder_hash(self, folder_path: str) -> str:
        """
        Compute a hash of the folder's document files based on
        filenames, modification times, and sizes.

        If any file changes, the hash changes -> triggers re-index.
        """
        folder = Path(folder_path).resolve()
        files_info = []

        if folder.is_dir():
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    try:
                        stat = f.stat()
                        files_info.append(
                            f"{f.name}:{stat.st_mtime_ns}:{stat.st_size}"
                        )
                    except OSError:
                        continue

        content = "|".join(files_info)
        return hashlib.md5(content.encode()).hexdigest()

    def needs_reindex(self, folder_path: str) -> bool:
        """Check if a folder needs re-indexing (files changed or new folder)."""
        resolved = str(Path(folder_path).resolve())
        current_hash = self._compute_folder_hash(resolved)

        row = self.conn.execute(
            "SELECT folder_hash FROM folders WHERE folder_path = ?",
            (resolved,)
        ).fetchone()

        if row is None:
            return True
        return row[0] != current_hash

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def index_folder(self, folder_path: str) -> dict:
        """
        Parse ALL documents in a folder and store content in the index.

        This is the one-time cost. After indexing, all reads are instant
        from SQLite instead of re-parsing PDFs/DOCX files.

        Returns dict with indexing stats.
        """
        resolved = str(Path(folder_path).resolve())
        folder_hash = self._compute_folder_hash(resolved)
        now = time.time()

        folder = Path(resolved)
        doc_files = [
            f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        indexed = 0
        errors = []

        for doc_file in doc_files:
            try:
                content = self._parse_document(str(doc_file))
                if content.startswith("Error"):
                    errors.append(f"{doc_file.name}: {content}")
                    continue

                # Truncate if extremely large
                if len(content) > MAX_CONTENT_CHARS:
                    content = content[:MAX_CONTENT_CHARS]

                # Summary = first 800 chars (for quick relevance display)
                summary = content[:800].strip()

                self.conn.execute("""
                    INSERT OR REPLACE INTO documents
                    (folder_path, file_path, file_name, file_hash,
                     content, summary, char_count, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    resolved,
                    str(doc_file),
                    doc_file.name,
                    hashlib.md5(content[:1000].encode()).hexdigest(),
                    content,
                    summary,
                    len(content),
                    now,
                ))
                indexed += 1

            except Exception as e:
                errors.append(f"{doc_file.name}: {e}")
                logger.warning(f"Failed to index {doc_file.name}: {e}")

        # Update folder record
        self.conn.execute("""
            INSERT OR REPLACE INTO folders
            (folder_path, folder_hash, indexed_at, doc_count)
            VALUES (?, ?, ?, ?)
        """, (resolved, folder_hash, now, indexed))

        self.conn.commit()

        logger.info(
            f"Indexed {indexed}/{len(doc_files)} documents in {resolved}"
        )

        return {
            "folder": resolved,
            "indexed": indexed,
            "total_files": len(doc_files),
            "errors": errors,
        }

    def ensure_indexed(self, folder_path: str) -> dict:
        """
        Index folder only if needed (files changed or first time).

        Returns stats dict. If already indexed and unchanged, returns
        cached stats without re-parsing anything.
        """
        if self.needs_reindex(folder_path):
            return self.index_folder(folder_path)

        resolved = str(Path(folder_path).resolve())
        row = self.conn.execute(
            "SELECT doc_count, indexed_at FROM folders WHERE folder_path = ?",
            (resolved,)
        ).fetchone()

        return {
            "folder": resolved,
            "indexed": row[0] if row else 0,
            "total_files": row[0] if row else 0,
            "errors": [],
            "from_cache": True,
            "indexed_at": row[1] if row else 0,
        }

    # -------------------------------------------------------------------------
    # Document retrieval
    # -------------------------------------------------------------------------

    def get_document_content(self, file_path: str) -> Optional[str]:
        """
        Get pre-parsed document content from the index.

        Returns None if not indexed (caller should fall back to live parsing).
        """
        resolved = str(Path(file_path).resolve())
        row = self.conn.execute(
            "SELECT content FROM documents WHERE file_path = ?",
            (resolved,)
        ).fetchone()
        return row[0] if row else None

    def get_all_documents(self, folder_path: str) -> list[dict]:
        """Get metadata for all indexed documents in a folder."""
        resolved = str(Path(folder_path).resolve())
        rows = self.conn.execute("""
            SELECT file_path, file_name, summary, char_count, indexed_at
            FROM documents WHERE folder_path = ?
            ORDER BY file_name
        """, (resolved,)).fetchall()

        return [
            {
                "file_path": r[0],
                "file_name": r[1],
                "summary": r[2],
                "char_count": r[3],
                "indexed_at": r[4],
            }
            for r in rows
        ]

    def get_relevant_documents(
        self,
        query: str,
        folder_path: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the most relevant documents for a query using keyword scoring.

        Uses simple but effective keyword matching:
        - Tokenize query into meaningful terms (>2 chars)
        - Score each document by term frequency in content
        - Bonus weight for terms appearing in filename
        - Returns top_k documents sorted by relevance

        For fixed doc sets this is fast and reliable. No embedding model needed.
        """
        resolved = str(Path(folder_path).resolve())

        rows = self.conn.execute("""
            SELECT file_path, file_name, content, summary, char_count
            FROM documents WHERE folder_path = ?
        """, (resolved,)).fetchall()

        if not rows:
            return []

        # Extract meaningful query terms (skip very short words)
        query_terms = [
            t for t in re.split(r'\W+', query.lower()) if len(t) > 2
        ]

        scored_docs = []
        for row in rows:
            file_path, file_name, content, summary, char_count = row
            content_lower = content.lower()
            name_lower = file_name.lower()

            # Score: count query terms found in document content
            score = sum(
                content_lower.count(term) for term in query_terms
            )
            # Bonus x3 for terms in filename
            score += sum(
                3 for term in query_terms if term in name_lower
            )

            scored_docs.append({
                "file_path": file_path,
                "file_name": file_name,
                "content": content,
                "summary": summary,
                "char_count": char_count,
                "relevance_score": score,
            })

        # Sort by relevance descending
        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)

        # If no terms matched any document, return all top_k (let LLM decide)
        return scored_docs[:top_k]

    # -------------------------------------------------------------------------
    # Answer caching
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_query(query: str) -> str:
        """
        Normalize a query for cache matching.

        Handles: case, whitespace, trailing punctuation so that
        "What is the purchase price?" matches "what is the purchase price"
        """
        normalized = query.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.rstrip('?!.')
        return normalized

    def get_cached_answer(self, query: str, folder_path: str) -> Optional[str]:
        """
        Check if we have a cached answer for this query.

        Uses normalized matching so "What is the price?" and
        "what is the price" both hit the same cache entry.
        """
        resolved = str(Path(folder_path).resolve())
        normalized = self._normalize_query(query)

        row = self.conn.execute("""
            SELECT answer, id FROM answer_cache
            WHERE query_normalized = ? AND folder_path = ?
        """, (normalized, resolved)).fetchone()

        if row:
            self.conn.execute(
                "UPDATE answer_cache SET hit_count = hit_count + 1 WHERE id = ?",
                (row[1],)
            )
            self.conn.commit()
            logger.info(f"Answer cache hit for: {query[:60]}...")
            return row[0]

        return None

    def cache_answer(self, query: str, folder_path: str, answer: str) -> None:
        """Cache an answer for future identical/similar queries."""
        resolved = str(Path(folder_path).resolve())
        normalized = self._normalize_query(query)

        self.conn.execute("""
            INSERT OR REPLACE INTO answer_cache
            (query_normalized, folder_path, answer, created_at, hit_count)
            VALUES (?, ?, ?, ?, 0)
        """, (normalized, resolved, answer, time.time()))
        self.conn.commit()
        logger.info(f"Cached answer for: {query[:60]}...")

    def clear_answer_cache(self, folder_path: Optional[str] = None) -> int:
        """
        Clear answer cache. Optionally only for a specific folder.

        Returns number of entries cleared.
        """
        if folder_path:
            resolved = str(Path(folder_path).resolve())
            cursor = self.conn.execute(
                "DELETE FROM answer_cache WHERE folder_path = ?",
                (resolved,)
            )
        else:
            cursor = self.conn.execute("DELETE FROM answer_cache")

        self.conn.commit()
        return cursor.rowcount

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get index statistics."""
        folders = self.conn.execute(
            "SELECT COUNT(*) FROM folders"
        ).fetchone()[0]
        documents = self.conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        cached_answers = self.conn.execute(
            "SELECT COUNT(*) FROM answer_cache"
        ).fetchone()[0]
        total_hits = self.conn.execute(
            "SELECT COALESCE(SUM(hit_count), 0) FROM answer_cache"
        ).fetchone()[0]
        total_chars = self.conn.execute(
            "SELECT COALESCE(SUM(char_count), 0) FROM documents"
        ).fetchone()[0]

        return {
            "indexed_folders": folders,
            "indexed_documents": documents,
            "total_content_chars": total_chars,
            "total_content_mb": round(total_chars / (1024 * 1024), 2),
            "cached_answers": cached_answers,
            "total_cache_hits": total_hits,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_document(file_path: str) -> str:
        """
        Parse a document using the existing fs.py infrastructure.

        Reuses all the optimizations: LRU cache, fast DOCX, smart PDF, etc.
        """
        from .fs import parse_file
        return parse_file(file_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
