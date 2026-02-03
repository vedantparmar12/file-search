"""
MCP Server for FsExplorer - document search for company documentation.

NO API KEY REQUIRED - the MCP client (Copilot, Codex CLI, Claude Code)
is the LLM. This server handles indexing, caching, and relevance scoring.
The client synthesizes answers from the document contents returned.

Exposes both high-level search and low-level filesystem tools
via the Model Context Protocol (MCP) for use with:
- GitHub Copilot (VS Code)
- OpenAI Codex CLI
- Claude Code
- Cursor / Windsurf / Cline
- Any MCP-compatible client

Performance for fixed document sets:
- First query: indexes all documents once (~30-60s depending on doc count)
- All subsequent queries: ~3ms (read from SQLite index, no re-parsing)
- Document retrieval: ~0.3ms per document
- Relevance scoring: ~0.5ms

Usage:
    # stdio transport (default, for CLI tools)
    uv run fs-explorer-mcp

    # Or directly
    python -m fs_explorer.mcp_server
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from mcp.server.fastmcp import FastMCP

from .document_index import DocumentIndex, SUPPORTED_EXTENSIONS, MAX_LLM_CONTEXT_CHARS
from .fs import (
    scan_folder as _scan_folder,
    parse_file as _parse_file,
    preview_file as _preview_file,
    read_file as _read_file,
    grep_file_content as _grep_file,
    glob_paths as _glob_paths,
)

logger = logging.getLogger(__name__)

# =============================================================================
# MCP Server Instance
# =============================================================================

mcp = FastMCP(
    "fs-explorer",
    instructions=(
        "Document search server for company documentation. "
        "Documents are pre-indexed in SQLite for instant access. "
        "No API key needed - you (the client LLM) synthesize answers.\n\n"
        "Recommended workflow:\n"
        "1. Use 'search_documents' to find relevant docs for a question\n"
        "2. Read the returned document contents\n"
        "3. Synthesize an answer with citations [Source: filename, Section]\n\n"
        "Use individual tools (parse_file, grep_file, etc.) for "
        "fine-grained document access."
    ),
)

# =============================================================================
# Document Index Singleton
# =============================================================================

_INDEX: Optional[DocumentIndex] = None


def _get_index() -> DocumentIndex:
    """Get or create the document index singleton."""
    global _INDEX
    if _INDEX is None:
        db_path = os.getenv("DOC_INDEX_DB", ".cache/doc_index.db")
        _INDEX = DocumentIndex(db_path)
        _INDEX.initialize()
    return _INDEX


# =============================================================================
# HIGH-LEVEL TOOL: Document Search (no API key needed)
# =============================================================================

@mcp.tool()
def search_documents(query: str, folder: str = ".", top_k: int = 5) -> str:
    """
    Find and return relevant document contents for a question.

    Pre-indexes the folder on first call (one-time ~30-60s), then all
    subsequent calls are instant (~3ms). Returns the FULL CONTENT of
    the most relevant documents so you can synthesize an answer.

    No API key required - the MCP client LLM reads the documents
    and generates the answer.

    Args:
        query: Natural language question (e.g. "What is the purchase price?")
        folder: Path to the document folder (default: current directory)
        top_k: Number of most relevant documents to return (default: 5)

    Returns:
        Full content of relevant documents with relevance metadata.
        Read these and synthesize an answer with [Source: filename, Section] citations.
    """
    index = _get_index()

    # 1. Ensure folder is indexed (instant if already done, ~30-60s first time)
    stats = index.ensure_indexed(folder)

    # 2. Find relevant documents using keyword scoring
    relevant_docs = index.get_relevant_documents(query, folder, top_k=top_k)

    if not relevant_docs:
        exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return (
            f"No documents found in '{folder}'. "
            f"Supported formats: {exts}"
        )

    # 3. Build response with full document contents
    total_chars = 0
    sections = []

    sections.append(
        f"Found {len(relevant_docs)} relevant documents for: \"{query}\"\n"
        f"Index: {stats['indexed']} docs"
        + (" (from cache)" if stats.get("from_cache") else " (freshly indexed)")
    )

    for i, doc in enumerate(relevant_docs, 1):
        content = doc["content"]

        # Cap total output to avoid overwhelming context
        remaining = MAX_LLM_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            sections.append(
                f"\n[Remaining {len(relevant_docs) - i + 1} documents omitted "
                f"to fit context window. Use parse_file for individual access.]"
            )
            break
        if len(content) > remaining:
            content = content[:remaining] + "\n\n[... truncated ...]"

        sections.append(
            f"\n{'=' * 60}\n"
            f"[{i}/{len(relevant_docs)}] {doc['file_name']}  "
            f"(relevance: {doc['relevance_score']}, {doc['char_count']:,} chars)\n"
            f"{'=' * 60}\n\n"
            f"{content}"
        )
        total_chars += len(content)

    sections.append(
        f"\n{'=' * 60}\n"
        f"Use the document contents above to answer: \"{query}\"\n"
        f"Include citations: [Source: filename, Section/Page]"
    )

    return "\n".join(sections)


# =============================================================================
# LOW-LEVEL TOOLS: Direct Filesystem Access
# =============================================================================

@mcp.tool()
def scan_folder(directory: str = ".") -> str:
    """
    Parallel scan of all documents in a folder with quick previews.

    Scans PDFs, DOCX, PPTX, XLSX, HTML, and MD files in the folder
    and returns a preview of each (~1 page). Use this to understand
    what documents are available before deep-diving into specific ones.

    Args:
        directory: Path to the folder to scan (default: current directory)
    """
    # Pre-index in background for future speed
    try:
        _get_index().ensure_indexed(directory)
    except Exception:
        pass  # Non-critical, scan_folder works without index

    return _scan_folder(directory)


@mcp.tool()
def parse_file(file_path: str) -> str:
    """
    Extract the complete content of a document file.

    Returns full text from PDF, DOCX, DOC, PPTX, XLSX, HTML, and MD files.
    Uses pre-indexed content when available (instant ~0.3ms), falls back to
    live parsing otherwise.

    Args:
        file_path: Path to the document file
    """
    # Try pre-indexed content first (instant)
    index = _get_index()
    cached = index.get_document_content(file_path)
    if cached:
        return cached

    return _parse_file(file_path)


@mcp.tool()
def preview_file(file_path: str) -> str:
    """
    Quick preview of a document (~first 2-3 pages).

    Faster than full parse. Use for initial relevance assessment
    before committing to a full parse_file call.

    Args:
        file_path: Path to the document file
    """
    return _preview_file(file_path)


@mcp.tool()
def read_file(file_path: str) -> str:
    """
    Read a plain text file (txt, csv, json, yaml, etc.).

    For rich documents (PDF, DOCX), use parse_file instead.

    Args:
        file_path: Path to the text file
    """
    return _read_file(file_path)


@mcp.tool()
def grep_file(file_path: str, pattern: str) -> str:
    """
    Search for a regex pattern in a file.

    Returns all lines matching the pattern. Useful for finding specific
    terms, numbers, dates, or names in documents.

    Args:
        file_path: Path to the file to search
        pattern: Regular expression pattern (e.g. "purchase price|total amount")
    """
    return _grep_file(file_path, pattern)


@mcp.tool()
def glob_files(directory: str, pattern: str) -> str:
    """
    Find files matching a glob pattern in a directory.

    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g. "*.pdf", "**/*.docx", "contracts/*.pdf")
    """
    return _glob_paths(directory, pattern)


# =============================================================================
# INDEX MANAGEMENT TOOLS
# =============================================================================

@mcp.tool()
def reindex(folder: str = ".") -> str:
    """
    Force re-index a document folder.

    Use when documents have been added, removed, or modified.
    Normally the index auto-detects changes via file hashing,
    but this forces a full re-parse of all documents.

    Args:
        folder: Path to the document folder to re-index
    """
    index = _get_index()
    stats = index.index_folder(folder)

    lines = [
        f"Re-indexed: {stats['indexed']}/{stats['total_files']} documents",
        f"Folder: {stats['folder']}",
    ]

    if stats["errors"]:
        lines.append(f"Errors ({len(stats['errors'])}):")
        for err in stats["errors"][:5]:
            lines.append(f"  - {err}")
    else:
        lines.append("No errors.")

    return "\n".join(lines)


@mcp.tool()
def index_stats() -> str:
    """
    Show document index statistics.

    Displays how many folders/documents are indexed and total content size.
    Useful for verifying the index is up to date.
    """
    index = _get_index()
    stats = index.get_stats()

    return (
        "Document Index Statistics\n"
        "========================\n"
        f"  Indexed Folders:    {stats['indexed_folders']}\n"
        f"  Indexed Documents:  {stats['indexed_documents']}\n"
        f"  Total Content:      {stats['total_content_mb']} MB "
        f"({stats['total_content_chars']:,} chars)\n"
        f"  Cached Answers:     {stats['cached_answers']}\n"
        f"  Cache Hits (total): {stats['total_cache_hits']}"
    )


# =============================================================================
# RESOURCES: Browsable Document Listings
# =============================================================================

@mcp.resource("index://stats")
def resource_index_stats() -> str:
    """Current document index statistics."""
    return index_stats()


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
