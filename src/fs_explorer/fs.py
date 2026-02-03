"""
Filesystem utilities for the FsExplorer agent.

This module provides functions for reading, searching, and parsing files
in the filesystem, including support for complex document formats via Docling.
"""

import os
import re
import sys
import glob as glob_module
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import docx2txt
    HAS_DOCX2TXT = True
except ImportError:
    HAS_DOCX2TXT = False


# =============================================================================
# Configuration Constants
# =============================================================================

# Supported document extensions for parsing
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".md"
})

# Preview settings
DEFAULT_PREVIEW_CHARS = 3000  # Characters for single file preview (~2-3 pages)
DEFAULT_SCAN_PREVIEW_CHARS = 1500  # Characters for folder scan preview (~1 page)
MAX_PREVIEW_LINES = 30  # Maximum lines to show in scan results

# Parallel processing settings
DEFAULT_MAX_WORKERS = 8  # Thread pool size for parallel document scanning (increased from 4)

# Cache settings
MAX_CACHE_SIZE_MB = 500  # Maximum cache size in megabytes
MAX_CACHE_ENTRIES = 100  # Maximum number of cached documents

# GPU acceleration settings
GPU_AVAILABLE = False  # Will be auto-detected on first use
GPU_CHECKED = False  # Flag to avoid repeated GPU detection


# =============================================================================
# Document Cache (LRU with size limits)
# =============================================================================

class LRUCache:
    """
    LRU cache with size limits for document caching.

    Automatically evicts least recently used entries when size or count limits are exceeded.
    """

    def __init__(self, max_size_mb: int = MAX_CACHE_SIZE_MB, max_entries: int = MAX_CACHE_ENTRIES):
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.current_size_bytes = 0

    def get(self, key: str) -> Optional[str]:
        """Get item from cache, moving it to end (most recently used)."""
        if key not in self.cache:
            return None
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: str) -> None:
        """Add item to cache, evicting old entries if needed."""
        value_size = sys.getsizeof(value)

        # Remove existing entry if updating
        if key in self.cache:
            old_value = self.cache[key]
            self.current_size_bytes -= sys.getsizeof(old_value)
            del self.cache[key]

        # Evict entries until we have space
        while (self.current_size_bytes + value_size > self.max_size_bytes or
               len(self.cache) >= self.max_entries) and self.cache:
            # Remove least recently used (first item)
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.current_size_bytes -= sys.getsizeof(oldest_value)

        # Add new entry
        self.cache[key] = value
        self.current_size_bytes += value_size

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.current_size_bytes = 0

    def __contains__(self, key: str) -> bool:
        return key in self.cache


# Global LRU cache instance
_DOCUMENT_CACHE = LRUCache()


def clear_document_cache() -> None:
    """Clear the document cache. Useful for testing or memory management."""
    _DOCUMENT_CACHE.clear()


def get_cache_stats() -> dict:
    """Get statistics about the document cache."""
    return {
        "entries": len(_DOCUMENT_CACHE.cache),
        "size_mb": _DOCUMENT_CACHE.current_size_bytes / (1024 * 1024),
        "max_entries": _DOCUMENT_CACHE.max_entries,
        "max_size_mb": _DOCUMENT_CACHE.max_size_bytes / (1024 * 1024),
    }


# =============================================================================
# GPU Detection and Optimization Helpers
# =============================================================================

def _check_gpu_available() -> bool:
    """
    Check if GPU acceleration is available for document processing.

    Returns:
        True if CUDA/GPU is available, False otherwise.
    """
    global GPU_AVAILABLE, GPU_CHECKED

    if GPU_CHECKED:
        return GPU_AVAILABLE

    GPU_CHECKED = True

    try:
        import torch
        GPU_AVAILABLE = torch.cuda.is_available()
        return GPU_AVAILABLE
    except ImportError:
        GPU_AVAILABLE = False
        return False


def _has_text_layer(pdf_path: str) -> bool:
    """
    Check if a PDF has an extractable text layer (doesn't need OCR).

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        True if PDF has text layer, False if it needs OCR.
    """
    if not HAS_PYPDF:
        # If pypdf not available, assume needs OCR (safer default)
        return False

    try:
        reader = pypdf.PdfReader(pdf_path)
        # Check first 3 pages for text content
        pages_to_check = min(3, len(reader.pages))

        for i in range(pages_to_check):
            text = reader.pages[i].extract_text().strip()
            if len(text) > 50:  # Has substantial text content
                return True

        return False
    except Exception:
        # On error, assume needs OCR
        return False


def _fast_parse_docx(file_path: str) -> str:
    """
    Fast DOCX parsing using docx2txt (faster than Docling for simple text extraction).

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Extracted text content.

    Raises:
        Exception: If parsing fails.
    """
    if not HAS_DOCX2TXT:
        raise ImportError("docx2txt not available")

    text = docx2txt.process(file_path)
    return f"# {os.path.basename(file_path)}\n\n{text}"


def _get_optimized_converter(file_path: str) -> DocumentConverter:
    """
    Get an optimized DocumentConverter based on file type and system capabilities.

    Args:
        file_path: Path to the document file.

    Returns:
        Configured DocumentConverter instance.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Configure PDF-specific optimizations
    if ext == ".pdf":
        pipeline_options = PdfPipelineOptions()

        # Auto-detect if PDF has text layer to skip OCR
        has_text = _has_text_layer(file_path)

        pipeline_options.do_ocr = not has_text  # Skip OCR if text layer exists
        pipeline_options.do_table_structure = True

        if has_text:
            # Use faster pypdfium2 backend for text PDFs
            return DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend,
                    )
                }
            )
        else:
            # Use default backend for OCR PDFs (no explicit backend)
            return DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )

    # Default converter for other formats
    return DocumentConverter()


def _get_cached_or_parse(file_path: str) -> str:
    """
    Get document content from cache or parse it with optimizations.

    Uses file modification time in cache key to invalidate stale entries.
    Automatically applies format-specific optimizations:
    - PDFs: Auto-detect text layer to skip OCR
    - DOCX: Use fast parser (docx2txt) if available
    - All: LRU cache with size limits

    Args:
        file_path: Path to the document file.

    Returns:
        The document content as markdown.

    Raises:
        Exception: If the document cannot be parsed.
    """
    abs_path = os.path.abspath(file_path)
    cache_key = f"{abs_path}:{os.path.getmtime(abs_path)}"

    # Check LRU cache
    cached_content = _DOCUMENT_CACHE.get(cache_key)
    if cached_content is not None:
        return cached_content

    # Parse document with optimizations
    ext = os.path.splitext(file_path)[1].lower()

    # Try fast DOCX parser first (3-5x faster than Docling)
    if ext in {".docx", ".doc"} and HAS_DOCX2TXT:
        try:
            content = _fast_parse_docx(file_path)
            _DOCUMENT_CACHE.put(cache_key, content)
            return content
        except Exception:
            # Fall back to Docling if fast parser fails
            pass

    # Use optimized converter for other formats or fallback
    converter = _get_optimized_converter(file_path)
    result = converter.convert(file_path)
    content = result.document.export_to_markdown()

    # Store in LRU cache
    _DOCUMENT_CACHE.put(cache_key, content)

    return content


# =============================================================================
# Directory Operations
# =============================================================================

def describe_dir_content(directory: str) -> str:
    """
    Describe the contents of a directory.
    
    Lists all files and subdirectories in the given directory path.
    
    Args:
        directory: Path to the directory to describe.
    
    Returns:
        A formatted string describing the directory contents,
        or an error message if the directory doesn't exist.
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"
    
    children = os.listdir(directory)
    if not children:
        return f"Directory {directory} is empty"
    
    files = []
    directories = []
    
    for child in children:
        fullpath = os.path.join(directory, child)
        if os.path.isfile(fullpath):
            files.append(fullpath)
        else:
            directories.append(fullpath)
    
    description = f"Content of {directory}\n"
    description += "FILES:\n- " + "\n- ".join(files)
    
    if not directories:
        description += "\nThis folder does not have any sub-folders"
    else:
        description += "\nSUBFOLDERS:\n- " + "\n- ".join(directories)
    
    return description


# =============================================================================
# Basic File Operations
# =============================================================================

def read_file(file_path: str) -> str:
    """
    Read the contents of a text file.
    
    Args:
        file_path: Path to the file to read.
    
    Returns:
        The file contents, or an error message if the file doesn't exist.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"
    
    with open(file_path, "r") as f:
        return f.read()


def grep_file_content(file_path: str, pattern: str) -> str:
    """
    Search for a regex pattern in a file.
    
    Args:
        file_path: Path to the file to search.
        pattern: Regular expression pattern to search for.
    
    Returns:
        A formatted string with matches, "No matches found",
        or an error message if the file doesn't exist.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"
    
    with open(file_path, "r") as f:
        content = f.read()
    
    regex = re.compile(pattern=pattern, flags=re.MULTILINE)
    matches = regex.findall(content)
    
    if matches:
        return f"MATCHES for {pattern} in {file_path}:\n\n- " + "\n- ".join(matches)
    return "No matches found"


def glob_paths(directory: str, pattern: str) -> str:
    """
    Find files matching a glob pattern in a directory.
    
    Args:
        directory: Path to the directory to search in.
        pattern: Glob pattern to match (e.g., "*.txt", "**/*.pdf").
    
    Returns:
        A formatted string with matching paths, "No matches found",
        or an error message if the directory doesn't exist.
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"
    
    # Use pathlib for cleaner path handling
    search_path = Path(directory) / pattern
    matches = glob_module.glob(str(search_path))
    
    if matches:
        return f"MATCHES for {pattern} in {directory}:\n\n- " + "\n- ".join(matches)
    return "No matches found"


# =============================================================================
# Streaming/Chunked Parsing for Large Documents
# =============================================================================

def parse_file_chunked(file_path: str, chunk_size: int = 50000) -> list[str]:
    """
    Parse a large document in chunks for memory-efficient processing.

    This is useful for very large documents (>100 pages) where parsing
    the entire document at once may consume too much memory.

    Args:
        file_path: Path to the document file.
        chunk_size: Number of characters per chunk (default: 50000).

    Returns:
        List of content chunks, or a single-item list with error message.
    """
    try:
        full_content = _get_cached_or_parse(file_path)

        # Split into chunks
        chunks = []
        for i in range(0, len(full_content), chunk_size):
            chunk = full_content[i:i + chunk_size]
            chunks.append(chunk)

        return chunks
    except Exception as e:
        return [f"Error parsing {file_path} in chunks: {e}"]


def get_document_metadata(file_path: str) -> dict:
    """
    Get metadata about a document without parsing full content.

    Args:
        file_path: Path to the document file.

    Returns:
        Dictionary with metadata (size, pages estimate, etc.)
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return {"error": f"No such file: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path)

    metadata = {
        "path": file_path,
        "extension": ext,
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
    }

    # PDF-specific metadata
    if ext == ".pdf" and HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(file_path)
            metadata["pages"] = len(reader.pages)
            metadata["has_text_layer"] = _has_text_layer(file_path)
        except Exception as e:
            metadata["pdf_error"] = str(e)

    return metadata


# =============================================================================
# Document Parsing Operations
# =============================================================================

def preview_file(file_path: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
    """
    Get a quick preview of a document file.
    
    Reads only the first portion of the document content for initial
    relevance assessment before doing a full parse.
    
    Args:
        file_path: Path to the document file.
        max_chars: Maximum characters to return (default: 3000, ~2-3 pages).
    
    Returns:
        A preview of the document content, or an error message.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported file extension: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        full_content = _get_cached_or_parse(file_path)
        preview = full_content[:max_chars]
        
        total_len = len(full_content)
        if total_len > max_chars:
            preview += (
                f"\n\n[... PREVIEW TRUNCATED. Full document has {total_len:,} "
                f"characters. Use parse_file() to read the complete document ...]"
            )
        
        return f"=== PREVIEW of {file_path} ===\n\n{preview}"
    except Exception as e:
        return f"Error previewing {file_path}: {e}"


def parse_file(file_path: str) -> str:
    """
    Parse and return the complete content of a document file.
    
    Use this after preview_file() confirms the document is relevant,
    or when you need to find cross-references to other documents.
    
    Supported formats: PDF, DOCX, DOC, PPTX, XLSX, HTML, MD.
    
    Args:
        file_path: Path to the document file.
    
    Returns:
        The complete document content as markdown, or an error message.
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"No such file: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return (
            f"Unsupported file extension: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        return _get_cached_or_parse(file_path)
    except Exception as e:
        return f"Error parsing {file_path}: {e}"


# =============================================================================
# Parallel Document Scanning
# =============================================================================

def _preview_single_file(file_path: str, preview_chars: int) -> dict:
    """
    Helper to preview a single file for parallel processing.
    
    Args:
        file_path: Path to the document file.
        preview_chars: Number of characters to include in preview.
    
    Returns:
        A dictionary with file info and preview content.
    """
    filename = os.path.basename(file_path)
    try:
        content = _get_cached_or_parse(file_path)
        preview = content[:preview_chars]
        return {
            "file": file_path,
            "filename": filename,
            "preview": preview,
            "total_chars": len(content),
            "status": "success"
        }
    except Exception as e:
        return {
            "file": file_path,
            "filename": filename,
            "preview": "",
            "total_chars": 0,
            "status": f"error: {e}"
        }


def scan_folder(
    directory: str,
    max_workers: int = DEFAULT_MAX_WORKERS,
    preview_chars: int = DEFAULT_SCAN_PREVIEW_CHARS,
) -> str:
    """
    Scan all documents in a folder in parallel and return quick previews.
    
    This is the FIRST step when exploring a folder with multiple documents.
    It efficiently processes all documents at once so you can assess relevance
    before doing deep dives into specific files.
    
    Args:
        directory: Path to the folder to scan.
        max_workers: Number of parallel workers (default: 4).
        preview_chars: Characters to preview per file (default: 1500, ~1 page).
    
    Returns:
        A formatted summary of all documents with their previews.
    """
    if not os.path.exists(directory) or not os.path.isdir(directory):
        return f"No such directory: {directory}"
    
    # Find all supported document files
    doc_files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            ext = os.path.splitext(item)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                doc_files.append(item_path)
    
    if not doc_files:
        return (
            f"No supported documents found in {directory}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    
    # Scan all documents in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_preview_single_file, f, preview_chars): f 
            for f in doc_files
        }
        for future in as_completed(future_to_file):
            results.append(future.result())
    
    # Sort by filename for consistent ordering
    results.sort(key=lambda x: x["filename"])
    
    # Build the summary report
    output = []
    output.append("═══════════════════════════════════════════════════════════════")
    output.append(f"  PARALLEL DOCUMENT SCAN: {directory}")
    output.append(f"  Found {len(results)} documents")
    output.append("═══════════════════════════════════════════════════════════════")
    output.append("")
    
    for i, result in enumerate(results, 1):
        output.append("┌─────────────────────────────────────────────────────────────")
        output.append(f"│ [{i}/{len(results)}] {result['filename']}")
        output.append(f"│ Path: {result['file']}")
        output.append(f"│ Status: {result['status']} | Total size: {result['total_chars']:,} chars")
        output.append("├─────────────────────────────────────────────────────────────")
        
        if result['status'] == 'success' and result['preview']:
            # Indent the preview content
            preview_lines = result['preview'].split('\n')
            for line in preview_lines[:MAX_PREVIEW_LINES]:
                output.append(f"│ {line}")
            if len(preview_lines) > MAX_PREVIEW_LINES:
                output.append("│ ... (preview truncated)")
        else:
            output.append("│ [No preview available]")
        
        output.append("└─────────────────────────────────────────────────────────────")
        output.append("")
    
    output.append("═══════════════════════════════════════════════════════════════")
    output.append("  NEXT STEPS:")
    output.append("  1. Assess which documents are RELEVANT to the user's query")
    output.append("  2. Use parse_file() for DEEP DIVE into relevant documents")
    output.append("  3. Watch for cross-references to other docs (may need backtracking)")
    output.append("═══════════════════════════════════════════════════════════════")
    
    return "\n".join(output)
