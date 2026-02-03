# file-search

MCP server for AI-powered company document search. Connects to GitHub Copilot, OpenAI Codex CLI, Claude Code, Cursor, and any MCP-compatible client.

**No API key required** - the client LLM (Copilot/Codex) synthesizes answers from pre-indexed document contents.

## How It Works

```
You ask Copilot: "What is the purchase price in the contracts?"
    |
    v
Copilot calls MCP tool: search_documents("purchase price", "./contracts")
    |
    v
MCP Server: finds relevant docs from SQLite index (~3ms)
    |
    v
Returns full document contents to Copilot
    |
    v
Copilot reads docs and answers with citations
```

### Performance (fixed document sets)

| Scenario | Time |
|---|---|
| First query (indexes folder) | ~30-60s one-time |
| Subsequent queries | **~3ms** |
| Document retrieval | **~0.3ms** |
| Relevance scoring | **~0.5ms** |

Documents are parsed once and stored in SQLite. No re-parsing on every query.

## Setup

```bash
# Clone
git clone https://github.com/vedantparmar12/file-search.git
cd file-search

# Install
uv sync

# Run (stdio transport for MCP clients)
uv run fs-explorer-mcp
```

## Connect to Your AI Tool

### GitHub Copilot (VS Code)

Add to VS Code `settings.json`:

```json
{
  "github.copilot.chat.mcp.servers": {
    "fs-explorer": {
      "command": "uv",
      "args": ["--directory", "/path/to/file-search", "run", "fs-explorer-mcp"],
      "env": {
        "DOC_INDEX_DB": ".cache/doc_index.db"
      }
    }
  }
}
```

### OpenAI Codex CLI

Add to your Codex config:

```json
{
  "mcpServers": {
    "fs-explorer": {
      "command": "uv",
      "args": ["--directory", "/path/to/file-search", "run", "fs-explorer-mcp"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add fs-explorer -- uv --directory /path/to/file-search run fs-explorer-mcp
```

### Cursor

Add to Cursor MCP settings:

```json
{
  "fs-explorer": {
    "command": "uv",
    "args": ["--directory", "/path/to/file-search", "run", "fs-explorer-mcp"]
  }
}
```

## MCP Tools

| Tool | Purpose |
|---|---|
| `search_documents` | Find relevant docs for a question and return full content |
| `scan_folder` | Preview all documents in a folder |
| `parse_file` | Full document content (instant from index) |
| `preview_file` | Quick preview (~2-3 pages) |
| `read_file` | Read plain text files |
| `grep_file` | Regex search in a file |
| `glob_files` | Find files by glob pattern |
| `reindex` | Force re-index when documents change |
| `index_stats` | Show index statistics |

### Supported Document Formats

PDF, DOCX, DOC, PPTX, XLSX, HTML, Markdown

## Architecture

```
src/fs_explorer/
  mcp_server.py        # MCP server with 9 tools (FastMCP)
  document_index.py    # SQLite-backed persistent document index
  fs.py                # Document parsing (Docling) + filesystem tools
```

### How the Index Solves the Speed Problem

Without index: every query re-parses all PDFs/DOCX files from scratch (~5-30s per query).

With index:
1. **First query**: parses all documents once, stores in SQLite
2. **All subsequent queries**: reads from SQLite (~0.3ms per document)
3. **Auto-detects changes**: folder hash invalidates index when files change
4. **`reindex` tool**: force refresh when needed

## Configuration

Copy `.env.example` to `.env` to customize:

```bash
# Path to the document index database
DOC_INDEX_DB=.cache/doc_index.db
```

No other configuration needed. No API keys required.
