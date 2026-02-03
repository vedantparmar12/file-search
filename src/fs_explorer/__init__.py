"""
FsExplorer MCP Server - document search for company documentation.

A Model Context Protocol (MCP) server that pre-indexes documents
and provides instant search access to AI coding assistants like
GitHub Copilot, OpenAI Codex CLI, Claude Code, and Cursor.

No API key required - the client LLM synthesizes answers.
"""

from .mcp_server import mcp, main
from .document_index import DocumentIndex

__all__ = ["mcp", "main", "DocumentIndex"]
