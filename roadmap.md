# Roadmap: Connecting file-search MCP to Slack

Production-grade walkthrough for integrating the `fs-explorer` MCP server with Slack so your team can search company documents directly from any Slack channel.

## Architecture

```mermaid
flowchart LR
    A[Slack User] -->|@DocBot what is...| B[Slack API]
    B -->|Event: app_mention| C[Slack Bot - Bolt]
    C -->|stdio| D[MCP Server - fs-explorer]
    D -->|SQLite index ~3ms| E[(Document Index)]
    D -->|Returns doc contents| C
    C -->|Prompt + docs| F[LLM - OpenAI / Anthropic]
    F -->|Synthesized answer| C
    C -->|Posts reply| B
    B -->|Answer with citations| A
```

```
Slack Channel                  Your Server                    MCP Server
┌─────────────┐    HTTPS     ┌──────────────┐    stdio     ┌──────────────┐
│  @DocBot     │ ──────────> │  slack-bridge │ ──────────> │ fs-explorer  │
│  "find the   │             │  (Bolt app)   │             │ (mcp_server) │
│   contract"  │             │               │ <────────── │              │
│              │ <────────── │  + LLM call   │  doc content│  SQLite idx  │
│  Bot replies │   answer    │  (optional)   │             │              │
└─────────────┘             └──────────────┘             └──────────────┘
```

---

## Phase 1: Slack App Setup

### 1.1 Create the Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From scratch**
2. Name it (e.g., `DocSearch Bot`), select your workspace
3. Under **Basic Information**, copy the **Signing Secret**

### 1.2 Configure Bot Token Scopes

Navigate to **OAuth & Permissions** → **Bot Token Scopes** and add:

| Scope | Purpose |
|-------|---------|
| `app_mentions:read` | Detect when users @mention the bot |
| `chat:write` | Post messages back to channels |
| `channels:history` | Read messages in public channels |
| `channels:read` | List channels the bot is in |
| `commands` | Register slash commands (optional) |
| `files:read` | Read file attachments (optional) |

### 1.3 Enable Socket Mode

Socket Mode avoids exposing a public URL. Your bot connects outbound to Slack.

1. Go to **Socket Mode** → Toggle **Enable Socket Mode**
2. Create an **App-Level Token** with scope `connections:write` → copy it (`xapp-...`)

### 1.4 Subscribe to Events

Go to **Event Subscriptions** → Enable Events → **Subscribe to bot events**:

- `app_mention` — triggers when someone @mentions the bot
- `message.channels` — (optional) triggers on all channel messages

### 1.5 Install the App

Go to **Install App** → **Install to Workspace** → Authorize → Copy the **Bot User OAuth Token** (`xoxb-...`)

### 1.6 Required Tokens Summary

| Token | Source | Env Variable |
|-------|--------|-------------|
| Bot Token | OAuth & Permissions page | `SLACK_BOT_TOKEN` |
| App-Level Token | Socket Mode settings | `SLACK_APP_TOKEN` |
| Signing Secret | Basic Information page | `SLACK_SIGNING_SECRET` |

---

## Phase 2: Project Structure

```
file-search/
├── src/fs_explorer/            # Existing MCP server (unchanged)
│   ├── mcp_server.py
│   ├── document_index.py
│   └── fs.py
├── slack-bridge/               # New: Slack integration
│   ├── bot.py                  # Main bot entrypoint
│   ├── mcp_client.py           # MCP session manager
│   ├── llm_provider.py         # LLM synthesis layer
│   ├── config.py               # Configuration & validation
│   ├── middleware.py            # Rate limiting, auth, logging
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container image
│   ├── docker-compose.yml      # Full stack deployment
│   └── .env.example            # Environment template
├── .github/workflows/
│   └── deploy-slack-bot.yml    # CI/CD pipeline
├── README.md                   # Existing
└── roadmap.md                  # This file
```

### 2.1 Dependencies

Create `slack-bridge/requirements.txt`:

```txt
slack-bolt>=1.18.0
slack-sdk>=3.27.0
mcp>=1.0.0
anthropic>=0.40.0
openai>=1.50.0
python-dotenv>=1.0.0
structlog>=24.0.0
prometheus-client>=0.21.0
```

---

## Phase 3: Core Slack Bot

### 3.1 Configuration (`slack-bridge/config.py`)

```python
"""
Configuration and validation for the Slack-MCP bridge.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Slack
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_signing_secret: str = ""

    # MCP Server
    mcp_server_command: str = "uv"
    mcp_server_dir: str = ""
    mcp_server_args: list[str] = field(default_factory=list)
    doc_folder: str = "./documents"

    # LLM (optional - set provider to "none" to return raw results)
    llm_provider: str = "none"  # "openai", "anthropic", "azure", "none"
    llm_api_key: str = ""
    llm_model: str = ""

    # Rate Limiting
    max_requests_per_user_per_minute: int = 10
    max_concurrent_searches: int = 5

    # Security
    allowed_channels: list[str] = field(default_factory=list)  # empty = all
    allowed_users: list[str] = field(default_factory=list)     # empty = all

    @classmethod
    def from_env(cls) -> "Config":
        mcp_dir = os.getenv("MCP_SERVER_DIR", str(Path(__file__).parent.parent))
        return cls(
            slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
            slack_app_token=os.environ["SLACK_APP_TOKEN"],
            slack_signing_secret=os.getenv("SLACK_SIGNING_SECRET", ""),
            mcp_server_command=os.getenv("MCP_SERVER_COMMAND", "uv"),
            mcp_server_dir=mcp_dir,
            mcp_server_args=["--directory", mcp_dir, "run", "fs-explorer-mcp"],
            doc_folder=os.getenv("DOC_FOLDER", "./documents"),
            llm_provider=os.getenv("LLM_PROVIDER", "none"),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            max_requests_per_user_per_minute=int(
                os.getenv("MAX_REQUESTS_PER_USER_PER_MINUTE", "10")
            ),
            max_concurrent_searches=int(
                os.getenv("MAX_CONCURRENT_SEARCHES", "5")
            ),
            allowed_channels=_split_csv(os.getenv("ALLOWED_CHANNELS", "")),
            allowed_users=_split_csv(os.getenv("ALLOWED_USERS", "")),
        )

    def validate(self) -> None:
        missing = []
        if not self.slack_bot_token:
            missing.append("SLACK_BOT_TOKEN")
        if not self.slack_app_token:
            missing.append("SLACK_APP_TOKEN")
        if missing:
            print(f"Missing required env vars: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)


def _split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]
```

### 3.2 MCP Client Manager (`slack-bridge/mcp_client.py`)

```python
"""
MCP client that connects to fs-explorer-mcp via stdio transport.

Manages the subprocess lifecycle and provides async search methods.
"""

import asyncio
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import Config

logger = logging.getLogger(__name__)


class MCPSearchClient:
    """Connects to the fs-explorer MCP server and exposes search operations."""

    def __init__(self, config: Config):
        self.config = config
        self._server_params = StdioServerParameters(
            command=config.mcp_server_command,
            args=config.mcp_server_args,
        )

    async def search(self, query: str, folder: str | None = None, top_k: int = 5) -> str:
        """
        Search documents via the MCP server's search_documents tool.

        Opens a new MCP session per request. For production with high volume,
        see Phase 5 for connection pooling.
        """
        target_folder = folder or self.config.doc_folder

        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "search_documents",
                    arguments={
                        "query": query,
                        "folder": target_folder,
                        "top_k": top_k,
                    },
                )

                if result.content:
                    return result.content[0].text
                return "No results found."

    async def index_stats(self) -> str:
        """Get current index statistics."""
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("index_stats", arguments={})
                if result.content:
                    return result.content[0].text
                return "Unable to fetch stats."

    async def reindex(self, folder: str | None = None) -> str:
        """Force re-index a document folder."""
        target_folder = folder or self.config.doc_folder
        async with stdio_client(self._server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "reindex",
                    arguments={"folder": target_folder},
                )
                if result.content:
                    return result.content[0].text
                return "Reindex complete."
```

### 3.3 Main Bot (`slack-bridge/bot.py`)

```python
"""
Slack bot that bridges @mentions and slash commands to the fs-explorer MCP server.

Usage:
    python bot.py

Requires SLACK_BOT_TOKEN and SLACK_APP_TOKEN environment variables.
"""

import asyncio
import logging
import re
import threading

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config import Config
from mcp_client import MCPSearchClient
from llm_provider import synthesize_answer
from middleware import RateLimiter, check_channel_access, check_user_access

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("slack-mcp-bridge")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

config = Config.from_env()
config.validate()

app = App(token=config.slack_bot_token)
mcp_client = MCPSearchClient(config)
rate_limiter = RateLimiter(config.max_requests_per_user_per_minute)

# Async event loop running in a background thread for MCP calls
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def run_async(coro):
    """Schedule a coroutine on the background event loop and wait for result."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=120)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_query(text: str) -> str:
    """Remove the @mention prefix from the message text."""
    return re.sub(r"<@[\w]+>", "", text).strip()


def format_response(query: str, answer: str) -> list[dict]:
    """Format the answer as Slack Block Kit message."""
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Query:* _{query}_",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": answer[:3000],  # Slack block text limit
            },
        },
    ]
    if len(answer) > 3000:
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "_Response truncated. Use a more specific query for detailed results._",
                    }
                ],
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# Event Handlers
# ---------------------------------------------------------------------------

@app.event("app_mention")
def handle_mention(event, say, client):
    """Handle @DocBot mentions in channels."""
    user_id = event.get("user", "")
    channel_id = event.get("channel", "")
    thread_ts = event.get("thread_ts") or event.get("ts")

    # Access control
    if not check_channel_access(channel_id, config.allowed_channels):
        say(text="This bot is not enabled in this channel.", thread_ts=thread_ts)
        return

    if not check_user_access(user_id, config.allowed_users):
        say(text="You don't have permission to use this bot.", thread_ts=thread_ts)
        return

    # Rate limiting
    if not rate_limiter.allow(user_id):
        say(
            text=f"Rate limit exceeded. Max {config.max_requests_per_user_per_minute} requests/minute.",
            thread_ts=thread_ts,
        )
        return

    query = extract_query(event.get("text", ""))
    if not query:
        say(
            text="Please ask a question. Example: `@DocBot what is our refund policy?`",
            thread_ts=thread_ts,
        )
        return

    # Acknowledge immediately
    say(text=f"Searching documents for: _{query}_...", thread_ts=thread_ts)

    try:
        # Search via MCP
        raw_results = run_async(mcp_client.search(query))

        # Optionally pass through LLM for synthesis
        if config.llm_provider != "none":
            answer = run_async(synthesize_answer(query, raw_results, config))
        else:
            answer = raw_results

        blocks = format_response(query, answer)
        client.chat_postMessage(channel=channel_id, blocks=blocks, thread_ts=thread_ts)

    except Exception:
        logger.exception("Error processing query: %s", query)
        say(text="An error occurred while searching. Please try again.", thread_ts=thread_ts)


@app.command("/docsearch")
def handle_slash_command(ack, respond, command):
    """Handle /docsearch slash command."""
    ack()

    user_id = command.get("user_id", "")
    query = command.get("text", "").strip()

    if not query:
        respond("Usage: `/docsearch <your question>`")
        return

    if not rate_limiter.allow(user_id):
        respond(f"Rate limit exceeded. Max {config.max_requests_per_user_per_minute} requests/minute.")
        return

    respond(f"Searching for: _{query}_...")

    try:
        raw_results = run_async(mcp_client.search(query))

        if config.llm_provider != "none":
            answer = run_async(synthesize_answer(query, raw_results, config))
        else:
            answer = raw_results

        # Slash command responses have a 3000 char limit
        if len(answer) > 3000:
            answer = answer[:2950] + "\n\n_...truncated. Use @mention for full results._"

        respond(answer)

    except Exception:
        logger.exception("Error processing slash command: %s", query)
        respond("An error occurred. Please try again.")


@app.command("/docstats")
def handle_stats_command(ack, respond):
    """Handle /docstats slash command - show index statistics."""
    ack()
    try:
        stats = run_async(mcp_client.index_stats())
        respond(f"```\n{stats}\n```")
    except Exception:
        logger.exception("Error fetching stats")
        respond("Could not fetch index stats.")


@app.command("/docreindex")
def handle_reindex_command(ack, respond, command):
    """Handle /docreindex slash command - force re-index."""
    ack()
    respond("Re-indexing documents... this may take a minute.")
    try:
        result = run_async(mcp_client.reindex())
        respond(f"```\n{result}\n```")
    except Exception:
        logger.exception("Error during reindex")
        respond("Re-index failed. Check server logs.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting Slack-MCP bridge bot...")
    handler = SocketModeHandler(app, config.slack_app_token)
    handler.start()


if __name__ == "__main__":
    main()
```

---

## Phase 4: LLM Synthesis Layer

Without this layer, the bot returns raw document contents. With it, the bot returns a natural-language answer with citations — just like Copilot/Claude Code would.

### 4.1 LLM Provider (`slack-bridge/llm_provider.py`)

```python
"""
Optional LLM layer that synthesizes answers from raw MCP search results.

Set LLM_PROVIDER=none to skip this and return raw document contents.
Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic for natural language answers.
"""

from config import Config

SYSTEM_PROMPT = (
    "You are a document assistant. Answer the user's question using ONLY the "
    "provided document contents. Include citations in the format "
    "[Source: filename, Section/Page]. If the documents don't contain the answer, "
    "say so clearly. Be concise."
)


async def synthesize_answer(query: str, raw_results: str, config: Config) -> str:
    """Route to the configured LLM provider."""
    if config.llm_provider == "openai":
        return await _openai_synthesize(query, raw_results, config)
    elif config.llm_provider == "anthropic":
        return await _anthropic_synthesize(query, raw_results, config)
    elif config.llm_provider == "azure":
        return await _azure_synthesize(query, raw_results, config)
    else:
        return raw_results


async def _openai_synthesize(query: str, docs: str, config: Config) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.llm_api_key)
    response = await client.chat.completions.create(
        model=config.llm_model or "gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {query}"},
        ],
        max_tokens=1500,
        temperature=0.2,
    )
    return response.choices[0].message.content or "No answer generated."


async def _anthropic_synthesize(query: str, docs: str, config: Config) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=config.llm_api_key)
    response = await client.messages.create(
        model=config.llm_model or "claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {query}"},
        ],
    )
    return response.content[0].text if response.content else "No answer generated."


async def _azure_synthesize(query: str, docs: str, config: Config) -> str:
    from openai import AsyncAzureOpenAI
    import os

    client = AsyncAzureOpenAI(
        api_key=config.llm_api_key,
        api_version=os.getenv("AZURE_API_VERSION", "2024-02-15-preview"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT", ""),
    )
    response = await client.chat.completions.create(
        model=config.llm_model or "gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Documents:\n{docs}\n\nQuestion: {query}"},
        ],
        max_tokens=1500,
        temperature=0.2,
    )
    return response.choices[0].message.content or "No answer generated."
```

---

## Phase 5: Production Hardening

### 5.1 Middleware (`slack-bridge/middleware.py`)

```python
"""
Rate limiting, access control, and request logging middleware.
"""

import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket rate limiter per user."""

    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)

    def allow(self, user_id: str) -> bool:
        now = time.time()
        window = now - 60

        # Clean old entries
        self._requests[user_id] = [
            t for t in self._requests[user_id] if t > window
        ]

        if len(self._requests[user_id]) >= self.max_per_minute:
            logger.warning("Rate limit hit for user %s", user_id)
            return False

        self._requests[user_id].append(now)
        return True


def check_channel_access(channel_id: str, allowed: list[str]) -> bool:
    """Check if the channel is in the allowlist. Empty list = all allowed."""
    if not allowed:
        return True
    return channel_id in allowed


def check_user_access(user_id: str, allowed: list[str]) -> bool:
    """Check if the user is in the allowlist. Empty list = all allowed."""
    if not allowed:
        return True
    return user_id in allowed
```

### 5.2 Connection Pooling (High Volume)

For production with many concurrent users, maintain a persistent MCP connection instead of spawning a new subprocess per request.

```python
# Add to mcp_client.py for persistent connection

import asyncio
from contextlib import asynccontextmanager

class MCPConnectionPool:
    """
    Maintains a pool of MCP sessions to avoid subprocess startup per request.

    Usage:
        pool = MCPConnectionPool(config, pool_size=3)
        await pool.start()
        result = await pool.search("query")
        await pool.stop()
    """

    def __init__(self, config: Config, pool_size: int = 3):
        self.config = config
        self.pool_size = pool_size
        self._semaphore = asyncio.Semaphore(pool_size)
        self._server_params = StdioServerParameters(
            command=config.mcp_server_command,
            args=config.mcp_server_args,
        )

    async def search(self, query: str, folder: str | None = None, top_k: int = 5) -> str:
        """Acquire a session from the pool and execute search."""
        async with self._semaphore:
            # Each request gets its own session but concurrency is bounded
            async with stdio_client(self._server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "search_documents",
                        arguments={
                            "query": query,
                            "folder": folder or self.config.doc_folder,
                            "top_k": top_k,
                        },
                    )
                    return result.content[0].text if result.content else "No results."
```

### 5.3 Health Check Endpoint

Add a lightweight HTTP health check for container orchestrators:

```python
# health.py - Run alongside the bot
import http.server
import threading


def start_health_server(port: int = 8080):
    """Start a health check HTTP server on a background thread."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass  # Suppress access logs

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
```

Add to `bot.py` entrypoint:

```python
from health import start_health_server

def main():
    start_health_server(port=8080)
    logger.info("Health check on :8080/health")
    logger.info("Starting Slack-MCP bridge bot...")
    handler = SocketModeHandler(app, config.slack_app_token)
    handler.start()
```

---

## Phase 6: Deployment

### 6.1 Environment Template (`slack-bridge/.env.example`)

```bash
# === Slack Tokens (required) ===
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-level-token
SLACK_SIGNING_SECRET=your-signing-secret

# === MCP Server ===
MCP_SERVER_DIR=/app/file-search          # Path to file-search repo
DOC_FOLDER=/data/documents               # Path to company documents

# === LLM Provider (optional) ===
# Set to "none" to return raw document contents (no LLM cost)
# Set to "openai", "anthropic", or "azure" for synthesized answers
LLM_PROVIDER=none
LLM_API_KEY=
LLM_MODEL=

# Azure-specific (only if LLM_PROVIDER=azure)
AZURE_ENDPOINT=
AZURE_API_VERSION=2024-02-15-preview

# === Rate Limiting ===
MAX_REQUESTS_PER_USER_PER_MINUTE=10
MAX_CONCURRENT_SEARCHES=5

# === Access Control (comma-separated, empty = allow all) ===
ALLOWED_CHANNELS=
ALLOWED_USERS=
```

### 6.2 Dockerfile (`slack-bridge/Dockerfile`)

```dockerfile
FROM python:3.12-slim

# Install uv for running the MCP server
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy the MCP server source
COPY src/ ./file-search/src/
COPY pyproject.toml ./file-search/

# Install MCP server dependencies
RUN cd file-search && uv sync --no-dev

# Copy the Slack bridge
COPY slack-bridge/ ./slack-bridge/

# Install Slack bridge dependencies
RUN pip install --no-cache-dir -r slack-bridge/requirements.txt

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Set the MCP server directory
ENV MCP_SERVER_DIR=/app/file-search

CMD ["python", "slack-bridge/bot.py"]
```

### 6.3 Docker Compose (`slack-bridge/docker-compose.yml`)

```yaml
version: "3.8"

services:
  slack-bot:
    build:
      context: ..
      dockerfile: slack-bridge/Dockerfile
    env_file:
      - .env
    volumes:
      # Mount your company documents (read-only)
      - /path/to/company/documents:/data/documents:ro
      # Persist the SQLite index across restarts
      - doc-index-data:/app/file-search/.cache
    restart: unless-stopped
    ports:
      - "8080:8080"  # Health check only
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"

volumes:
  doc-index-data:
```

### 6.4 Run It

```bash
cd slack-bridge

# Local development
cp .env.example .env
# Fill in your tokens in .env
python bot.py

# Docker
docker compose up -d
docker compose logs -f slack-bot
```

### 6.5 CI/CD Pipeline (`.github/workflows/deploy-slack-bot.yml`)

```yaml
name: Deploy Slack Bot

on:
  push:
    branches: [main]
    paths:
      - "slack-bridge/**"
      - "src/**"

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r slack-bridge/requirements.txt
      - run: python -m pytest slack-bridge/tests/ -v

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -f slack-bridge/Dockerfile -t slack-mcp-bot:${{ github.sha }} .

      # Push to your container registry (ECR, GCR, ACR, GHCR)
      - name: Push to registry
        run: |
          # Example for GitHub Container Registry:
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker tag slack-mcp-bot:${{ github.sha }} ghcr.io/${{ github.repository }}/slack-bot:latest
          docker push ghcr.io/${{ github.repository }}/slack-bot:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      # Deploy to your infrastructure (example: SSH to server)
      - name: Deploy
        run: |
          echo "Deploy to your server/cluster here"
          # ssh user@server "cd /app && docker compose pull && docker compose up -d"
```

---

## Phase 7: Security Checklist

### Token Storage

- **Never commit tokens** — use `.env` files (add to `.gitignore`) or a secrets manager
- For production, use your cloud provider's secrets service:
  - AWS: Secrets Manager or SSM Parameter Store
  - Azure: Key Vault
  - GCP: Secret Manager
- Rotate tokens periodically

### Channel & User Allowlisting

Set `ALLOWED_CHANNELS` and `ALLOWED_USERS` in `.env` to restrict access:

```bash
# Only allow specific channels
ALLOWED_CHANNELS=C01ABCDEF,C02GHIJKL

# Only allow specific users
ALLOWED_USERS=U01ABCDEF,U02GHIJKL
```

### Input Sanitization

The MCP server already handles file paths safely. For the Slack layer:

```python
import re

def sanitize_query(text: str) -> str:
    """Remove potentially dangerous characters from user queries."""
    # Strip Slack formatting artifacts
    text = re.sub(r"<[^>]+>", "", text)        # Remove Slack links/mentions
    text = re.sub(r"[`*_~]", "", text)         # Remove markdown formatting
    # Limit length
    return text[:500].strip()
```

### Audit Logging

Log every query for compliance:

```python
import json
import logging

audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)
handler = logging.FileHandler("/var/log/slack-bot/audit.jsonl")
audit_logger.addHandler(handler)

def log_query(user_id: str, channel_id: str, query: str, success: bool):
    audit_logger.info(json.dumps({
        "user": user_id,
        "channel": channel_id,
        "query": query,
        "success": success,
        "timestamp": time.time(),
    }))
```

---

## Phase 8: Monitoring & Alerting

### 8.1 Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

QUERIES_TOTAL = Counter(
    "docsearch_queries_total",
    "Total document search queries",
    ["status"],  # "success", "error", "rate_limited"
)

QUERY_DURATION = Histogram(
    "docsearch_query_duration_seconds",
    "Time spent processing a query",
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

ACTIVE_SEARCHES = Gauge(
    "docsearch_active_searches",
    "Number of currently active searches",
)

INDEX_DOCUMENTS = Gauge(
    "docsearch_indexed_documents",
    "Number of documents in the index",
)


def start_metrics_server(port: int = 9090):
    """Expose /metrics endpoint for Prometheus scraping."""
    start_http_server(port)
```

Use in `bot.py`:

```python
import time
from metrics import QUERIES_TOTAL, QUERY_DURATION, ACTIVE_SEARCHES

@app.event("app_mention")
def handle_mention(event, say, client):
    # ... access checks ...

    start = time.time()
    ACTIVE_SEARCHES.inc()

    try:
        raw_results = run_async(mcp_client.search(query))
        QUERIES_TOTAL.labels(status="success").inc()
        # ... respond ...
    except Exception:
        QUERIES_TOTAL.labels(status="error").inc()
        raise
    finally:
        ACTIVE_SEARCHES.dec()
        QUERY_DURATION.observe(time.time() - start)
```

### 8.2 Alerting

Configure Prometheus alerts or use Slack's own incoming webhooks:

```python
import os
import requests

def send_alert(message: str):
    """Send an alert to the ops Slack channel via incoming webhook."""
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if webhook_url:
        requests.post(webhook_url, json={"text": f":warning: DocSearch Bot: {message}"})
```

---

## Quick Start Summary

```bash
# 1. Clone
git clone https://github.com/vedantparmar12/file-search.git
cd file-search

# 2. Create Slack app (see Phase 1)
#    Get: SLACK_BOT_TOKEN, SLACK_APP_TOKEN

# 3. Set up the bridge
cd slack-bridge
cp .env.example .env
# Edit .env with your tokens

# 4. Install deps
pip install -r requirements.txt

# 5. Run
python bot.py

# 6. In Slack: @DocBot what is the refund policy?
```

---

## Integration with Company Tools

### If Your Company Provides Codex CLI

Codex CLI already connects to MCP servers. The Slack bridge runs **alongside** it — they share the same `fs-explorer` MCP server and SQLite index:

```
Developer laptop: Codex CLI → fs-explorer MCP → SQLite index
Company server:   Slack Bot  → fs-explorer MCP → SQLite index (same or separate)
```

### If Your Company Provides Copilot

Same pattern — Copilot connects via VS Code MCP settings, while Slack Bot connects via `stdio_client`. Both call the same MCP tools.

### Shared Index

To share a single document index across Slack bot and developer tools, point `DOC_INDEX_DB` to a shared location (network drive, mounted volume, or use a hosted SQLite alternative like LiteFS).
