# LocalGPT - Private Document Intelligence Platform

<div align="center">

<p align="center">
<a href="https://trendshift.io/repositories/2947" target="_blank"><img src="https://trendshift.io/api/badge/repositories/2947" alt="PromtEngineer%2FlocalGPT | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

[![CI](https://github.com/rajmyagentit-del/localgpt-advanced-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/rajmyagentit-del/localgpt-advanced-rag/actions/workflows/ci.yml)
[![GitHub Stars](https://img.shields.io/github/stars/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/PromtEngineer/localGPT?style=flat-square)](https://github.com/PromtEngineer/localGPT/pulls)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg?style=flat-square)](https://www.docker.com/)

<p align="center">
    <a href="https://x.com/engineerrprompt">
      <img src="https://img.shields.io/badge/Follow%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow on X" />
    </a>
    <a href="https://discord.gg/tUDWAFGc">
      <img src="https://img.shields.io/badge/Join%20our%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Join our Discord" />
    </a>
  </p>
</div>

## 🍴 About This Fork

This is my own fork of [PromtEngineer/localGPT](https://github.com/PromtEngineer/localGPT) (MIT License - see [LICENSE](LICENSE)), which I'm using as a portfolio project to demonstrate production-engineering skills on top of a genuinely advanced RAG system. All core retrieval/agent logic below is the original author's work; the improvements in this section are mine.

**Completed so far:**

| # | Improvement | What changed |
|---|---|---|
| 1 | Structured logging | Replaced ~594 `print()` calls across 30 files with leveled, environment-configurable logging (`LOG_LEVEL`, `LOG_FORMAT=json` for machine-parseable output). Also fixed a real bug where 4 different files independently called `logging.basicConfig()`, causing inconsistent/duplicate log output depending on import order. |
| 2 | Centralized configuration | Replaced ~8 scattered `os.getenv()` calls with a single validated `Settings` object (`rag_system/config.py`, Pydantic Settings) - invalid config now fails fast at startup with a clear error instead of a confusing runtime bug. |
| 3 | Real health checks | `/health` now actually verifies Ollama, SQLite, and LanceDB, returning HTTP 503 (not a fake 200) when a required dependency is down - the correct signal for a load balancer or container orchestrator. |
| 4 | Linting/formatting/type-checking | Added `ruff`, `black`, and `mypy` via `pyproject.toml` + pre-commit hooks. Not just decorative - running it surfaced and fixed real issues, including one auto-fix regression (see below). |
| 5 | Unit tests | Added a `pytest` suite (`tests/`) covering the pure logic (cosine similarity, cache key generation, logging formatter, input validation, rate limiter) - 32 tests, all passing. |
| 6 | Input validation | Session/index IDs are now validated as well-formed UUIDs at every route before touching the database; chat messages are validated for type, emptiness, and max length; uploads are capped (413 on oversized requests). |
| 7 | README overhaul | This section, plus corrected environment-variable documentation (the original docs referenced env vars - `DATABASE_PATH`, `VECTOR_DB_PATH` - that didn't match the actual code). |
| 8 | Rate limiting | Per-IP rate limiting on `/chat`-type and upload endpoints (in-memory, fixed-window), returning proper `429` + `Retry-After` headers. |
| 12 | Observability (tracing) | Every query is now traced end-to-end (triage → retrieval → verification) with [OpenTelemetry](https://opentelemetry.io/) - latency and key attributes (query type, doc counts, confidence scores) per step, viewable via console output by default (`LOG_LEVEL`-style env var control: `OTEL_TRACES_EXPORTER=console\|file\|otlp\|none`). Deliberately **not** defaulted to a cloud SaaS backend (e.g. Langfuse Cloud) - that would work against this project's "100% local, private" positioning. Point it at Langfuse/Jaeger/any OTLP backend later via `OTEL_EXPORTER_OTLP_ENDPOINT` without touching the instrumentation code. |
| 13 | Automated RAG evaluation (`eval/`) | Ragas-based suite scoring faithfulness, answer relevancy, context precision, and context recall against a golden question/reference-answer dataset, run through the REAL agent (not mocked). Judged by a **local Ollama model** via the OpenAI-compatible client (`openai` package, no LangChain needed) - consistent with the local-first ethos. See `eval/requirements.txt` for a real, documented dependency pin needed to make `ragas` import cleanly against the current LangChain ecosystem. |
| 9 | FastAPI migration | `backend/app.py` replaces the raw `http.server`-based backend entirely - 21 routes, automatic request validation via Pydantic, interactive docs at `/docs`. Business logic extracted into `backend/chat_service.py` and reused, not rewritten. `backend/server.py` kept for historical reference, marked deprecated. |
| 16 | API versioning | Every route lives under `/v1/` - came essentially free with the FastAPI migration. |
| 10 | JWT auth + session ownership | `backend/auth.py`: stdlib password hashing (no bcrypt dependency), JWT tokens. New `users` table + nullable `user_id` columns (backward-compatible with pre-auth data). Verified with real cross-user access blocking, not just unit tests of the ownership-check function. |
| 17 | Retry/backoff | Scoped retry on Ollama HTTP calls and LanceDB's `get_table()` - narrowly targeted (only `OSError` for LanceDB, since it's a local embedded DB where most failures are real bugs a retry wouldn't fix). |
| 15 | Redis-backed semantic cache | `rag_system/agent/semantic_cache.py` - drop-in `MutableMapping` replacement for the original in-process cache, shared across instances when Redis is configured, falls back automatically when it isn't. |
| 14 | Async indexing queue | RQ (Redis Queue)-based background indexing - `POST /v1/sessions/{id}/index` returns immediately with a job ID instead of blocking for minutes, falls back to the original synchronous behavior if Redis isn't available. |
| 18 | Docker hardening | All three Dockerfiles converted to genuine multi-stage builds (build tooling no longer ships in the runtime image), non-root users, Next.js standalone output mode. Honest limitation: verified via YAML validation and manual review, not an actual `docker compose build` (no Docker daemon in the environment this was built in). |
| 11 | CI/CD (GitHub Actions) | `.github/workflows/ci.yml`: lint, type-check, test, and matrix Docker builds on every PR. Lint/type-check scoped to the paths this fork actually maintains (see `pyproject.toml`) rather than the whole legacy repo - setting this up is what surfaced the two bugs described below. |
| 19 | Complete GraphRAG (Hard tier) | The graph-based retrieval path existed only as scaffolding that had **never once been successfully executed** - see the six real bugs below. `GraphRetriever` extracted into its own dependency-light module (`rag_system/retrieval/graph_retriever.py` - genuinely no torch/transformers dependency, unlike where it used to live), a real `retrieve_structured(start_node, edge_label)` method added with proper fuzzy entity/relationship matching (including SNAKE_CASE-to-natural-language label normalization), and `Agent._run_graph_query()` fixed to actually call it correctly. 23 new tests, all against real graphs (not mocked retrieval logic). Deliberately still **opt-in** (`"graph": {"enabled": False}` by default in `rag_system/main.py`) since knowledge-graph extraction adds real indexing cost (extra LLM calls per chunk) - the roadmap asked to make the feature *work*, not to force it on everyone. |
| 23 | Self-correcting agentic loop (Hard tier) | The `Verifier` already existed and could detect an ungrounded/low-confidence answer, but the code just tagged the answer with a warning and gave up - `max_retries` was accepted as a parameter and never actually used anywhere. Now, on verification failure, `Agent._reformulate_query()` asks the LLM to rewrite the query using the verifier's own stated reasoning for *why* it failed, retries retrieval + generation with the reformulated query, and keeps whichever attempt scored highest across up to `max_retries` tries - not just whichever ran last. 5 new tests invoke the *real* `_run_async()` end-to-end (not an isolated helper), proving reformulation only fires on genuine failure, the best-scoring attempt wins even when a later attempt scores worse, and a retry that retrieves nothing falls back gracefully. |
| 20 | PostgreSQL migration (Hard tier) | `backend/database.py` rewritten from raw `sqlite3` to SQLAlchemy Core (`backend/db_schema.py` defines the dialect-agnostic schema), with **Alembic** for real versioned migrations. Defaults to SQLite (zero-config, matches the original experience); set `DATABASE_URL` to a PostgreSQL connection string for production. All ~20 database methods converted and re-verified against the full existing test suite (175 tests, including 38+ that exercise this exact layer through the real FastAPI app). Found and fixed a real, pre-existing design smell along the way: the file instantiated a global `ChatDatabase()` at *import time*, which silently broke Alembic's schema autogeneration (the database already matched the target schema by the time Alembic compared them, producing an empty migration). Honest limitation: PostgreSQL itself can't be live-tested in this environment (no server available) - verified via real SQLite execution plus PostgreSQL DDL compilation (confirmed `SERIAL`, `ON DELETE CASCADE`, and all constraints translate correctly), not an actual connection to a live Postgres instance. |
| 21 | Multi-tenant isolation (Hard tier) | Found and fixed real, exploitable gaps left over from item 10: `GET /v1/sessions` and `GET /v1/indexes` returned **every user's data with zero filtering** (any authenticated *or anonymous* caller could list everyone's sessions/indexes), and 7 routes (`upload_files_to_session`, `index_session_documents`, `get_session_indexes`, `link_index_to_session`, `upload_files_to_index`, `build_index`, `get_session_documents`) had no ownership check at all. All fixed, secure by default (anonymous callers now see only unowned/legacy data, never other tenants'). Also added real usage tracking - per-user storage quota (`max_storage_bytes_per_user`, default 500MB, enforced *before* writing to disk) and daily query quota (`max_queries_per_day`, default 200) - the difference between "has login" and an actual multi-tenant platform. 24 new tests: cross-tenant list filtering, blocked cross-tenant uploads/builds/links, and both quota types actually triggering (413/429) at the exact configured threshold, not just unit tests of the checking function in isolation. |

**A real bug found along the way:** while running the new `ruff --fix` auto-formatter (item 4) against the codebase, it silently rewrote `Optional[callable]` to `callable | None` in `agent/loop.py` - which looks equivalent but isn't (`callable`, the builtin function, doesn't support the `|` operator the way an actual type does), and broke the module at import time. My new test suite (item 5) caught it immediately on the next run. Fixed by using `typing.Callable` instead. This is exactly why automated fixes get re-tested, not just trusted.

**A second one, from building the tracing (item 12):** OpenTelemetry's global `TracerProvider` can only be set once per process - later attempts are silently ignored. Since `tests/` has no `__init__.py`, a test file doing `from tests.conftest import X` actually re-imports `conftest.py` under a second, disconnected module identity, creating a second provider that never receives real spans. Fixed by moving the test fixture into `conftest.py` itself and letting pytest's fixture injection handle it, instead of an explicit cross-module import.

**A third, from building the eval suite (item 13):** the latest `ragas` release fails to import at all against the current LangChain ecosystem (`ImportError: cannot import name 'ChatVertexAI'` - a symbol removed from a newer `langchain-community` than `ragas` expects). Pinning `langchain-community==0.3.19` (see `eval/requirements.txt`) fixes it - verified in an isolated environment rather than blindly forcing the pin into the main project's dependencies. Separately, my first attempt at wiring Ragas to a local Ollama model used a synchronous `OpenAI` client while calling the async `.ascore()` methods - it failed immediately and clearly ("Cannot use agenerate() with a synchronous client"), fixed by switching to `AsyncOpenAI`.

**Two more, from setting up CI (item 11) - these were the most serious ones:**
1. `backend/database.py`'s `inspect_and_populate_index_metadata()` had a redundant local `from datetime import datetime` deep inside the function, which shadowed the module-level import for the *entire* function scope - causing a real `UnboundLocalError` on every single call, silently swallowed by a bare `except: pass` a few lines later. Caught by `ruff` (rule F823), reproduced with a real test *before* fixing it, confirmed fixed *after*.
2. `rag_system/agent/loop.py`'s LLM-based document router computed the real document overviews into a variable, then never used it - the prompt sent to the LLM hardcoded a generic placeholder ("Invoices, DeepSeek-V3 research papers", clearly leftover demo data) instead. This meant the "intelligent" routing decision was never actually seeing what documents exist, for any project whose content isn't literally invoices and DeepSeek papers. Caught by `ruff` (rule F841, unused variable) - the unused variable was the tell.

Both are covered by permanent regression tests (`tests/test_regressions.py`) and were pre-existing in the original codebase, not introduced by this fork.

**Four more, from actually completing GraphRAG (item 19)** - this was, by a wide margin, the most bug-riddled piece of code in the project, almost certainly because it had never once been successfully executed:
3. `GraphRetriever.retrieve()` called `logger.info(...)` but `logger` was never defined anywhere in the file - guaranteed `NameError` on first use.
4. `Agent._run_graph_query()` passed a `dict` to `GraphRetriever.retrieve()`, which was typed to take a `str` - and even if that were fixed, it accessed `result["details"]["node_id"]` on a return shape that had no `"details"` key at all.
5. Passing a NetworkX `NodeView` directly to `fuzzywuzzy.process.extractOne()` silently matches *nothing*, ever, regardless of input - meaning entity matching had never worked, even in the simple plain-text retrieval path. Needed converting to a plain `list()` first.
6. `MultiVectorRetriever.retrieve()` - the **main vector/hybrid search method used for every non-graph RAG query** - had a redundant local `logger = logging.getLogger(__name__)` that shadowed the module-level logger for the whole method, guaranteeing `UnboundLocalError` on the earlier `logger.info(...)` call a few lines above it, on every single call. Same root-cause pattern as bug 1, caught by the same ruff rule (F823).

Six real, verified bugs, six real fixes, all covered by regression tests. Given how many of these were basic "would crash on first call" errors, it's a strong signal that a decent fraction of this codebase's less-common code paths were written but never actually run - which is exactly why the testing and CI work earlier in this list matters as much as it does.

**Honest limitation on item 19:** the retrieval-side logic (fuzzy entity/relationship matching, structured query handling) is verified with real tests against real graphs. What's *not* yet verified is a full live run - indexing real documents with `GraphExtractor` (needs a running Ollama instance to actually extract entities/relationships) and confirming the agent's LLM-based triage reliably routes appropriate questions to `graph_query` in practice. That's the natural next step once this is run with the full stack live, not a gap in the code itself.

**One more, from the PostgreSQL migration (item 20):** `backend/database.py` instantiated a global `ChatDatabase()` at *import time* - meaning simply importing the module (something Alembic's `env.py` has to do) had the side effect of opening a real database connection and creating the full schema. This silently broke Alembic's autogenerate: by the time it compared the target database against the schema, the import side effect had already made them match, so the "initial schema" migration came out completely empty (`pass`) instead of containing the actual `CREATE TABLE` statements. Fixed by moving that instantiation into the `if __name__ == "__main__":` block where it belongs, and giving the one real caller (the deprecated `backend/server.py`) its own instance instead of relying on a shared one.

**The most consequential one, from building multi-tenant isolation (item 21):** auditing every route against the ownership model revealed that `GET /v1/sessions` and `GET /v1/indexes` had **no filtering whatsoever** - any caller, authenticated or not, got every session and every index from every user in the system. This wasn't a subtle edge case; it was the two most commonly-used list endpoints in the app, returning 100% of everyone's data on every call. Fixed by making both secure by default (an authenticated caller sees only their own data; an anonymous caller sees only unowned/legacy data, never another tenant's). Seven more routes (uploads, index building, linking) had no ownership check at all and would let any authenticated user modify any other user's resources by ID. All fixed and covered by tests that actually create two real users and confirm cross-tenant access is blocked, not just unit tests of the ownership-check function in isolation.

See the "Roadmap" section further down this README for what's planned next.


## 🚀 What is LocalGPT?

LocalGPT is a **fully private, on-premise Document Intelligence platform**. Ask questions, summarise, and uncover insights from your files with state-of-the-art AI—no data ever leaves your machine.

More than a traditional RAG (Retrieval-Augmented Generation) tool, LocalGPT features a **hybrid search engine** that blends semantic similarity, keyword matching, and [Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) for long-context precision. A **smart router** automatically selects between RAG and direct LLM answering for every query, while **contextual enrichment** and sentence-level [Context Pruning](https://huggingface.co/naver/provence-reranker-debertav3-v1) surface only the most relevant content. An independent **verification** pass adds an extra layer of accuracy.

The architecture is **modular and lightweight**—enable only the components you need. With a pure-Python core and minimal dependencies, LocalGPT is simple to deploy, run, and maintain on any infrastructure.The system has minimal dependencies on frameworks and libraries, making it easy to deploy and maintain. The RAG system is pure python and does not require any additional dependencies.

## ▶️ Video
Watch this [video](https://youtu.be/JTbtGH3secI) to get started with LocalGPT. 

| Home | Create Index | Chat |
|------|--------------|------|
| ![](Documentation/images/Home.png) | ![](Documentation/images/Index%20Creation.png) | ![](Documentation/images/Retrieval%20Process.png) |

## ✨ Features

- **Utmost Privacy**: Your data remains on your computer, ensuring 100% security.
- **Versatile Model Support**: Seamlessly integrate a variety of open-source models via Ollama.
- **Diverse Embeddings**: Choose from a range of open-source embeddings.
- **Reuse Your LLM**: Once downloaded, reuse your LLM without the need for repeated downloads.
- **Chat History**: Remembers your previous conversations (in a session).
- **API**: LocalGPT has an API that you can use for building RAG Applications.
- **GPU, CPU, HPU & MPS Support**: Supports multiple platforms out of the box, Chat with your data using `CUDA`, `CPU`, `HPU (Intel® Gaudi®)` or `MPS` and more!

### 📖 Document Processing
- **Multi-format Support**: PDF, DOCX, TXT, Markdown, and more (Currently only PDF is supported)
- **Contextual Enrichment**: Enhanced document understanding with AI-generated context, inspired by [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- **Batch Processing**: Handle multiple documents simultaneously

### 🤖 AI-Powered Chat
- **Natural Language Queries**: Ask questions in plain English
- **Source Attribution**: Every answer includes document references
- **Smart Routing**: Automatically chooses between RAG and direct LLM responses
- **Query Decomposition**: Breaks complex queries into sub-questions for better answers
- **Semantic Caching**: TTL-based caching with similarity matching for faster responses
- **Session-Aware History**: Maintains conversation context across interactions
- **Answer Verification**: Independent verification pass for accuracy
- **Multiple AI Models**: Ollama for inference, HuggingFace for embeddings and reranking


### 🛠️ Developer-Friendly
- **RESTful APIs**: Complete API access for integration
- **Real-time Progress**: Live updates during document processing
- **Flexible Configuration**: Customize models, chunk sizes, and search parameters
- **Extensible Architecture**: Plugin system for custom components

### 🎨 Modern Interface
- **Intuitive Web UI**: Clean, responsive design
- **Session Management**: Organize conversations by topic
- **Index Management**: Easy document collection management
- **Real-time Chat**: Streaming responses for immediate feedback

---

## 🚀 Quick Start

Note: The installation is currently only tested on macOS. 

### Prerequisites
- Python 3.8 or higher (tested with Python 3.11.5)
- Node.js 16+ and npm (tested with Node.js 23.10.0, npm 10.9.2)
- Docker (optional, for containerized deployment)
- 8GB+ RAM (16GB+ recommended)
- Ollama (required for both deployment approaches)

### ***NOTE***
Before this brach is moved to the main branch, please clone this branch for instalation:

```bash
git clone -b localgpt-v2 https://github.com/PromtEngineer/localGPT.git
cd localGPT
```

### Option 1: Docker Deployment 

```bash
# Clone the repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Install Ollama locally (required even for Docker)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:0.6b
ollama pull qwen3:8b

# Start Ollama
ollama serve

# Start with Docker (in a new terminal)
./start-docker.sh

# Access the application
open http://localhost:3000
```

**Docker Management Commands:**
```bash
# Check container status
docker compose ps

# View logs
docker compose logs -f

# Stop containers
./start-docker.sh stop
```

### Option 2: Direct Development (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Install Python dependencies
pip install -r requirements.txt

# Key dependencies installed:
# - torch==2.4.1, transformers==4.51.0 (AI models)
# - lancedb (vector database)
# - rank_bm25, fuzzywuzzy (search algorithms)
# - sentence_transformers, rerankers (embedding/reranking)
# - docling (document processing)
# - colpali-engine (multimodal processing - support coming soon)

# Install Node.js dependencies
npm install

# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:0.6b
ollama pull qwen3:8b
ollama serve

# Start the system (in a new terminal)
python run_system.py

# Access the application
open http://localhost:3000
```

**System Management:**
```bash
# Check system health (comprehensive diagnostics)
python system_health_check.py

# Check service status and health
python run_system.py --health

# Start in production mode
python run_system.py --mode prod

# Skip frontend (backend + RAG API only)
python run_system.py --no-frontend

# View aggregated logs
python run_system.py --logs-only

# Stop all services
python run_system.py --stop
# Or press Ctrl+C in the terminal running python run_system.py
```

**Service Architecture:**
The `run_system.py` launcher manages four key services:
- **Ollama Server** (port 11434): AI model serving
- **RAG API Server** (port 8001): Document processing and retrieval
- **Backend Server** (port 8000): Session management and API endpoints
- **Frontend Server** (port 3000): React/Next.js web interface

### Option 3: Manual Component Startup

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start RAG API
python -m rag_system.api_server

# Terminal 3: Start Backend
cd backend && python server.py

# Terminal 4: Start Frontend
npm run dev

# Access at http://localhost:3000
```

---

### Detailed Installation

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.8 python3-pip nodejs npm docker.io docker-compose
```

**macOS:**
```bash
brew install python@3.8 node npm docker docker-compose
```

**Windows:**
```bash
# Install Python 3.8+, Node.js, and Docker Desktop
# Then use PowerShell or WSL2
```

#### 2. Install AI Models

**Install Ollama (Recommended):**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull qwen3:0.6b          # Fast generation model
ollama pull qwen3:8b            # High-quality generation model
```

#### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Key Configuration Options:**

> All of these are now read through a single validated config object
> (`rag_system/config.py`, built on Pydantic Settings) instead of scattered
> `os.getenv()` calls - invalid values fail fast at startup with a clear
> error instead of surfacing as a confusing bug three requests later.

```env
# LLM Backend
LLM_BACKEND=ollama          # "ollama" or "watsonx"
OLLAMA_HOST=http://localhost:11434

# IBM Watson X (only required if LLM_BACKEND=watsonx)
WATSONX_API_KEY=
WATSONX_PROJECT_ID=
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_GENERATION_MODEL=ibm/granite-13b-chat-v2
WATSONX_ENRICHMENT_MODEL=ibm/granite-8b-japanese

# Storage paths
LANCEDB_PATH=./rag_system/index_store/lancedb
CHAT_DB_PATH=chat_data.db

# Server ports
BACKEND_PORT=8000
RAG_API_PORT=8001
FRONTEND_PORT=3000

# Pipeline mode
RAG_CONFIG_MODE=default     # "default" or "fast"

# Logging (see "Logging" section below)
LOG_LEVEL=INFO               # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=text              # text (human-readable) | json (machine-parseable)

# Optional: HuggingFace token, for gated models
HF_TOKEN=
```

#### 4. Initialize the System

```bash
# Run system health check
python system_health_check.py

# Initialize databases
python -c "from backend.database import ChatDatabase; ChatDatabase().init_database()"

# Test installation
python -c "from rag_system.main import get_agent; print('✅ Installation successful!')"

# Validate complete setup
python run_system.py --health
```

---

## 🎯 Getting Started

### 1. Create Your First Index

An **index** is a collection of processed documents that you can chat with.

#### Using the Web Interface:
1. Open http://localhost:3000
2. Click "Create New Index"
3. Upload your documents (PDF, DOCX, TXT)
4. Configure processing options
5. Click "Build Index"

#### Using Scripts:
```bash
# Simple script approach
./simple_create_index.sh "My Documents" "path/to/document.pdf"

# Interactive script
python create_index_script.py
```

#### Using API:
```bash
# Create index
curl -X POST http://localhost:8000/indexes \
  -H "Content-Type: application/json" \
  -d '{"name": "My Index", "description": "My documents"}'

# Upload documents
curl -X POST http://localhost:8000/indexes/INDEX_ID/upload \
  -F "files=@document.pdf"

# Build index
curl -X POST http://localhost:8000/indexes/INDEX_ID/build
```

### 2. Start Chatting

Once your index is built:

1. **Create a Chat Session**: Click "New Chat" or use an existing session
2. **Select Your Index**: Choose which document collection to query
3. **Ask Questions**: Type natural language questions about your documents
4. **Get Answers**: Receive AI-generated responses with source citations

### 3. Advanced Features

#### Custom Model Configuration
```bash
# Use different models for different tasks
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "title": "High Quality Session",
    "model": "qwen3:8b",
    "embedding_model": "Qwen/Qwen3-Embedding-4B"
  }'
```

#### Batch Document Processing
```bash
# Process multiple documents at once
python demo_batch_indexing.py --config batch_indexing_config.json
```

#### API Integration
```python
import requests

# Chat with your documents via API
response = requests.post('http://localhost:8000/chat', json={
    'query': 'What are the key findings in the research papers?',
    'session_id': 'your-session-id',
    'search_type': 'hybrid',
    'retrieval_k': 20
})

print(response.json()['response'])
```

---

## 🔧 Configuration

### Model Configuration

LocalGPT supports multiple AI model providers with centralized configuration:

#### Ollama Models (Local Inference)
```python
OLLAMA_CONFIG = {
    "host": "http://localhost:11434",
    "generation_model": "qwen3:8b",        # Main text generation
    "enrichment_model": "qwen3:0.6b"       # Lightweight routing/enrichment
}
```

#### External Models (HuggingFace Direct)
```python
EXTERNAL_MODELS = {
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",           # 1024 dimensions
    "reranker_model": "answerdotai/answerai-colbert-small-v1", # ColBERT reranker
    "fallback_reranker": "BAAI/bge-reranker-base"             # Backup reranker
}
```

### Pipeline Configuration

LocalGPT offers two main pipeline configurations:

#### Default Pipeline (Production-Ready)
```python
"default": {
    "description": "Production-ready pipeline with hybrid search, AI reranking, and verification",
    "storage": {
        "lancedb_uri": "./lancedb",
        "text_table_name": "text_pages_v3",
        "bm25_path": "./index_store/bm25"
    },
    "retrieval": {
        "retriever": "multivector",
        "search_type": "hybrid",
        "late_chunking": {"enabled": True},
        "dense": {"enabled": True, "weight": 0.7},
        "bm25": {"enabled": True}
    },
    "reranker": {
        "enabled": True,
        "type": "ai",
        "strategy": "rerankers-lib",
        "model_name": "answerdotai/answerai-colbert-small-v1",
        "top_k": 10
    },
    "query_decomposition": {"enabled": True, "max_sub_queries": 3},
    "verification": {"enabled": True},
    "retrieval_k": 20,
    "contextual_enricher": {"enabled": True, "window_size": 1}
}
```

#### Fast Pipeline (Speed-Optimized)
```python
"fast": {
    "description": "Speed-optimized pipeline with minimal overhead",
    "retrieval": {
        "search_type": "vector_only",
        "late_chunking": {"enabled": False}
    },
    "reranker": {"enabled": False},
    "query_decomposition": {"enabled": False},
    "verification": {"enabled": False},
    "retrieval_k": 10,
    "contextual_enricher": {"enabled": False}
}
```

### Search Configuration

```python
SEARCH_CONFIG = {
    'hybrid': {
        'dense_weight': 0.7,
        'sparse_weight': 0.3,
        'retrieval_k': 20,
        'reranker_top_k': 10
    }
}
```
---

## 🛠️ Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Check Python version
python --version  # Should be 3.8+

# Check dependencies
pip list | grep -E "(torch|transformers|lancedb)"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Model Loading Issues
```bash
# Check Ollama status
ollama list
curl http://localhost:11434/api/tags

# Pull missing models
ollama pull qwen3:0.6b
```

#### Database Issues
```bash
# Check database connectivity
python -c "from backend.database import ChatDatabase; db = ChatDatabase(); print('✅ Database OK')"

# Reset database (WARNING: This deletes all data)
rm backend/chat_data.db
python -c "from backend.database import ChatDatabase; ChatDatabase().init_database()"
```

#### Performance Issues
```bash
# Check system resources
python system_health_check.py

# Monitor memory usage
htop  # or Task Manager on Windows

# Optimize for low-memory systems
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Getting Help

1. **Check Logs**: The system creates structured logs in the `logs/` directory:
   - `logs/system.log`: Main system events and errors
   - `logs/ollama.log`: Ollama server logs
   - `logs/rag-api.log`: RAG API processing logs
   - `logs/backend.log`: Backend server logs
   - `logs/frontend.log`: Frontend build and runtime logs

2. **System Health**: Run comprehensive diagnostics:
   ```bash
   python system_health_check.py  # Full system diagnostics
   python run_system.py --health  # Service status check
   ```

3. **Health Endpoints**: Check individual service health:
   - Backend: `http://localhost:8000/health`
   - RAG API: `http://localhost:8001/health`
   - Ollama: `http://localhost:11434/api/tags`

4. **Documentation**: Check the [Technical Documentation](TECHNICAL_DOCS.md)
5. **GitHub Issues**: Report bugs and request features
6. **Community**: Join our Discord/Slack community

---

## 🔗 API Reference

### Core Endpoints

#### Chat API
```http
# Session-based chat (recommended)
POST /sessions/{session_id}/chat
Content-Type: application/json

{
  "query": "What are the main topics discussed?",
  "search_type": "hybrid",
  "retrieval_k": 20,
  "ai_rerank": true,
  "context_window_size": 5
}

# Legacy chat endpoint
POST /chat
Content-Type: application/json

{
  "query": "What are the main topics discussed?",
  "session_id": "uuid",
  "search_type": "hybrid",
  "retrieval_k": 20
}
```

#### Index Management
```http
# Create index
POST /indexes
Content-Type: application/json
{
  "name": "My Index",
  "description": "Description",
  "config": "default"
}

# Get all indexes
GET /indexes

# Get specific index
GET /indexes/{id}

# Upload documents to index
POST /indexes/{id}/upload
Content-Type: multipart/form-data
files: [file1.pdf, file2.pdf, ...]

# Build index (process uploaded documents)
POST /indexes/{id}/build
Content-Type: application/json
{
  "config_mode": "default",
  "enable_enrich": true,
  "chunk_size": 512
}

# Delete index
DELETE /indexes/{id}
```

#### Session Management
```http
# Create session
POST /sessions
Content-Type: application/json
{
  "title": "My Session",
  "model": "qwen3:0.6b"
}

# Get all sessions
GET /sessions

# Get specific session
GET /sessions/{session_id}

# Get session documents
GET /sessions/{session_id}/documents

# Get session indexes
GET /sessions/{session_id}/indexes

# Link index to session
POST /sessions/{session_id}/indexes/{index_id}

# Delete session
DELETE /sessions/{session_id}

# Rename session
POST /sessions/{session_id}/rename
Content-Type: application/json
{
  "new_title": "Updated Session Name"
}
```

### Advanced Features

#### Query Decomposition
The system can break complex queries into sub-questions for better answers:
```http
POST /sessions/{session_id}/chat
Content-Type: application/json

{
  "query": "Compare the methodologies and analyze their effectiveness",
  "query_decompose": true,
  "compose_sub_answers": true
}
```

#### Answer Verification
Independent verification pass for accuracy using a separate verification model:
```http
POST /sessions/{session_id}/chat
Content-Type: application/json

{
  "query": "What are the key findings?",
  "verify": true
}
```

#### Contextual Enrichment
Document context enrichment during indexing for better understanding:
```bash
# Enable during index building
POST /indexes/{id}/build
{
  "enable_enrich": true,
  "window_size": 2
}
```

#### Late Chunking
Better context preservation by chunking after embedding:
```bash
# Configure in pipeline
"late_chunking": {"enabled": true}
```

#### Streaming Chat
```http
POST /chat/stream
Content-Type: application/json

{
  "query": "Explain the methodology",
  "session_id": "uuid",
  "stream": true
}
```

#### Batch Processing
```bash
# Using the batch indexing script
python demo_batch_indexing.py --config batch_indexing_config.json

# Example batch configuration (batch_indexing_config.json):
{
  "index_name": "Sample Batch Index",
  "index_description": "Example batch index configuration",
  "documents": [
    "./rag_system/documents/invoice_1039.pdf",
    "./rag_system/documents/invoice_1041.pdf"
  ],
  "processing": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "enable_enrich": true,
    "enable_latechunk": true,
    "enable_docling": true,
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "generation_model": "qwen3:0.6b",
    "retrieval_mode": "hybrid",
    "window_size": 2
  }
}
```

```http
# API endpoint for batch processing
POST /batch/index
Content-Type: application/json

{
  "file_paths": ["doc1.pdf", "doc2.pdf"],
  "config": {
    "chunk_size": 512,
    "enable_enrich": true,
    "enable_latechunk": true,
    "enable_docling": true
  }
}
```

For complete API documentation, see [API_REFERENCE.md](API_REFERENCE.md).

---

## 🏗️ Architecture

LocalGPT is built with a modular, scalable architecture:

```mermaid
graph TB
    UI[Web Interface] --> API[Backend API]
    API --> Agent[RAG Agent]
    Agent --> Retrieval[Retrieval Pipeline]
    Agent --> Generation[Generation Pipeline]

    Retrieval --> Vector[Vector Search]
    Retrieval --> BM25[BM25 Search]
    Retrieval --> Rerank[Reranking]

    Vector --> LanceDB[(LanceDB)]
    BM25 --> BM25DB[(BM25 Index)]

    Generation --> Ollama[Ollama Models]
    Generation --> HF[Hugging Face Models]

    API --> SQLite[(SQLite DB)]
```

Overview of the Retrieval Agent

```mermaid
graph TD
    classDef llmcall fill:#e6f3ff,stroke:#007bff;
    classDef pipeline fill:#e6ffe6,stroke:#28a745;
    classDef cache fill:#fff3e0,stroke:#fd7e14;
    classDef logic fill:#f8f9fa,stroke:#6c757d;
    classDef thread stroke-dasharray: 5 5;

    A(Start: Agent.run) --> B_asyncio.run(_run_async);
    B --> C{_run_async};

    C --> C1[Get Chat History];
    C1 --> T1[Build Triage Prompt <br/> Query + Doc Overviews ];
    T1 --> T2["(asyncio.to_thread)<br/>LLM Triage: RAG or LLM_DIRECT?"]; class T2 llmcall,thread;
    T2 --> T3{Decision?};

    T3 -- RAG --> RAG_Path;
    T3 -- LLM_DIRECT --> LLM_Path;

    subgraph RAG Path
        RAG_Path --> R1[Format Query + History];
        R1 --> R2["(asyncio.to_thread)<br/>Generate Query Embedding"]; class R2 pipeline,thread;
        R2 --> R3{{Check Semantic Cache}}; class R3 cache;
        R3 -- Hit --> R_Cache_Hit(Return Cached Result);
        R_Cache_Hit --> R_Hist_Update;
        R3 -- Miss --> R4{Decomposition <br/> Enabled?};

        R4 -- Yes --> R5["(asyncio.to_thread)<br/>Decompose Raw Query"]; class R5 llmcall,thread;
        R5 --> R6{{Run Sub-Queries <br/> Parallel RAG Pipeline}}; class R6 pipeline,thread;
        R6 --> R7[Collect Results & Docs];
        R7 --> R8["(asyncio.to_thread)<br/>Compose Final Answer"]; class R8 llmcall,thread;
        R8 --> V1(RAG Answer);

        R4 -- No --> R9["(asyncio.to_thread)<br/>Run Single Query <br/>(RAG Pipeline)"]; class R9 pipeline,thread;
        R9 --> V1;

        V1 --> V2{{Verification <br/> await verify_async}}; class V2 llmcall;
        V2 --> V3(Final RAG Result);
        V3 --> R_Cache_Store{{Store in Semantic Cache}}; class R_Cache_Store cache;
        R_Cache_Store --> FinalResult;
    end

    subgraph Direct LLM Path
        LLM_Path --> L1[Format Query + History];
        L1 --> L2["(asyncio.to_thread)<br/>Generate Direct LLM Answer <br/> (No RAG)"]; class L2 llmcall,thread;
        L2 --> FinalResult(Final Direct Result);
    end

    FinalResult --> R_Hist_Update(Update Chat History);
    R_Hist_Update --> ZZZ(End: Return Result);
```

---

## 🗄️ Database Migrations

By default, the app uses SQLite with zero configuration - it just works, matching the original project's experience. For production deployments needing real concurrent multi-user access (which SQLite does not safely support), point it at PostgreSQL instead:

```bash
export DATABASE_URL="postgresql://user:password@host:5432/dbname"
```

Schema changes are managed with [Alembic](https://alembic.sqlalchemy.org/):

```bash
pip install -r requirements.txt  # includes sqlalchemy, alembic, psycopg2-binary

# Apply all migrations (creates the schema from scratch on a fresh database)
alembic upgrade head

# After changing backend/db_schema.py, generate a new migration:
alembic revision --autogenerate -m "describe your change"

# Review the generated migration in alembic/versions/ before applying it -
# autogenerate is a strong starting point, not a substitute for reading
# the diff, especially for anything Alembic can't infer (data migrations,
# renames it might see as a drop+add, etc).
alembic upgrade head
```

`alembic/env.py` resolves the target database the same way the app itself does (`DATABASE_URL` if set, otherwise the local SQLite file) - there's no separate connection string to keep in sync.

## 🧪 Testing & Code Quality

This fork adds a real test suite and enforced code quality tooling on top of the original project.

```bash
# Run the test suite
pip install pytest
python -m pytest tests/ -v

# Lint, format-check, and type-check
pip install ruff black mypy
ruff check .
black --check .
mypy rag_system/config.py rag_system/agent/ rag_system/utils/

# One-time setup: run all of the above automatically on every commit
pip install pre-commit
pre-commit install
```

Current coverage focuses on pure, dependency-free logic (fast, no mocking of Ollama/LanceDB required): cosine similarity, semantic-cache key generation, the JSON log formatter, input validation, the rate limiter, and the eval-suite's aggregation/pass-fail logic.

### Running the RAG Evaluation Suite

Unlike the unit tests above, this needs a real, running Ollama instance (it judges answer quality with an actual local LLM - there's no meaningful way to fake that judgment):

```bash
pip install -r eval/requirements.txt
ollama pull qwen3:8b
ollama pull nomic-embed-text

python -m eval.run_eval --json-out eval_results.json --md-out eval_results.md
```

Exits with status code 1 if any metric average falls below its threshold (see `eval/report.py`) - designed to gate a CI pipeline once one exists (see Roadmap). The example questions in `eval/golden_dataset.py` are placeholders; replace them with real question/reference-answer pairs from documents you've actually indexed.

## 🗺️ Roadmap

This fork is being built out as a portfolio project, following a 23-item production-readiness roadmap (Easy → Medium → Hard). Current status:

- ✅ **Easy tier (8/8 complete):** structured logging, centralized config, real health checks, lint/format/type tooling, unit tests, input validation, this README, rate limiting.
- ✅ **Medium tier (10/10 complete):** observability/tracing, automated RAG evaluation, FastAPI migration, API versioning, JWT auth + session ownership, retry/backoff, Redis-backed cache, async indexing queue, Docker hardening, CI/CD.
- 🟡 **Hard tier (4/5):** ✅ GraphRAG completion, ✅ self-correcting agentic loop, ✅ PostgreSQL migration, ✅ multi-tenant isolation. Remaining: a live evaluation/regression dashboard.

---

## 🤝 Contributing

We welcome contributions from developers of all skill levels! LocalGPT is an open-source project that benefits from community involvement.

### 🚀 Quick Start for Contributors

```bash
# Fork and clone the repository
git clone https://github.com/PromtEngineer/localGPT.git
cd localGPT

# Set up development environment
pip install -r requirements.txt
npm install

# Install Ollama and models
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:0.6b qwen3:8b

# Verify setup
python system_health_check.py
python run_system.py --mode dev
```

### 📋 How to Contribute

1. **🐛 Report Bugs**: Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
2. **💡 Request Features**: Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
3. **🔧 Submit Code**: Follow our [development workflow](CONTRIBUTING.md#development-workflow)
4. **📚 Improve Docs**: Help make our documentation better

### 📖 Detailed Guidelines

For comprehensive contributing guidelines, including:
- Development setup and workflow
- Coding standards and best practices
- Testing requirements
- Documentation standards
- Release process

**👉 See our [CONTRIBUTING.md](CONTRIBUTING.md) guide**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. For models, please check their respective licenses.

---

## 📞 Support

- **Documentation**: [Technical Docs](TECHNICAL_DOCS.md)
- **Issues on this fork**: [GitHub Issues](https://github.com/rajmyagentit-del/localgpt-advanced-rag/issues)
- **Original project**: [PromtEngineer/localGPT](https://github.com/PromtEngineer/localGPT) - for upstream discussions, business inquiries, and the original Discord/community links
---

<div align="center">

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PromtEngineer/localGPT&type=Date)](https://star-history.com/#PromtEngineer/localGPT&Date)
