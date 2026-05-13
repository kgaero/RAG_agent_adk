# Vertex AI RAG Agent with Google ADK

Production-oriented Google ADK agent for managing Vertex AI RAG corpora,
ingesting Google-managed documents, querying retrieved context, and deploying
the agent to Vertex AI Agent Engine.

The implementation is intentionally code-first and reuse-first. The runtime
agent is defined with `google.adk.agents.Agent`, receives runtime state through
`google.adk.tools.ToolContext`, and delegates all corpus operations to
`vertexai.rag`.

## What It Does

- Lists, creates, inspects, and deletes Vertex AI RAG corpora.
- Imports supported Google Cloud Storage, Google Drive, and Google Docs sources.
- Normalizes Google Docs links and Cloud Storage HTTPS links before ingestion.
- Retrieves context chunks from an active or named corpus with configurable
  `top_k` and distance-threshold settings.
- Tracks the active corpus in ADK session state.
- Uses ADK Memory Bank tools to respect remembered corpus preferences.
- Blocks destructive document and corpus deletion unless the latest user
  message exactly matches the required confirmation phrase.
- Sanitizes unsupported direct chat attachments before model calls.
- Runs locally in ADK Web and deploys to Vertex AI Agent Engine.
- Provides deterministic local RAG evaluation reports and repeated-run
  uncertainty plots.

## Architecture

```text
agents/ragagent/
  agent.py        ADK root_agent, instructions, tool wiring, delete guardrail
  tools.py        Vertex AI RAG corpus, ingestion, deletion, and query tools
  bootstrap.py    .env loading, runtime validation, Vertex AI initialization
  callbacks.py    request sanitization and ingestion routing hints

deployment/
  deploy.py       packages the ADK app for Vertex AI Agent Engine
  invoke.py       streams responses from a deployed Agent Engine resource

evals/
  rag_eval.py     deterministic scoring and report generation
  golden_qna.jsonl
  sources/
  reports/

scripts/
  run_rag_eval.py
  run_rag_eval_repeated.py
  shallow_clone.sh
```

The root agent exposes these tools:

- `rag_query`
- `list_corpora`
- `create_corpus`
- `add_data`
- `get_corpus_info`
- `delete_document`
- `delete_corpus`
- ADK `preload_memory`
- ADK `load_memory`

## Requirements

- Python 3.11 or newer recommended.
- Google Cloud project with Vertex AI enabled.
- Application Default Credentials or another credential chain supported by the
  Vertex AI SDK.
- Access to the source documents you want to ingest.
- A staging Cloud Storage bucket for Agent Engine deployment.

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in the required values:

```bash
cp .env.example .env
```

Required runtime variables:

```env
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=<project-id>
GOOGLE_CLOUD_LOCATION=<vertex-location>
RAG_AGENT_LLM_MODEL=<gemini-model-id>
DEFAULT_EMBEDDING_MODEL=text-embedding-005
DEFAULT_CHUNK_SIZE=512
DEFAULT_CHUNK_OVERLAP=100
DEFAULT_TOP_K=5
DEFAULT_DISTANCE_THRESHOLD=0.4
```

Required for deployment:

```env
GOOGLE_CLOUD_STORAGE_BUCKET=<existing-staging-bucket>
```

Optional for invoking a deployed agent:

```env
AGENT_ENGINE_RESOURCE_NAME=projects/.../locations/.../reasoningEngines/...
```

`GOOGLE_GENAI_USE_VERTEXAI` must be truthy. This project uses Vertex AI for
both the Gemini model and Vertex AI RAG. Retired Gecko embedding models are
rejected at startup and before corpus creation.

## Run Locally

Authenticate first:

```bash
gcloud auth application-default login
```

Start the ADK development UI from the repository root:

```bash
adk web agents
```

Then select `ragagent` in the ADK Web agent picker.

Example prompts:

```text
list all corpora
```

```text
create a corpus named productdocs
```

```text
add this document to productdocs:
https://docs.google.com/document/d/<document-id>/edit
```

```text
using productdocs, what does the PRD say about destructive actions?
```

Direct chat file uploads are not ingested by this agent. Provide a supported
`gs://`, Google Drive, or Google Docs path instead.

## Supported Ingestion Sources

`add_data` accepts:

- `gs://bucket/path/file`
- `https://storage.googleapis.com/bucket/path/file`
- `https://bucket.storage.googleapis.com/path/file`
- `https://drive.google.com/...`
- `https://docs.google.com/document/d/...`

Google Docs URLs are converted to equivalent Drive file URLs before ingestion.
Cloud Storage HTTPS URLs are converted to `gs://` paths.

## Destructive Actions

Deletion tools require both `confirm=True` at the tool level and an exact latest
user confirmation phrase:

```text
CONFIRM DELETE DOCUMENT <document-name-or-number>
CONFIRM DELETE CORPUS <corpus-name>
```

Eval sessions set `eval_mode=True`, which blocks destructive operations even if
the confirmation phrase is present.

## Evaluate

The local eval suite is deterministic and does not use a judge LLM. It still
calls the live local ADK agent and Vertex AI RAG, so normal Vertex/Gemini costs
can apply.

Before running, ingest `evals/sources/prd_eval_source.md` or
`docs/ai_docs/PRD.md` into an existing Vertex RAG corpus.

Run one eval pass:

```bash
python scripts/run_rag_eval.py --corpus-name <existing-corpus-display-name>
```

Run repeated evals with aggregate reports and SVG plots:

```bash
python scripts/run_rag_eval_repeated.py \
  --corpus-name <existing-corpus-display-name> \
  --runs 3 \
  --distance-threshold 0.7
```

Reports are written to `evals/reports/`.

## Deploy

Deployment packages `agents/ragagent` as an ADK app and creates a Vertex AI
Agent Engine resource. The script also verifies that Agent Registry contains
the deployed resource before reporting success.

```bash
python deployment/deploy.py
```

Invoke a deployed resource:

```bash
export AGENT_ENGINE_RESOURCE_NAME=projects/.../locations/.../reasoningEngines/...
python deployment/invoke.py "list all corpora"
```

Reuse a session:

```bash
python deployment/invoke.py \
  --session-id=<returned-session-id> \
  "using productdocs, summarize the ingestion requirements"
```

See `deployment/README.md` for deployment metadata and additional invocation
options.

## Test And Quality Gates

Run the unit test suite from the repository root:

```bash
pytest tests/unittests -q
```

Run linting:

```bash
ruff check agents deployment evals scripts tests
```

The tests mock external Vertex AI calls where appropriate and cover:

- environment validation and legacy aliases
- path normalization and ingestion behavior
- corpus state handling
- retrieval payload shape
- destructive-action guardrails
- callback sanitization
- deployment helper output
- deterministic eval scoring and report generation

## Repository Notes

- `AGENTS.md` is the canonical coding-agent guide for this repository.
- `llms-full.txt` is an ADK reference file and must not be modified.
- `refs/` contains read-only external examples and ADK references. Do not import
  from it at runtime and do not edit it.
- Use `scripts/shallow_clone.sh` for clone and submodule hydration workflows.
- Keep new tests under `tests/`.

## Further Reading

- `docs/ai_docs/PRD.md` contains the project requirements that shaped the
  current implementation.
- `evals/README.md` describes the deterministic RAG evaluation workflow.
- `deployment/README.md` documents Agent Engine deployment and invocation.
