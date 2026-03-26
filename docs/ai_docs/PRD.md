Read the AGENTS.md

# PRD: Vertex AI RAG Agent with Google ADK

## 1. Objective

Build a production-grade RAG agent using Google ADK and Vertex AI that can:

- create and manage Vertex AI RAG corpora
- ingest documents from supported Google-managed sources
- retrieve context with Vertex AI RAG
- keep an active corpus in session state
- run locally with `adk web`
- deploy programmatically to Vertex Agent Engine

This PRD is binding for the current implementation and must stay aligned with
`AGENTS.md`.

## 2. Primary Implementation Rules

- Reuse ADK and Vertex AI SDK components before writing new code.
- Do not introduce custom abstractions when ADK or `vertexai.rag` already
  provides the required behavior.
- The implementation must explicitly reuse:
  - `google.adk.agents.Agent`
  - `google.adk.tools.ToolContext`
  - `vertexai.rag`
- The system must remain code-first and environment-driven.
- All corpus operations must go through tools, not the agent reasoning layer.

## 3. System Scope

The system manages the full RAG lifecycle:

`create -> ingest -> query -> inspect -> delete`

Supported user tasks:

- list corpora
- create a corpus
- import data into the active or named corpus
- query a corpus semantically
- inspect corpus contents
- delete a document with confirmation
- delete a corpus with confirmation

## 4. Runtime and Platform Baseline

- Framework: Google ADK
- RAG backend: Vertex AI RAG Engine
- LLM backend: Vertex AI
- Local UI: `adk web`
- Deployment target: Vertex Agent Engine
- Authentication: Google CLI application default credentials or an equivalent
  Google Cloud credential chain supported by Vertex AI

The implementation must initialize Vertex AI before any tool usage.

## 5. Environment Configuration

All runtime and deployment configuration must come from `.env`.

Do not introduce a separate settings layer or hardcoded config constants for
deployment-specific values.

### 5.1 Critical Environment Requirements

- `GOOGLE_GENAI_USE_VERTEXAI` must be truthy so the agent uses Vertex AI for
  both the LLM and RAG.
- `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` must be present so Vertex
  AI can initialize correctly.
- `DEFAULT_EMBEDDING_MODEL` is critical because a bad value creates unusable
  corpora.
- `GOOGLE_CLOUD_STORAGE_BUCKET` is required for deployment flows.
- Validation must run at startup and fail fast with a clear error message.

## 6. Embedding Model Requirements

Corpus creation must use a valid, current Vertex AI publisher embedding model.

### 6.1 Approved Publisher Models

The implementation must support Google publisher embedding models such as:

- `text-embedding-005`
- `text-embedding-004`
- `text-multilingual-embedding-002`

The current default for new corpora is `text-embedding-005`.

### 6.2 Disallowed Retired Models

The implementation must reject retired Gecko embedding models during startup and
before corpus creation. At minimum, reject:

- `textembedding-gecko@001`
- `textembedding-gecko@002`
- `textembedding-gecko@003`
- `textembedding-gecko-multilingual@001`

### 6.3 Corpus Immutability Rule

The embedding model used when a corpus is created is effectively fixed for that
corpus lifecycle.

If the embedding model configuration changes:

- existing corpora are not retrofitted automatically
- a new corpus must be created to use the new embedding model
- the system must not imply that changing `.env` repairs an already-created
  corpus

## 7. Tooling Requirements

Implement the following ADK tools:

- `rag_query`
- `list_corpora`
- `create_corpus`
- `add_data`
- `get_corpus_info`
- `delete_document`
- `delete_corpus`

### 7.1 Shared Tool Rules

- Tools must call `vertexai.rag` directly.
- Tools must return structured responses:
  - `{ "status": ..., "message": ..., "data": ... }`
- Inputs must be validated strictly.
- Errors must be explicit and actionable.
- User-facing responses must not expose internal corpus resource names.

### 7.2 `create_corpus`

Requirements:

- If a corpus with the same display name already exists, reuse it as the active
  corpus instead of creating a duplicate.
- New corpora must be created with `vertexai.rag.RagVectorDbConfig`.
- New corpora must use the configured approved embedding model.
- If the embedding model is retired or invalid, corpus creation must fail
  clearly.

### 7.3 `add_data`

Requirements:

- Support only:
  - Google Cloud Storage paths
  - Google Drive URLs
  - Google Docs URLs
- Normalize supported Cloud Storage HTTPS URLs into `gs://...` form.
- Normalize Google Docs URLs into the equivalent Google Drive file URL before
  ingestion.
- Reject empty paths.
- Reject unsupported source formats.
- Use the selected corpus or the current active corpus.
- Report:
  - imported count
  - failed count
  - skipped count
  - normalized paths
  - partial failures path when Vertex AI provides one

Operational rule:

- A failed corpus import does not necessarily indicate a connectivity issue.
  Corpus configuration, source access, and source format must all be considered.

### 7.4 `rag_query`

Requirements:

- Use the selected corpus or active corpus.
- Use configurable `top_k`.
- Apply configurable distance-threshold filtering.
- Return retrieved contexts with:
  - document display name
  - matched text
  - distance

### 7.5 `get_corpus_info`

Requirements:

- Return corpus summary information.
- Return numbered documents for follow-up operations.
- The numbered list must be suitable for delete disambiguation.

### 7.6 Delete Tools

Requirements:

- `delete_document` must require `confirm=True`.
- `delete_corpus` must require `confirm=True`.
- The implementation must never delete when confirmation is absent or false.

## 8. Ingestion and Attachment Rules

### 8.1 Supported User Inputs

The agent may ingest content only from supported paths that the tool can
normalize:

- `gs://bucket/path`
- `https://storage.googleapis.com/bucket/path`
- `https://<bucket>.storage.googleapis.com/path`
- `https://drive.google.com/...`
- `https://docs.google.com/document/...`

### 8.2 Unsupported Inputs

Direct inline chat attachments are not a supported ingestion source for this
agent.

If the user uploads a chat attachment instead of providing a supported path, the
agent must explain that they need to provide a Google Drive URL, Google Docs
URL, or `gs://` path.

### 8.3 Routing Rule

If the latest user message contains supported ingestion paths and the request
looks like an ingestion action, the system should route directly to `add_data`
instead of asking the user to reformat already-supported paths.

## 9. State Management

Use ADK `ToolContext` and `session.state`.

The system must track:

- `current_corpus`
- `current_corpus_resource`
- `corpus_exists_<slug>`
- `corpus_resource_<slug>`

Rules:

- creating or resolving a corpus must set it as current
- corpus-specific operations may fall back to the current corpus
- internal resource names must stay internal
- state must remain simple and serializable

## 10. Agent Behavior Requirements

Use `google.adk.agents.Agent` for the root agent.

The instruction block must include:

- role
- skills
- decision logic
- tool rules
- safety rules
- communication guidelines
- internal rules

### 10.1 Mandatory Behavior

- knowledge questions must use `rag_query`
- corpus-management actions must use the matching tool
- the agent must determine the target corpus before corpus-specific work
- the agent must mention which corpus it used
- the agent must explain actions after tool calls
- the agent must not invent backend causes that the tool did not return

## 11. Local Execution Requirements

The system must support local execution through:

```bash
adk web
```

Startup flow:

1. load `.env`
2. validate runtime environment
3. initialize Vertex AI
4. construct the root agent
5. serve through ADK Web

## 12. Deployment Requirements

Provide:

- `deployment/deploy.py`
- `deployment/invoke.py`
- `deployment/requirements.txt`
- `deployment/README.md`

Deployment flow:

1. load `.env`
2. validate runtime and deployment environment variables
3. initialize Vertex AI
4. package the agent
5. deploy to Vertex Agent Engine
6. print deployment status and deployed resource details
7. support a documented invocation path for the deployed agent

Rules:

- no manual UI deployment steps
- no hardcoded credentials
- use environment variables only
- include logging and explicit error handling
- package the agent so the deployed runtime can import the same agent module
  path that exists locally
- pass only the application runtime variables that the deployed agent actually
  needs at startup
- deployment is not considered successful unless the deployed resource can be
  invoked after creation

### 12.1 Invocation Requirements

Provide a supported invocation path for the deployed Agent Engine resource.

Requirements:

- the repository must include a documented CLI or SDK-based invocation flow
- the invocation flow must accept the deployed resource name
- if no session id is supplied, the invocation flow must create a new session
  before sending the first message
- the invocation flow must support reusing a returned session id
- the invocation flow must stream or print the deployed agent response in a
  developer-usable form

## 13. Quality and Safety

- follow `AGENTS.md` style and architecture constraints
- keep functions small and focused
- use clear docstrings that explain intent and rationale
- avoid duplicate abstractions
- validate user inputs
- never expose secrets
- never expose internal resource names in user-facing output

## 14. Acceptance Criteria

The final system is acceptable only if all of the following are true:

- local testing works through ADK Web
- a new corpus can be created using a supported embedding model
- a newly created corpus can ingest supported Google Drive, Docs, or GCS data
- the active corpus is tracked in session state
- queries return structured retrieval results
- document and corpus deletion require explicit confirmation
- deployment can be performed with one command
- the deployed Agent Engine resource appears successfully in the target region
- the deployed agent can answer at least one real request through the supported
  invocation path
- retired Gecko embedding models are rejected before they create unusable corpora
