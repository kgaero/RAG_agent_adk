Read the AGENTS.md

build a production-grade RAG AI agent system using Google ADK and Google Vertex AI RAG engine.

Follow ALL constraints defined in AGENTS.md. Do not override them.

--------------------------------------------------
1. PRIMARY IMPLEMENTATION PRINCIPLES
--------------------------------------------------

- Strictly follow AGENTS.md as the source of truth
- Reuse ADK components wherever possible
- Do NOT create custom abstractions if ADK provides them
- Explicitly use:
  - google.adk.agents.Agent
  - google.adk.tools
  - vertexai.rag APIs

Before writing code:
- Identify which ADK components are reused
- Avoid duplicating functionality already provided by ADK

--------------------------------------------------
2. SYSTEM GOAL
--------------------------------------------------

Build a RAG-enabled agent that can:

- Manage document corpora
- Ingest documents (Google Drive, Docs, GCS)
- Perform semantic retrieval using Vertex AI RAG
- Maintain internal state for active corpus
- Deploy programmatically to Vertex Agent Engine

--------------------------------------------------
3. TOOLING (REQUIRED)
--------------------------------------------------

Implement tools using ADK tool pattern (no custom wrappers beyond necessity):

- rag_query
- list_corpora
- create_corpus
- add_data
- get_corpus_info
- delete_document
- delete_corpus

Requirements:

- Use vertexai.rag directly
- Return structured outputs: {status, message, data}
- Validate all inputs
- Handle errors explicitly (no silent failures)

Reuse:
- Prefer existing ADK tool patterns
- Do NOT create new tool base classes

--------------------------------------------------
4. STATE MANAGEMENT
--------------------------------------------------

Use ADK ToolContext / session.state:

Track:
- current_corpus
- corpus_exists_<name>

Rules:
- Automatically set current corpus when created/used
- Allow fallback to current corpus if not provided
- Never expose internal resource names to users

--------------------------------------------------
5. AGENT IMPLEMENTATION
--------------------------------------------------

Use:

google.adk.agents.Agent

The agent must:

- Use tools for ALL external operations
- Never directly call APIs from reasoning layer
- Route user requests correctly to tools

--------------------------------------------------
6. AGENT INSTRUCTION (MANDATORY)
--------------------------------------------------

Define a clear instruction block inside the Agent.

The instruction must enforce:

### Role
- RAG agent for corpus management + retrieval

### Skills
- Query documents (rag_query)
- Manage corpora (create/list/add/info/delete)

### Decision Logic
- If user asks a question → use rag_query
- If user manages data → use corresponding tool
- Always identify corpus before operations

### Tool Rules
- Use current corpus if not specified
- Prefer internal resource names when available

### Safety
- Require confirmation before:
  - delete_document
  - delete_corpus

### Communication
- Clearly explain actions taken
- Mention corpus used
- Provide actionable error messages

Do NOT:
- Expose internal system details
- Skip confirmation for destructive actions

--------------------------------------------------
7. CONFIGURATION
--------------------------------------------------

All configuration must be centralized.

- Load from .env using dotenv
- No hardcoded values

Include:
- project_id
- location
- LLM model
- embedding model
- chunking config
- retrieval config

All modules must import from config

--------------------------------------------------
8. LOCAL EXECUTION
--------------------------------------------------

Ensure compatibility with:

adk web

Provide:
- entrypoint to run agent locally

Local flow:
- Load config
- Initialize Vertex AI
- Start agent

--------------------------------------------------
9. DEPLOYMENT (CRITICAL)
--------------------------------------------------

Implement deployment script:

deployment/deploy.py

Requirements:

- Fully automated deployment to Vertex Agent Engine
- No manual UI steps
- Must run via:

  python deployment/deploy.py

Flow:

1. Load config
2. Validate environment
3. Initialize Vertex AI
4. Package agent
5. Deploy to Agent Engine
6. Print deployment details

Rules:

- Use config values only
- No hardcoded credentials
- Include logging + error handling

--------------------------------------------------
10. CODE QUALITY
--------------------------------------------------

- Follow AGENTS.md style rules
- Use clear docstrings (why, not what)
- Keep functions small and focused
- Avoid duplication
- Prefer simple solutions over abstraction

--------------------------------------------------
11. CONSTRAINTS
--------------------------------------------------

- MUST follow AGENTS.md rules
- MUST reuse ADK components
- MUST use vertexai.rag
- MUST use tool-based agent design
- MUST support local + deployed execution

--------------------------------------------------
12. GOAL
--------------------------------------------------

Deliver a system where:

- Agent runs locally via ADK Web
- Same codebase deploys via script
- Full RAG lifecycle works:

  create → ingest → query → inspect → delete

This must behave as a clean, production-grade ADK agent system.

USe the llm - "Gemma 3 27B". Take API key from config file