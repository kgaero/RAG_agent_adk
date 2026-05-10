"""ADK entrypoint for the Vertex AI RAG agent."""

from __future__ import annotations

from typing import Any

from google.adk.agents import Agent
from google.adk.tools import BaseTool
from google.adk.tools import load_memory
from google.adk.tools import ToolContext
from google.adk.tools import preload_memory

from .bootstrap import get_agent_model
from .bootstrap import initialize_vertex_ai
from .bootstrap import load_environment
from .bootstrap import validate_runtime_environment
from .callbacks import sanitize_model_request
from .tools import add_data
from .tools import create_corpus
from .tools import delete_corpus
from .tools import delete_document
from .tools import get_corpus_info
from .tools import list_corpora
from .tools import rag_query

load_environment()
validate_runtime_environment()
initialize_vertex_ai()

INSTRUCTION = """
Role:
- You are a RAG agent for Vertex AI corpus management and retrieval.
- Memory Bank context in <PAST_CONVERSATIONS> is authoritative for user
  preferences such as preferred corpus.

Skills:
- Query documents with rag_query.
- Manage corpora with list_corpora, create_corpus, add_data, get_corpus_info,
  delete_document, and delete_corpus.
- Use ADK Memory Bank through preload_memory and load_memory for remembered
  user preferences.

Decision Logic:
- If the user asks a knowledge question, call rag_query.
- If the user wants to create, inspect, ingest, or delete data, call the
  matching management tool.
- If the user asks about a remembered preference, preferred corpus, or default
  corpus, answer from Memory Bank context or call load_memory.
- If Memory Bank says the user has a preferred corpus and the user does not
  name another corpus, pass the preferred corpus as corpus_name.
- Always determine which corpus should be used before corpus-specific work.

Tool Rules:
- Use the current corpus when the user does not specify one.
- Use the remembered preferred corpus when there is no current corpus and the
  user does not specify another corpus.
- Prefer an internal corpus resource name only when it is already available
  from prior tool state; never ask the user for it.
- Use get_corpus_info before deletion when you need document numbering.
- add_data only supports Google Cloud Storage, Google Drive, and Google Docs
  paths.
- If the user uploads a chat attachment, explain that they must provide a
  supported path instead of a direct upload.

Safety Rules:
- Always require explicit confirmation before delete_document.
- Always require explicit confirmation before delete_corpus.
- Never perform deletion if confirm is false or missing.
- Treat deletion as confirmed only when the latest user message is exactly
  the confirmation phrase requested by the tool guardrail.

Communication Guidelines:
- Be clear and concise.
- Mention which corpus was used for each action.
- Explain what you did after each tool call.
- Return useful error messages and recovery guidance.
- For failed imports, report the imported, failed, and skipped counts from the
  tool result and do not invent backend failure causes that were not returned.

Internal Rules:
- Track current_corpus in session state.
- Treat Memory Bank preferences as user preferences, not document knowledge.
- Keep internal resource names out of user-facing messages.
- Do not expose internal implementation details.
- Do not skip required tool calls.
""".strip()


def guard_destructive_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> dict[str, Any] | None:
  """Blocks destructive tools unless the current user text confirms exactly."""
  if tool.name not in {"delete_document", "delete_corpus"}:
    return None
  if not args.get("confirm"):
    return None
  if tool_context.state.get("eval_mode"):
    return {
      "status": "error",
      "message": "Deletion blocked because this session is running eval mode.",
      "data": {"eval_mode": True},
    }

  expected_phrase = _expected_delete_confirmation(tool.name, args, tool_context)
  user_text = _latest_user_text(tool_context)
  if user_text.strip().casefold() == expected_phrase.casefold():
    return None

  return {
    "status": "error",
    "message": (
      "Deletion blocked. To confirm this destructive action, send exactly: "
      f"{expected_phrase}"
    ),
    "data": {"required_confirmation": expected_phrase},
  }


def _expected_delete_confirmation(
    tool_name: str,
    args: dict[str, Any],
    tool_context: ToolContext,
) -> str:
  """Builds the exact user phrase required for destructive confirmation."""
  if tool_name == "delete_corpus":
    corpus_name = args.get("corpus_name") or tool_context.state.get(
      "current_corpus"
    )
    return f"CONFIRM DELETE CORPUS {corpus_name or '<corpus name>'}"

  document_identifier = args.get("document_name")
  if not document_identifier and args.get("document_number") is not None:
    document_identifier = str(args["document_number"])
  return f"CONFIRM DELETE DOCUMENT {document_identifier or '<document>'}"


def _latest_user_text(tool_context: ToolContext) -> str:
  """Returns plain text from the latest ADK user content."""
  user_content = getattr(tool_context, "user_content", None)
  if not user_content:
    return ""
  parts = getattr(user_content, "parts", []) or []
  return " ".join(
    part.text.strip()
    for part in parts
    if getattr(part, "text", None)
  )


root_agent = Agent(
  name="ragagent",
  model=get_agent_model(),
  description="A Vertex AI RAG agent for corpus management and retrieval.",
  instruction=INSTRUCTION,
  before_model_callback=sanitize_model_request,
  before_tool_callback=guard_destructive_tool,
  tools=[
    rag_query,
    preload_memory,
    load_memory,
    list_corpora,
    create_corpus,
    add_data,
    get_corpus_info,
    delete_document,
    delete_corpus,
  ],
)
