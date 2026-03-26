"""Unit tests for the RAG ADK tools."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock
from unittest.mock import patch

from google.adk.sessions.state import State

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "demo-project")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")
os.environ.setdefault("RAG_AGENT_LLM_MODEL", "gemini-2.5-flash")
os.environ.setdefault(
  "DEFAULT_EMBEDDING_MODEL",
  "publishers/google/models/text-embedding-005",
)
os.environ.setdefault("DEFAULT_CHUNK_SIZE", "512")
os.environ.setdefault("DEFAULT_CHUNK_OVERLAP", "100")
os.environ.setdefault("DEFAULT_TOP_K", "5")
os.environ.setdefault("DEFAULT_DISTANCE_THRESHOLD", "0.4")

from agents.ragagent import tools


class DummyToolContext:
  """Minimal stand-in for ToolContext state used by unit tests."""

  def __init__(self) -> None:
    self.state = {}


class DummyStateToolContext:
  """Uses ADK's State object to match runtime behavior more closely."""

  def __init__(self, state: State) -> None:
    self.state = state


class RagToolTests(IsolatedAsyncioTestCase):
  """Exercises tool behavior without calling Vertex AI."""

  async def test_create_corpus_sets_current_state(self) -> None:
    tool_context = DummyToolContext()
    created_corpus = SimpleNamespace(
      name="projects/demo/locations/us/ragCorpora/123",
      display_name="finance",
      description="finance docs",
    )

    with patch.object(tools, "_list_corpora", new=AsyncMock(return_value=[])):
      with patch.object(
        tools,
        "get_embedding_model",
        return_value="publishers/google/models/text-embedding-005",
      ):
        with patch.object(
          tools.asyncio,
          "to_thread",
          new=AsyncMock(return_value=created_corpus),
        ) as to_thread:
          result = await tools.create_corpus(
            corpus_name="finance",
            description="finance docs",
            tool_context=tool_context,
          )

    self.assertEqual(result["status"], "success")
    self.assertEqual(tool_context.state["current_corpus"], "finance")
    self.assertTrue(tool_context.state["corpus_exists_finance"])
    self.assertEqual(
      tool_context.state["current_corpus_resource"],
      "projects/demo/locations/us/ragCorpora/123",
    )
    self.assertTrue(to_thread.awaited)

  async def test_create_corpus_returns_error_for_retired_gecko_model(
      self,
  ) -> None:
    tool_context = DummyToolContext()

    with patch.object(tools, "_list_corpora", new=AsyncMock(return_value=[])):
      with patch.object(
        tools,
        "get_embedding_model",
        side_effect=ValueError("retired publisher model"),
      ):
        result = await tools.create_corpus(
          corpus_name="finance",
          description="finance docs",
          tool_context=tool_context,
        )

    self.assertEqual(result["status"], "error")
    self.assertIn("retired publisher model", result["message"])

  async def test_add_data_normalizes_google_docs_url(self) -> None:
    tool_context = DummyToolContext()
    tool_context.state["current_corpus"] = "finance"
    tool_context.state["current_corpus_resource"] = (
      "projects/demo/locations/us/ragCorpora/123"
    )
    imported_response = SimpleNamespace(imported_rag_files_count=1)

    with patch.object(
      tools,
      "_resolve_corpus",
      new=AsyncMock(
        return_value=("finance", "projects/demo/locations/us/ragCorpora/123")
      ),
    ):
      with patch.object(tools, "get_default_chunk_size", return_value=512):
        with patch.object(tools, "get_default_chunk_overlap", return_value=100):
          with patch.object(
            tools.asyncio,
            "to_thread",
            new=AsyncMock(return_value=imported_response),
          ) as to_thread:
            result = await tools.add_data(
              paths=[
                (
                  "https://docs.google.com/document/d/test-doc-id/"
                  "edit?usp=sharing"
                )
              ],
              tool_context=tool_context,
            )

    self.assertEqual(result["status"], "success")
    self.assertEqual(
      result["data"]["paths"],
      ["https://drive.google.com/file/d/test-doc-id/view"],
    )
    self.assertTrue(to_thread.awaited)

  async def test_add_data_normalizes_https_gcs_url(self) -> None:
    tool_context = DummyToolContext()
    imported_response = SimpleNamespace(
      imported_rag_files_count=1,
      failed_rag_files_count=0,
      skipped_rag_files_count=0,
      partial_failures_gcs_path="",
    )

    with patch.object(
      tools,
      "_resolve_corpus",
      new=AsyncMock(
        return_value=("finance", "projects/demo/locations/us/ragCorpora/123")
      ),
    ):
      with patch.object(tools, "get_default_chunk_size", return_value=512):
        with patch.object(tools, "get_default_chunk_overlap", return_value=100):
          with patch.object(
            tools.asyncio,
            "to_thread",
            new=AsyncMock(return_value=imported_response),
          ):
            result = await tools.add_data(
              paths=[
                "https://storage.googleapis.com/demo-bucket/path/to/PRD.md"
              ],
              tool_context=tool_context,
            )

    self.assertEqual(result["status"], "success")
    self.assertEqual(
      result["data"]["paths"],
      ["gs://demo-bucket/path/to/PRD.md"],
    )

  async def test_add_data_returns_error_for_zero_imports(self) -> None:
    tool_context = DummyToolContext()
    failed_response = SimpleNamespace(
      imported_rag_files_count=0,
      failed_rag_files_count=1,
      skipped_rag_files_count=0,
      partial_failures_gcs_path="",
    )

    with patch.object(
      tools,
      "_resolve_corpus",
      new=AsyncMock(
        return_value=("finance", "projects/demo/locations/us/ragCorpora/123")
      ),
    ):
      with patch.object(tools, "get_default_chunk_size", return_value=512):
        with patch.object(tools, "get_default_chunk_overlap", return_value=100):
          with patch.object(
            tools.asyncio,
            "to_thread",
            new=AsyncMock(return_value=failed_response),
          ):
            result = await tools.add_data(
              paths=["gs://demo-bucket/path/to/PRD.md"],
              tool_context=tool_context,
            )

    self.assertEqual(result["status"], "error")
    self.assertEqual(result["data"]["failed_count"], 1)
    self.assertIn("No documents were imported", result["message"])

  async def test_delete_document_requires_confirmation(self) -> None:
    result = await tools.delete_document(
      document_name="quarterly-report.pdf",
      confirm=False,
      tool_context=DummyToolContext(),
    )

    self.assertEqual(result["status"], "error")
    self.assertIn("confirm=True", result["message"])

  def test_remove_corpus_state_supports_adk_state(self) -> None:
    state = State(
      value={
        "current_corpus": "finance",
        "current_corpus_resource": "projects/demo/locations/us/ragCorpora/123",
        "corpus_exists_finance": True,
        "corpus_resource_finance": "projects/demo/locations/us/ragCorpora/123",
      },
      delta={},
    )
    tool_context = DummyStateToolContext(state)

    tools._remove_corpus_state(tool_context, display_name="finance")

    self.assertFalse(tool_context.state["corpus_exists_finance"])
    self.assertIsNone(tool_context.state["corpus_resource_finance"])
    self.assertIsNone(tool_context.state["current_corpus"])
    self.assertIsNone(tool_context.state["current_corpus_resource"])

  async def test_rag_query_returns_context_payload(self) -> None:
    tool_context = DummyToolContext()
    query_response = SimpleNamespace(
      contexts=SimpleNamespace(
        contexts=[
          SimpleNamespace(
            source_display_name="quarterly-report.pdf",
            text="Revenue grew 12 percent year over year.",
            distance=0.11,
          )
        ]
      )
    )

    with patch.object(
      tools,
      "_resolve_corpus",
      new=AsyncMock(
        return_value=("finance", "projects/demo/locations/us/ragCorpora/123")
      ),
    ):
      with patch.object(tools, "get_default_top_k", return_value=4):
        with patch.object(
          tools,
          "get_default_distance_threshold",
          return_value=0.35,
        ):
          with patch.object(
            tools.asyncio,
            "to_thread",
            new=AsyncMock(return_value=query_response),
          ):
            result = await tools.rag_query(
              query="How did revenue change?",
              tool_context=tool_context,
            )

    self.assertEqual(result["status"], "success")
    self.assertEqual(result["data"]["corpus"], "finance")
    self.assertEqual(result["data"]["top_k"], 4)
    self.assertEqual(result["data"]["distance_threshold"], 0.35)
    self.assertEqual(
      result["data"]["contexts"][0]["document_name"],
      "quarterly-report.pdf",
    )
