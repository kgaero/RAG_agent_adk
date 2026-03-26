"""Unit tests for model request sanitization callbacks."""

from __future__ import annotations

import os
from unittest import IsolatedAsyncioTestCase

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

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from agents.ragagent.callbacks import sanitize_model_request


class RagCallbackTests(IsolatedAsyncioTestCase):
  """Covers request sanitization for malformed attachment parts."""

  async def test_sanitize_model_request_replaces_invalid_inline_data(
      self,
  ) -> None:
    llm_request = LlmRequest(
      contents=[
        types.Content(
          role="user",
          parts=[
            types.Part.from_text(text="Add the attached file."),
            types.Part(
              inline_data=types.Blob(
                data=b"requirements",
                mime_type=None,
                display_name="PRD.md",
              )
            ),
          ],
        )
      ]
    )

    await sanitize_model_request(callback_context=None, llm_request=llm_request)

    parts = llm_request.contents[0].parts
    self.assertEqual(parts[0].text, "Add the attached file.")
    self.assertIsNone(parts[1].inline_data)
    self.assertIn("PRD.md", parts[1].text)
    self.assertIn("Google Drive URL", parts[1].text)

  async def test_sanitize_model_request_preserves_valid_inline_data(
      self,
  ) -> None:
    llm_request = LlmRequest(
      contents=[
        types.Content(
          role="user",
          parts=[
            types.Part(
              inline_data=types.Blob(
                data=b"requirements",
                mime_type="text/plain",
                display_name="notes.txt",
              )
            )
          ],
        )
      ]
    )

    await sanitize_model_request(callback_context=None, llm_request=llm_request)

    part = llm_request.contents[0].parts[0]
    self.assertIsNotNone(part.inline_data)
    self.assertEqual(part.inline_data.mime_type, "text/plain")

  async def test_sanitize_model_request_adds_ingestion_hint_for_gcs_url(
      self,
  ) -> None:
    llm_request = LlmRequest(
      contents=[
        types.Content(
          role="user",
          parts=[
            types.Part.from_text(
              text=(
                "add below document to Fan corpus\n"
                "https://storage.googleapis.com/demo-bucket/path/to/PRD.md"
              )
            )
          ],
        )
      ]
    )

    await sanitize_model_request(callback_context=None, llm_request=llm_request)

    system_instruction = llm_request.config.system_instruction
    self.assertIsInstance(system_instruction, str)
    self.assertIn("must be to call add_data", system_instruction)
    self.assertIn("gs://demo-bucket/path/to/PRD.md", system_instruction)
