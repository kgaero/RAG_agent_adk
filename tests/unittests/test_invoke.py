"""Tests for the Agent Engine invocation helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from deployment import invoke


class InvokeTests(TestCase):
  """Covers CLI helper behavior without calling Vertex AI."""

  def test_resolve_value_prefers_explicit_argument(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
        invoke._resolve_value(
          " projects/demo/locations/us/reasoningEngines/123 ",
          "AGENT_ENGINE_RESOURCE_NAME",
        ),
        "projects/demo/locations/us/reasoningEngines/123",
      )

  def test_resolve_value_falls_back_to_environment(self) -> None:
    env = {
      "AGENT_ENGINE_RESOURCE_NAME": (
        "projects/demo/locations/us-central1/reasoningEngines/123"
      ),
    }

    with patch.dict(os.environ, env, clear=True):
      self.assertEqual(
        invoke._resolve_value(None, "AGENT_ENGINE_RESOURCE_NAME"),
        env["AGENT_ENGINE_RESOURCE_NAME"],
      )

  def test_extract_session_id_supports_object_response(self) -> None:
    session = SimpleNamespace(id="session-123")

    self.assertEqual(invoke._extract_session_id(session), "session-123")

  def test_extract_text_reads_content_parts(self) -> None:
    event = SimpleNamespace(
      content=SimpleNamespace(
        parts=[
          SimpleNamespace(text="first"),
          SimpleNamespace(text="second"),
        ]
      )
    )

    self.assertEqual(invoke._extract_text(event), "first\nsecond")
