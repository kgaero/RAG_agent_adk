"""Tests for the Agent Engine deployment helpers."""

from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from deployment import deploy


class DeployTests(TestCase):
  """Covers deploy-time helpers without running a real deployment."""

  def test_deployed_env_vars_reads_required_runtime_values(self) -> None:
    env = {
      "GOOGLE_GENAI_USE_VERTEXAI": "1",
      "RAG_AGENT_LLM_MODEL": "gemini-2.5-flash",
      "DEFAULT_EMBEDDING_MODEL": "text-embedding-005",
      "DEFAULT_CHUNK_SIZE": "512",
      "DEFAULT_CHUNK_OVERLAP": "100",
      "DEFAULT_TOP_K": "5",
      "DEFAULT_DISTANCE_THRESHOLD": "0.4",
    }
    expected_env = dict(env)
    expected_env["PYTHONPATH"] = "/code:/code/agents"

    with patch.dict(os.environ, env, clear=True):
      self.assertEqual(deploy._deployed_env_vars(), expected_env)

  def test_execution_urls_are_derived_from_resource_name(self) -> None:
    resource_name = "projects/demo/locations/us-central1/reasoningEngines/123"

    urls = deploy._execution_urls(resource_name, "us-central1")

    self.assertEqual(
      urls["metadata"],
      (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/demo/locations/us-central1/reasoningEngines/123"
      ),
    )
    self.assertEqual(
      urls["query"],
      (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/demo/locations/us-central1/reasoningEngines/123:query"
      ),
    )
    self.assertEqual(
      urls["stream_query"],
      (
        "https://us-central1-aiplatform.googleapis.com/v1/"
        "projects/demo/locations/us-central1/reasoningEngines/123:streamQuery"
      ),
    )

  def test_extra_packages_use_repo_relative_agents_path(self) -> None:
    self.assertEqual(deploy._extra_packages(), ["agents"])
