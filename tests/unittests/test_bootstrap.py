"""Tests for environment bootstrap helpers."""

from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

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

from agents.ragagent import bootstrap


class BootstrapTests(TestCase):
  """Keeps environment validation deterministic and explicit."""

  def test_validate_runtime_environment_accepts_vertex_settings(self) -> None:
    env = {
      "GOOGLE_CLOUD_PROJECT": "demo-project",
      "GOOGLE_CLOUD_LOCATION": "us-central1",
      "GOOGLE_GENAI_USE_VERTEXAI": "1",
      "RAG_AGENT_LLM_MODEL": "gemini-2.5-flash",
      "DEFAULT_EMBEDDING_MODEL": (
        "publishers/google/models/text-embedding-005"
      ),
      "DEFAULT_CHUNK_SIZE": "512",
      "DEFAULT_CHUNK_OVERLAP": "100",
      "DEFAULT_TOP_K": "5",
      "DEFAULT_DISTANCE_THRESHOLD": "0.4",
    }

    with patch.dict(os.environ, env, clear=True):
      bootstrap.validate_runtime_environment()

  def test_validate_runtime_environment_requires_vertex_backend(self) -> None:
    env = {
      "GOOGLE_CLOUD_PROJECT": "demo-project",
      "GOOGLE_CLOUD_LOCATION": "us-central1",
      "GOOGLE_GENAI_USE_VERTEXAI": "0",
      "RAG_AGENT_LLM_MODEL": "gemini-2.5-flash",
      "DEFAULT_EMBEDDING_MODEL": (
        "publishers/google/models/text-embedding-005"
      ),
      "DEFAULT_CHUNK_SIZE": "512",
      "DEFAULT_CHUNK_OVERLAP": "100",
      "DEFAULT_TOP_K": "5",
      "DEFAULT_DISTANCE_THRESHOLD": "0.4",
    }

    with patch.dict(os.environ, env, clear=True):
      with self.assertRaisesRegex(ValueError, "GOOGLE_GENAI_USE_VERTEXAI"):
        bootstrap.validate_runtime_environment()

  def test_normalize_bucket_uri_accepts_plain_bucket_names(self) -> None:
    self.assertEqual(
      bootstrap.normalize_bucket_uri("my-staging-bucket"),
      "gs://my-staging-bucket",
    )

  def test_get_embedding_model_adds_vertex_publisher_prefix(self) -> None:
    env = {"DEFAULT_EMBEDDING_MODEL": "text-embedding-005"}

    with patch.dict(os.environ, env, clear=True):
      self.assertEqual(
        bootstrap.get_embedding_model(),
        "publishers/google/models/text-embedding-005",
      )

  def test_validate_runtime_environment_rejects_retired_gecko_model(
      self,
  ) -> None:
    env = {
      "GOOGLE_CLOUD_PROJECT": "demo-project",
      "GOOGLE_CLOUD_LOCATION": "us-central1",
      "GOOGLE_GENAI_USE_VERTEXAI": "1",
      "RAG_AGENT_LLM_MODEL": "gemini-2.5-flash",
      "DEFAULT_EMBEDDING_MODEL": "textembedding-gecko@003",
      "DEFAULT_CHUNK_SIZE": "512",
      "DEFAULT_CHUNK_OVERLAP": "100",
      "DEFAULT_TOP_K": "5",
      "DEFAULT_DISTANCE_THRESHOLD": "0.4",
    }

    with patch.dict(os.environ, env, clear=True):
      with self.assertRaisesRegex(ValueError, "retired publisher model"):
        bootstrap.validate_runtime_environment()

  def test_load_environment_applies_legacy_aliases(self) -> None:
    env = {
      "GOOGLE_CLOUD_PROJECT": "demo-project",
      "GOOGLE_CLOUD_LOCATION": "us-central1",
      "GOOGLE_GENAI_USE_VERTEXAI": "1",
      "LLM_MODEL": "gemini-2.5-flash",
      "EMBEDDING_MODEL": "publishers/google/models/text-embedding-005",
      "RAG_CHUNK_SIZE": "512",
      "RAG_CHUNK_OVERLAP": "100",
      "RAG_TOP_K": "5",
      "GOOGLE_CLOUD_STAGING_BUCKET": "demo-bucket",
    }

    with patch.dict(os.environ, env, clear=True):
      bootstrap._apply_legacy_aliases()
      self.assertEqual(
        os.environ["RAG_AGENT_LLM_MODEL"],
        "gemini-2.5-flash",
      )
      self.assertEqual(
        os.environ["DEFAULT_EMBEDDING_MODEL"],
        "publishers/google/models/text-embedding-005",
      )
      self.assertEqual(os.environ["DEFAULT_CHUNK_SIZE"], "512")
      self.assertEqual(os.environ["DEFAULT_CHUNK_OVERLAP"], "100")
      self.assertEqual(os.environ["DEFAULT_TOP_K"], "5")
      self.assertEqual(os.environ["GOOGLE_CLOUD_STORAGE_BUCKET"], "demo-bucket")
