"""Environment and Vertex AI bootstrap helpers.

This module stays intentionally small so the runtime still reads directly from
`.env` and `os.environ`, while centralizing the fail-fast validation required
by the PRD.
"""

from __future__ import annotations

import os
from pathlib import Path
import re

from dotenv import load_dotenv
import vertexai

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

RUNTIME_ENV_VARS = (
  "GOOGLE_CLOUD_PROJECT",
  "GOOGLE_CLOUD_LOCATION",
  "RAG_AGENT_LLM_MODEL",
  "DEFAULT_EMBEDDING_MODEL",
  "DEFAULT_CHUNK_SIZE",
  "DEFAULT_CHUNK_OVERLAP",
  "DEFAULT_TOP_K",
  "DEFAULT_DISTANCE_THRESHOLD",
)

DEPLOYMENT_ENV_VARS = (
  "GOOGLE_CLOUD_STORAGE_BUCKET",
)

TRUTHY_VALUES = {"1", "true", "yes", "on"}
PUBLISHER_MODEL_PREFIX = "publishers/google/models/"
RETIRED_PUBLISHER_EMBEDDING_MODELS = frozenset({
  "textembedding-gecko@001",
  "textembedding-gecko@002",
  "textembedding-gecko@003",
  "textembedding-gecko-multilingual@001",
})
SUPPORTED_PUBLISHER_EMBEDDING_MODELS = (
  "text-embedding-005",
  "text-embedding-004",
  "text-multilingual-embedding-002",
)

ENV_ALIASES = {
  "RAG_AGENT_LLM_MODEL": ("LLM_MODEL",),
  "DEFAULT_EMBEDDING_MODEL": ("EMBEDDING_MODEL",),
  "DEFAULT_CHUNK_SIZE": ("RAG_CHUNK_SIZE",),
  "DEFAULT_CHUNK_OVERLAP": ("RAG_CHUNK_OVERLAP",),
  "DEFAULT_TOP_K": ("RAG_TOP_K",),
  "GOOGLE_CLOUD_STORAGE_BUCKET": ("GOOGLE_CLOUD_STAGING_BUCKET",),
}


def _apply_legacy_aliases() -> None:
  """Maps older repo env keys into the PRD-required canonical names.

  This keeps the runtime compatible with the existing workspace while the
  canonical PRD names remain the source of truth for all new configuration.
  """
  for canonical_name, alias_names in ENV_ALIASES.items():
    current_value = os.getenv(canonical_name)
    if current_value is not None and current_value.strip():
      continue
    for alias_name in alias_names:
      alias_value = os.getenv(alias_name)
      if alias_value is not None and alias_value.strip():
        os.environ[canonical_name] = alias_value.strip()
        break


def load_environment() -> None:
  """Loads `.env` so local ADK and deployment runs see the same variables."""
  load_dotenv(dotenv_path=ENV_PATH)
  _apply_legacy_aliases()


def require_env(name: str) -> str:
  """Returns a required environment variable or raises a clear error."""
  value = os.getenv(name)
  if value is None or not value.strip():
    raise ValueError(f"Missing required environment variable: {name}")
  return value.strip()


def require_int_env(name: str) -> int:
  """Parses a positive integer environment variable used by RAG settings."""
  value = require_env(name)
  try:
    parsed_value = int(value)
  except ValueError as err:
    raise ValueError(f"{name} must be an integer.") from err
  if parsed_value <= 0:
    raise ValueError(f"{name} must be greater than zero.")
  return parsed_value


def require_float_env(name: str) -> float:
  """Parses a non-negative float environment variable for retrieval limits."""
  value = require_env(name)
  try:
    parsed_value = float(value)
  except ValueError as err:
    raise ValueError(f"{name} must be a float.") from err
  if parsed_value < 0:
    raise ValueError(f"{name} must be greater than or equal to zero.")
  return parsed_value


def _validate_vertex_backend() -> None:
  """Ensures the ADK agent uses Vertex AI instead of API-key mode."""
  backend_value = require_env("GOOGLE_GENAI_USE_VERTEXAI")
  if backend_value.lower() not in TRUTHY_VALUES:
    raise ValueError(
        "GOOGLE_GENAI_USE_VERTEXAI must be set to a truthy value because "
        "this project uses Vertex AI for both the LLM and RAG."
    )


def _extract_publisher_model_id(model_name: str) -> str | None:
  """Returns the Google publisher model ID when the name matches one."""
  normalized_name = model_name.strip()
  if not normalized_name:
    return None

  if normalized_name.startswith(PUBLISHER_MODEL_PREFIX):
    return normalized_name.removeprefix(PUBLISHER_MODEL_PREFIX)

  full_resource_name = re.fullmatch(
      (
        r"projects/[^/]+/locations/[^/]+/publishers/google/models/"
        r"(?P<model_id>[^/]+)"
      ),
      normalized_name,
  )
  if full_resource_name:
    return full_resource_name.group("model_id")

  if "/" not in normalized_name:
    return normalized_name
  return None


def _validate_embedding_model(name: str) -> str:
  """Rejects retired Google publisher models before corpus creation."""
  normalized_name = require_env(name)
  publisher_model_id = _extract_publisher_model_id(normalized_name)
  if publisher_model_id in RETIRED_PUBLISHER_EMBEDDING_MODELS:
    supported_models = ", ".join(SUPPORTED_PUBLISHER_EMBEDDING_MODELS)
    raise ValueError(
        f"{name} uses retired publisher model '{publisher_model_id}'. Use one "
        f"of: {supported_models}."
    )
  return normalized_name


def validate_runtime_environment() -> None:
  """Validates all runtime settings before the agent is created."""
  for env_var in RUNTIME_ENV_VARS:
    require_env(env_var)
  _validate_vertex_backend()
  _validate_embedding_model("DEFAULT_EMBEDDING_MODEL")
  require_int_env("DEFAULT_CHUNK_SIZE")
  require_int_env("DEFAULT_CHUNK_OVERLAP")
  require_int_env("DEFAULT_TOP_K")
  require_float_env("DEFAULT_DISTANCE_THRESHOLD")


def validate_deployment_environment() -> None:
  """Validates deployment-only settings in addition to runtime settings."""
  validate_runtime_environment()
  for env_var in DEPLOYMENT_ENV_VARS:
    require_env(env_var)


def normalize_bucket_uri(bucket_value: str) -> str:
  """Normalizes a bucket name into the URI format expected by Vertex AI."""
  if bucket_value.startswith("gs://"):
    return bucket_value
  return f"gs://{bucket_value}"


def initialize_vertex_ai(*, require_staging_bucket: bool = False) -> None:
  """Initializes Vertex AI exactly once per process before tool usage."""
  load_environment()
  if require_staging_bucket:
    validate_deployment_environment()
  else:
    validate_runtime_environment()

  init_kwargs = {
    "project": require_env("GOOGLE_CLOUD_PROJECT"),
    "location": require_env("GOOGLE_CLOUD_LOCATION"),
  }

  if require_staging_bucket:
    init_kwargs["staging_bucket"] = normalize_bucket_uri(
        require_env("GOOGLE_CLOUD_STORAGE_BUCKET")
    )

  vertexai.init(**init_kwargs)


def get_agent_model() -> str:
  """Returns the configured Gemini model for the ADK agent."""
  return require_env("RAG_AGENT_LLM_MODEL")


def get_embedding_model() -> str:
  """Returns the configured Vertex embedding model resource."""
  model_name = _validate_embedding_model("DEFAULT_EMBEDDING_MODEL")
  if model_name.startswith("projects/"):
    return model_name
  if model_name.startswith(PUBLISHER_MODEL_PREFIX):
    return model_name
  return f"{PUBLISHER_MODEL_PREFIX}{model_name}"


def get_default_chunk_size() -> int:
  """Returns the configured chunk size for RAG ingestion."""
  return require_int_env("DEFAULT_CHUNK_SIZE")


def get_default_chunk_overlap() -> int:
  """Returns the configured chunk overlap for RAG ingestion."""
  return require_int_env("DEFAULT_CHUNK_OVERLAP")


def get_default_top_k() -> int:
  """Returns the configured retrieval depth for RAG queries."""
  return require_int_env("DEFAULT_TOP_K")


def get_default_distance_threshold() -> float:
  """Returns the configured retrieval distance threshold for RAG queries."""
  return require_float_env("DEFAULT_DISTANCE_THRESHOLD")
