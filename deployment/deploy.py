"""Deploys the ADK RAG agent to Vertex AI Agent Engine."""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import vertexai
from vertexai.agent_engines import AdkApp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = PROJECT_ROOT / "agents"
AGENTS_PACKAGE_PATH = AGENTS_ROOT.relative_to(PROJECT_ROOT)

if str(AGENTS_ROOT) not in sys.path:
  sys.path.insert(0, str(AGENTS_ROOT))

from ragagent.bootstrap import load_environment
from ragagent.bootstrap import normalize_bucket_uri
from ragagent.bootstrap import require_env
from ragagent.bootstrap import validate_deployment_environment

LOGGER = logging.getLogger(__name__)

DEPLOYED_ENV_VARS = [
  "GOOGLE_GENAI_USE_VERTEXAI",
  "RAG_AGENT_LLM_MODEL",
  "DEFAULT_EMBEDDING_MODEL",
  "DEFAULT_CHUNK_SIZE",
  "DEFAULT_CHUNK_OVERLAP",
  "DEFAULT_TOP_K",
  "DEFAULT_DISTANCE_THRESHOLD",
]
DEPLOYED_PYTHONPATH = "/code:/code/agents"


def _load_root_agent():
  """Loads the ADK root agent only when deployment actually runs."""
  from ragagent.agent import root_agent

  return root_agent


def _deployed_env_vars() -> dict[str, str]:
  """Returns the runtime env values copied into Agent Engine."""
  env_vars = {
    env_var: require_env(env_var)
    for env_var in DEPLOYED_ENV_VARS
  }
  env_vars["PYTHONPATH"] = DEPLOYED_PYTHONPATH
  return env_vars


def _extra_packages() -> list[str]:
  """Returns relative package paths so Agent Engine preserves imports."""
  return [str(AGENTS_PACKAGE_PATH)]


def _execution_urls(resource_name: str, location: str) -> dict[str, str]:
  """Builds authenticated Agent Engine REST endpoints."""
  api_base = f"https://{location}-aiplatform.googleapis.com/v1/{resource_name}"
  return {
    "metadata": api_base,
    "query": f"{api_base}:query",
    "stream_query": f"{api_base}:streamQuery",
  }


def main() -> None:
  """Packages and deploys the ADK agent with environment-based settings."""
  logging.basicConfig(level=logging.INFO)
  load_environment()
  validate_deployment_environment()
  root_agent = _load_root_agent()

  project = require_env("GOOGLE_CLOUD_PROJECT")
  location = require_env("GOOGLE_CLOUD_LOCATION")
  staging_bucket = normalize_bucket_uri(
    require_env("GOOGLE_CLOUD_STORAGE_BUCKET")
  )
  client = vertexai.Client(project=project, location=location)

  LOGGER.info("Packaging ADK app for Vertex AI Agent Engine.")
  adk_app = AdkApp(
    agent=root_agent,
    app_name=root_agent.name,
    enable_tracing=True,
  )

  remote_agent = client.agent_engines.create(
    agent=adk_app,
    config={
      "staging_bucket": staging_bucket,
      "requirements": str(PROJECT_ROOT / "deployment" / "requirements.txt"),
      "display_name": root_agent.name,
      "description": (
        "Vertex AI RAG agent deployed from the repo deployment script."
      ),
      "extra_packages": _extra_packages(),
      "env_vars": _deployed_env_vars(),
    },
  )

  resource_name = getattr(
    getattr(remote_agent, "api_resource", None),
    "name",
    getattr(remote_agent, "resource_name", ""),
  )
  urls = _execution_urls(resource_name, location)

  print("Deployment status: success")
  print(f"Agent resource name: {resource_name}")
  print(f"Python SDK get: client.agent_engines.get(name='{resource_name}')")
  print(f"Metadata URL: {urls['metadata']}")
  print(f"Query URL: {urls['query']}")
  print(f"Stream query URL: {urls['stream_query']}")


if __name__ == "__main__":
  main()
