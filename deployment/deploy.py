"""Deploys the ADK RAG agent to Vertex AI Agent Engine."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

import google.auth
from google.auth.transport.requests import AuthorizedSession
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

AGENT_OWNER = "kg.aero@gmail.com"
DEPLOYMENT_ENVIRONMENT = "production"
CANONICAL_RESOURCE_NAME = (
  "projects/298838101629/locations/us-central1/"
  "reasoningEngines/1387333535357992960"
)
AGENT_DESCRIPTION = (
  "Vertex AI RAG agent for corpus management and retrieval. "
  f"Owner: {AGENT_OWNER}. Environment: {DEPLOYMENT_ENVIRONMENT}."
)
AGENT_LABELS = {
  "app": "ragagent",
  "environment": DEPLOYMENT_ENVIRONMENT,
  "owner": "kg-aero",
}
AGENT_REGISTRY_BASE_URL = "https://agentregistry.googleapis.com/v1alpha"
AGENT_REGISTRY_VERIFY_ATTEMPTS = 12
AGENT_REGISTRY_VERIFY_DELAY_SECONDS = 5

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
DEPLOYED_ENV_ALIASES = {
  "RAG_AGENT_GOOGLE_CLOUD_PROJECT": "GOOGLE_CLOUD_PROJECT",
  "RAG_AGENT_GOOGLE_CLOUD_LOCATION": "GOOGLE_CLOUD_LOCATION",
}


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
  for deployed_name, local_name in DEPLOYED_ENV_ALIASES.items():
    env_vars[deployed_name] = require_env(local_name)
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


def _agent_id_for_resource(resource_name: str) -> str:
  """Returns the Agent Registry ID derived from an Agent Engine resource."""
  registry_resource = resource_name.replace(
    "/reasoningEngines/",
    "/aiplatform/reasoningEngines/",
  )
  return "urn:agent:projects-298838101629:" + registry_resource.replace(
    "/",
    ":",
  )


def _registry_agents_url(project: str, location: str) -> str:
  """Builds the Agent Registry list URL for deployed agents."""
  return (
    f"{AGENT_REGISTRY_BASE_URL}/projects/{project}/locations/{location}/"
    "agents?pageSize=100"
  )


def _find_agent_registry_entry(
    project: str,
    location: str,
    resource_name: str,
) -> dict[str, object] | None:
  """Finds the Agent Registry entry that points to the deployed resource."""
  credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  session = AuthorizedSession(credentials)
  expected_agent_id = _agent_id_for_resource(resource_name)
  response = session.get(_registry_agents_url(project, location), timeout=60)
  response.raise_for_status()

  for agent in response.json().get("agents", []):
    if agent.get("agentId") == expected_agent_id:
      return agent
  return None


def _verify_agent_registry_entry(
    project: str,
    location: str,
    resource_name: str,
) -> dict[str, object]:
  """Waits until Agent Registry contains the deployed Agent Engine resource."""
  for attempt in range(AGENT_REGISTRY_VERIFY_ATTEMPTS):
    registry_entry = _find_agent_registry_entry(project, location, resource_name)
    if registry_entry is not None:
      return registry_entry
    if attempt < AGENT_REGISTRY_VERIFY_ATTEMPTS - 1:
      time.sleep(AGENT_REGISTRY_VERIFY_DELAY_SECONDS)

  raise RuntimeError(
    "Agent Registry did not contain the deployed resource after deployment: "
    f"{resource_name}"
  )


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
      "description": AGENT_DESCRIPTION,
      "labels": AGENT_LABELS,
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
  registry_entry = _verify_agent_registry_entry(project, location, resource_name)

  print("Deployment status: success")
  print(f"Agent resource name: {resource_name}")
  print(f"Agent owner: {AGENT_OWNER}")
  print(f"Deployment environment: {DEPLOYMENT_ENVIRONMENT}")
  print(f"Agent Registry entry: {registry_entry.get('name')}")
  print(f"Agent Registry ID: {registry_entry.get('agentId')}")
  print(f"Previous canonical resource: {CANONICAL_RESOURCE_NAME}")
  print(f"Python SDK get: client.agent_engines.get(name='{resource_name}')")
  print(f"Metadata URL: {urls['metadata']}")
  print(f"Query URL: {urls['query']}")
  print(f"Stream query URL: {urls['stream_query']}")


if __name__ == "__main__":
  main()
