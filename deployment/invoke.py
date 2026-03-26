"""Invokes a deployed Vertex AI Agent Engine instance from the CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

import vertexai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = PROJECT_ROOT / "agents"
DEFAULT_USER_ID = "local-user"

if str(AGENTS_ROOT) not in sys.path:
  sys.path.insert(0, str(AGENTS_ROOT))

from ragagent.bootstrap import load_environment
from ragagent.bootstrap import require_env


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments for invoking the deployed agent."""
  parser = argparse.ArgumentParser(
    description="Invoke a deployed Vertex AI Agent Engine instance.",
  )
  parser.add_argument(
    "message",
    help="User message to send to the deployed agent.",
  )
  parser.add_argument(
    "--resource-name",
    default=None,
    help=(
      "Full Agent Engine resource name. Falls back to "
      "AGENT_ENGINE_RESOURCE_NAME."
    ),
  )
  parser.add_argument(
    "--project",
    default=None,
    help="Google Cloud project. Falls back to GOOGLE_CLOUD_PROJECT.",
  )
  parser.add_argument(
    "--location",
    default=None,
    help="Vertex AI location. Falls back to GOOGLE_CLOUD_LOCATION.",
  )
  parser.add_argument(
    "--user-id",
    default=DEFAULT_USER_ID,
    help="User id attached to the request. Defaults to local-user.",
  )
  parser.add_argument(
    "--session-id",
    default=None,
    help=(
      "Existing session id to reuse. If omitted, the script creates a new "
      "session first."
    ),
  )
  parser.add_argument(
    "--raw-events",
    action="store_true",
    help="Print raw streamed event payloads instead of extracted text.",
  )
  return parser.parse_args(argv)


def _resolve_value(
    explicit_value: str | None,
    env_name: str,
) -> str:
  """Returns a CLI value or falls back to a required environment variable."""
  if explicit_value is not None and explicit_value.strip():
    return explicit_value.strip()
  return require_env(env_name)


def _extract_session_id(session: Any) -> str:
  """Returns the session id from a Vertex AI session response object."""
  if isinstance(session, dict):
    for key in ("id", "session_id"):
      value = session.get(key)
      if isinstance(value, str) and value.strip():
        return value.strip()

  for attribute_name in ("id", "session_id"):
    value = getattr(session, attribute_name, None)
    if isinstance(value, str) and value.strip():
      return value.strip()

  raise ValueError("Vertex AI did not return a usable session id.")


def _extract_text(event: Any) -> str:
  """Extracts plain text from a streamed event when possible."""
  if isinstance(event, str):
    return event

  text_value = None
  if isinstance(event, dict):
    text_value = event.get("text")
  else:
    text_value = getattr(event, "text", None)
  if isinstance(text_value, str) and text_value.strip():
    return text_value.strip()

  parts: list[Any] = []
  content = event.get("content") if isinstance(event, dict) else getattr(
    event, "content", None
  )
  if isinstance(content, dict):
    parts = list(content.get("parts") or [])
  elif content is not None:
    parts = list(getattr(content, "parts", []) or [])

  text_chunks = []
  for part in parts:
    if isinstance(part, dict):
      part_text = part.get("text")
    else:
      part_text = getattr(part, "text", None)
    if isinstance(part_text, str) and part_text.strip():
      text_chunks.append(part_text.strip())

  return "\n".join(text_chunks).strip()


def _serialize_event(event: Any) -> str:
  """Serializes a streamed event into a readable string."""
  if hasattr(event, "model_dump"):
    return json.dumps(event.model_dump(mode="json"), indent=2, default=str)
  if hasattr(event, "to_dict"):
    return json.dumps(event.to_dict(), indent=2, default=str)
  if isinstance(event, dict):
    return json.dumps(event, indent=2, default=str)
  return str(event)


async def _invoke(args: argparse.Namespace) -> None:
  """Creates or reuses a session, then streams a single agent response."""
  load_environment()
  client = vertexai.Client(
    project=_resolve_value(args.project, "GOOGLE_CLOUD_PROJECT"),
    location=_resolve_value(args.location, "GOOGLE_CLOUD_LOCATION"),
  )
  remote_agent = client.agent_engines.get(
    name=_resolve_value(args.resource_name, "AGENT_ENGINE_RESOURCE_NAME"),
  )

  session_id = args.session_id
  if session_id is None:
    session = await remote_agent.async_create_session(user_id=args.user_id)
    session_id = _extract_session_id(session)
    print(f"Created session id: {session_id}", file=sys.stderr)
  else:
    print(f"Using session id: {session_id}", file=sys.stderr)

  last_output = None
  async for event in remote_agent.async_stream_query(
      user_id=args.user_id,
      session_id=session_id,
      message=args.message,
  ):
    output = (
      _serialize_event(event)
      if args.raw_events
      else _extract_text(event)
    )
    output = output.strip()
    if not output or output == last_output:
      continue
    print(output, flush=True)
    last_output = output


def main(argv: list[str] | None = None) -> None:
  """Runs the CLI invocation entry point."""
  asyncio.run(_invoke(_parse_args(argv)))


if __name__ == "__main__":
  main()
