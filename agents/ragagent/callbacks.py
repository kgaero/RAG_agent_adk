"""ADK callbacks for request sanitization."""

from __future__ import annotations

import logging
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from .tools import _normalize_path

LOGGER = logging.getLogger(__name__)

_ATTACHMENT_NOTE = (
  "Direct chat attachments are not supported by this agent. Use a Google "
  "Drive URL, Google Docs URL, or gs:// path instead."
)
_PATH_TOKEN_PATTERN = re.compile(r"(gs://\S+|https://\S+)")
_INGESTION_HINT = (
  "Routing note for the latest user turn: the message contains supported "
  "ingestion path(s): {paths}. https://storage.googleapis.com URLs are valid "
  "and will be normalized to gs:// by the tool. Do not ask the user to "
  "reformat those URLs. If the user is asking to add or ingest documents, "
  "your next step must be to call add_data with these path values."
)
_INGESTION_KEYWORDS = (
  "add",
  "corpus",
  "document",
  "file",
  "import",
  "ingest",
  "load",
  "upload",
)


def _build_attachment_notice(display_name: Optional[str]) -> types.Part:
  """Builds a short text placeholder for unsupported chat attachments."""
  attachment_label = (
    f" '{display_name}'" if display_name and display_name.strip() else ""
  )
  return types.Part.from_text(
    text=(
      f"[Unsupported chat attachment{attachment_label} omitted. "
      f"{_ATTACHMENT_NOTE}]"
    )
  )


def _sanitize_content(content: types.Content) -> tuple[types.Content, int]:
  """Replaces inline data parts that are missing a MIME type."""
  sanitized_parts: list[types.Part] = []
  replaced_parts = 0

  for part in content.parts or []:
    inline_data = part.inline_data
    if inline_data and not (inline_data.mime_type or "").strip():
      sanitized_parts.append(
        _build_attachment_notice(getattr(inline_data, "display_name", None))
      )
      replaced_parts += 1
      continue
    sanitized_parts.append(part)

  if not replaced_parts:
    return content, 0

  return content.model_copy(update={"parts": sanitized_parts}), replaced_parts


def _latest_user_text(llm_request: LlmRequest) -> str:
  """Returns the latest plain-text user content from the request."""
  for content in reversed(llm_request.contents):
    if content.role != "user":
      continue
    text_parts = [
      part.text.strip()
      for part in content.parts or []
      if part.text and part.text.strip()
    ]
    if text_parts:
      return "\n".join(text_parts)
  return ""


def _extract_supported_paths(message: str) -> list[str]:
  """Finds supported ingestion paths in free-form user text."""
  normalized_paths: list[str] = []
  for token in _PATH_TOKEN_PATTERN.findall(message):
    candidate = token.rstrip(".,);]")
    try:
      normalized_path = _normalize_path(candidate)
    except ValueError:
      continue
    if normalized_path not in normalized_paths:
      normalized_paths.append(normalized_path)
  return normalized_paths


def _should_route_to_add_data(
    message: str,
    normalized_paths: list[str],
) -> bool:
  """Determines when the latest user turn should force add_data routing."""
  if not normalized_paths:
    return False

  lowered_message = message.lower()
  if any(keyword in lowered_message for keyword in _INGESTION_KEYWORDS):
    return True

  stripped_message = message.strip()
  for normalized_path in normalized_paths:
    if stripped_message == normalized_path:
      return True

  raw_tokens = _PATH_TOKEN_PATTERN.findall(stripped_message)
  return bool(raw_tokens) and len(raw_tokens) == len(normalized_paths)


async def sanitize_model_request(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> None:
  """Removes malformed inline attachments before the model request is sent."""
  del callback_context

  sanitized_contents: list[types.Content] = []
  replaced_parts = 0

  for content in llm_request.contents:
    sanitized_content, content_replacements = _sanitize_content(content)
    sanitized_contents.append(sanitized_content)
    replaced_parts += content_replacements

  if not replaced_parts:
    pass
  else:
    llm_request.contents = sanitized_contents
    LOGGER.warning(
      "Sanitized %s inline attachment part(s) without mime_type before "
      "sending the request to the model.",
      replaced_parts,
    )

  latest_user_message = _latest_user_text(llm_request)
  normalized_paths = _extract_supported_paths(latest_user_message)
  if _should_route_to_add_data(latest_user_message, normalized_paths):
    llm_request.append_instructions([
      _INGESTION_HINT.format(paths=", ".join(normalized_paths))
    ])

  return None
