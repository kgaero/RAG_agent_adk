"""ADK tools for Vertex AI RAG corpus management and retrieval."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from typing import Optional
from typing import Sequence
from urllib.parse import unquote
from urllib.parse import urlparse

from google.adk.tools import ToolContext
from google.api_core.exceptions import GoogleAPICallError
from google.api_core.exceptions import RetryError
from vertexai import rag

from .bootstrap import get_default_chunk_overlap
from .bootstrap import get_default_chunk_size
from .bootstrap import get_default_distance_threshold
from .bootstrap import get_embedding_model
from .bootstrap import get_default_top_k

LOGGER = logging.getLogger(__name__)

CURRENT_CORPUS_KEY = "current_corpus"
CURRENT_CORPUS_RESOURCE_KEY = "current_corpus_resource"
CORPUS_EXISTS_PREFIX = "corpus_exists_"
CORPUS_RESOURCE_PREFIX = "corpus_resource_"

DOCS_URL_PATTERN = re.compile(
  r"^https://docs\.google\.com/document/d/([A-Za-z0-9_-]+)"
)


def _success(
    message: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  return {"status": "success", "message": message, "data": data or {}}


def _error(
    message: str,
    data: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  return {"status": "error", "message": message, "data": data or {}}


def _slugify(value: str) -> str:
  slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
  return slug.strip("_")


def _is_resource_name(value: str) -> bool:
  return value.startswith("projects/") and "/ragCorpora/" in value


def _is_supported_path(value: str) -> bool:
  return (
    value.startswith("gs://")
    or value.startswith("https://drive.google.com/")
    or value.startswith("https://docs.google.com/document/")
    or _normalize_gcs_https_url(value) is not None
  )


def _normalize_gcs_https_url(path: str) -> Optional[str]:
  """Converts supported HTTPS Cloud Storage URLs into a gs:// path."""
  parsed = urlparse(path)
  if parsed.scheme != "https":
    return None

  object_path = parsed.path.lstrip("/")
  if parsed.netloc == "storage.googleapis.com":
    if "/" not in object_path:
      return None
    bucket_name, blob_name = object_path.split("/", 1)
    return f"gs://{bucket_name}/{unquote(blob_name)}"

  suffix = ".storage.googleapis.com"
  if not parsed.netloc.endswith(suffix) or not object_path:
    return None

  bucket_name = parsed.netloc.removesuffix(suffix)
  return f"gs://{bucket_name}/{unquote(object_path)}"


def _normalize_path(path: str) -> str:
  stripped_path = path.strip()
  if not stripped_path:
    raise ValueError("Each input path must be non-empty.")
  if not _is_supported_path(stripped_path):
    raise ValueError(
        "Only Google Cloud Storage, Google Drive, and Google Docs paths are "
        "supported."
    )

  normalized_gcs_path = _normalize_gcs_https_url(stripped_path)
  if normalized_gcs_path:
    return normalized_gcs_path

  match = DOCS_URL_PATTERN.match(stripped_path)
  if not match:
    return stripped_path

  file_id = match.group(1)
  # Vertex AI RAG imports Drive links, so docs links are rewritten into the
  # equivalent Drive file URL before ingestion.
  return f"https://drive.google.com/file/d/{file_id}/view"


def _normalize_paths(paths: Sequence[str]) -> list[str]:
  if not paths:
    raise ValueError("At least one source path is required.")
  return [_normalize_path(path) for path in paths]


def _corpus_exists_key(display_name: str) -> str:
  return f"{CORPUS_EXISTS_PREFIX}{_slugify(display_name)}"


def _corpus_resource_key(display_name: str) -> str:
  return f"{CORPUS_RESOURCE_PREFIX}{_slugify(display_name)}"


def _set_current_corpus(
    tool_context: ToolContext,
    *,
    display_name: str,
    resource_name: str,
) -> None:
  tool_context.state[CURRENT_CORPUS_KEY] = display_name
  tool_context.state[CURRENT_CORPUS_RESOURCE_KEY] = resource_name
  tool_context.state[_corpus_exists_key(display_name)] = True
  tool_context.state[_corpus_resource_key(display_name)] = resource_name


def _clear_current_corpus(tool_context: ToolContext) -> None:
  tool_context.state.update({
      CURRENT_CORPUS_KEY: None,
      CURRENT_CORPUS_RESOURCE_KEY: None,
  })


def _remove_corpus_state(
    tool_context: ToolContext,
    *,
    display_name: str,
) -> None:
  tool_context.state.update({
      _corpus_exists_key(display_name): False,
      _corpus_resource_key(display_name): None,
  })
  if tool_context.state.get(CURRENT_CORPUS_KEY) == display_name:
    _clear_current_corpus(tool_context)


def _describe_corpus(
    corpus: Any,
    *,
    current_corpus: Optional[str],
) -> dict[str, Any]:
  return {
    "name": corpus.display_name,
    "description": getattr(corpus, "description", "") or "",
    "is_current": corpus.display_name == current_corpus,
  }


def _describe_file(index: int, rag_file: Any) -> dict[str, Any]:
  return {
    "document_number": index,
    "display_name": getattr(rag_file, "display_name", "") or "",
    "description": getattr(rag_file, "description", "") or "",
  }


async def _list_corpora() -> list[Any]:
  return await asyncio.to_thread(lambda: list(rag.list_corpora()))


async def _get_corpus(resource_name: str) -> Any:
  return await asyncio.to_thread(rag.get_corpus, name=resource_name)


async def _list_files(resource_name: str) -> list[Any]:
  return await asyncio.to_thread(
      lambda: list(rag.list_files(corpus_name=resource_name))
  )


async def _resolve_corpus(
    tool_context: ToolContext,
    corpus_name: Optional[str],
) -> tuple[str, str]:
  candidate_name = corpus_name or tool_context.state.get(CURRENT_CORPUS_KEY)
  candidate_resource = tool_context.state.get(CURRENT_CORPUS_RESOURCE_KEY)

  if corpus_name and _is_resource_name(corpus_name):
    corpus = await _get_corpus(corpus_name)
    _set_current_corpus(
        tool_context,
        display_name=corpus.display_name,
        resource_name=corpus.name,
    )
    return corpus.display_name, corpus.name

  if candidate_name:
    remembered_resource = tool_context.state.get(
        _corpus_resource_key(candidate_name)
    )
    if remembered_resource:
      corpus = await _get_corpus(remembered_resource)
      _set_current_corpus(
          tool_context,
          display_name=corpus.display_name,
          resource_name=corpus.name,
      )
      return corpus.display_name, corpus.name

    corpora = await _list_corpora()
    for corpus in corpora:
      if corpus.display_name.lower() == candidate_name.lower():
        _set_current_corpus(
            tool_context,
            display_name=corpus.display_name,
            resource_name=corpus.name,
        )
        return corpus.display_name, corpus.name

  if candidate_resource:
    corpus = await _get_corpus(candidate_resource)
    _set_current_corpus(
        tool_context,
        display_name=corpus.display_name,
        resource_name=corpus.name,
    )
    return corpus.display_name, corpus.name

  raise ValueError(
      "No corpus was specified and there is no active corpus in session state."
  )


def _select_file(
    *,
    files: Sequence[Any],
    document_name: Optional[str],
    document_number: Optional[int],
) -> Any:
  if document_number is not None:
    if document_number <= 0:
      raise ValueError("document_number must be greater than zero.")
    if document_number > len(files):
      raise ValueError(
          "document_number does not match any document in the active corpus."
      )
    return files[document_number - 1]

  if not document_name or not document_name.strip():
    raise ValueError(
        "Provide document_name or document_number to identify the document."
    )

  matching_files = [
    rag_file
    for rag_file in files
    if getattr(rag_file, "display_name", "").lower()
    == document_name.strip().lower()
  ]

  if not matching_files:
    raise ValueError("No document matched the supplied document_name.")
  if len(matching_files) > 1:
    raise ValueError(
        "Multiple documents share that display name. Use document_number from "
        "get_corpus_info to disambiguate."
    )
  return matching_files[0]


async def list_corpora(tool_context: ToolContext) -> dict[str, Any]:
  """Lists available Vertex AI RAG corpora without exposing resource names."""
  try:
    corpora = await _list_corpora()
  except (GoogleAPICallError, RetryError, ValueError) as err:
    LOGGER.exception("Failed to list corpora.")
    return _error(f"Unable to list corpora: {err}")

  current_corpus = tool_context.state.get(CURRENT_CORPUS_KEY)
  data = {
    "corpora": [
      _describe_corpus(corpus, current_corpus=current_corpus)
      for corpus in corpora
    ],
    "current_corpus": current_corpus,
  }
  return _success(f"Found {len(corpora)} corpora.", data)


async def create_corpus(
    corpus_name: str,
    description: str = "",
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
  """Creates a corpus or reuses an existing display name as the active corpus.
  """
  if tool_context is None:
    return _error("Tool context is required.")
  if not corpus_name or not corpus_name.strip():
    return _error("corpus_name is required.")

  normalized_name = corpus_name.strip()
  try:
    corpora = await _list_corpora()
    for corpus in corpora:
      if corpus.display_name.lower() == normalized_name.lower():
        _set_current_corpus(
            tool_context,
            display_name=corpus.display_name,
            resource_name=corpus.name,
        )
        return _success(
            f"Using existing corpus '{corpus.display_name}'.",
            {
              "corpus": _describe_corpus(
                  corpus,
                  current_corpus=corpus.display_name,
              )
            },
        )

    backend_config = rag.RagVectorDbConfig(
        rag_embedding_model_config=rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=get_embedding_model()
            )
        )
    )
    created_corpus = await asyncio.to_thread(
        rag.create_corpus,
        display_name=normalized_name,
        description=description.strip(),
        backend_config=backend_config,
    )
  except (GoogleAPICallError, RetryError, ValueError) as err:
    LOGGER.exception("Failed to create corpus.")
    return _error(f"Unable to create corpus '{normalized_name}': {err}")

  _set_current_corpus(
      tool_context,
      display_name=created_corpus.display_name,
      resource_name=created_corpus.name,
  )
  return _success(
      f"Created corpus '{created_corpus.display_name}'.",
      {
        "corpus": _describe_corpus(
            created_corpus,
            current_corpus=created_corpus.display_name,
        )
      },
  )


async def add_data(
    paths: Sequence[str],
    corpus_name: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
  """Imports Google Drive, Docs, or GCS content into the selected corpus."""
  if tool_context is None:
    return _error("Tool context is required.")
  try:
    display_name, resource_name = await _resolve_corpus(
        tool_context,
        corpus_name,
    )
    normalized_paths = _normalize_paths(paths)
    selected_chunk_size = (
      chunk_size if chunk_size is not None else get_default_chunk_size()
    )
    selected_chunk_overlap = (
      chunk_overlap
      if chunk_overlap is not None
      else get_default_chunk_overlap()
    )
    if selected_chunk_overlap >= selected_chunk_size:
      raise ValueError(
          "chunk_overlap must be smaller than chunk_size."
      )

    response = await asyncio.to_thread(
        rag.import_files,
        corpus_name=resource_name,
        paths=normalized_paths,
        transformation_config=rag.TransformationConfig(
            rag.ChunkingConfig(
                chunk_size=selected_chunk_size,
                chunk_overlap=selected_chunk_overlap,
            )
        ),
    )
  except (GoogleAPICallError, RetryError, ValueError) as err:
    LOGGER.exception("Failed to import data.")
    return _error(f"Unable to add data: {err}")

  _set_current_corpus(
      tool_context,
      display_name=display_name,
      resource_name=resource_name,
  )
  imported_count = getattr(response, "imported_rag_files_count", 0)
  failed_count = getattr(response, "failed_rag_files_count", 0)
  skipped_count = getattr(response, "skipped_rag_files_count", 0)
  partial_failures_path = (
      getattr(response, "partial_failures_gcs_path", "") or ""
  )

  data = {
    "corpus": display_name,
    "imported_count": imported_count,
    "failed_count": failed_count,
    "skipped_count": skipped_count,
    "paths": normalized_paths,
  }
  if partial_failures_path:
    data["partial_failures_path"] = partial_failures_path

  if imported_count <= 0:
    summary_parts = []
    if failed_count:
      summary_parts.append(f"{failed_count} failed")
    if skipped_count:
      summary_parts.append(f"{skipped_count} skipped")
    summary = ", ".join(summary_parts) or "no documents were imported"
    message = (
        f"No documents were imported into '{display_name}' ({summary}). "
        "Verify the source path, service-account access, and source format."
    )
    return _error(message, data)

  message = f"Imported {imported_count} document(s) into '{display_name}'."
  if failed_count or skipped_count:
    summary_parts = []
    if failed_count:
      summary_parts.append(f"{failed_count} failed")
    if skipped_count:
      summary_parts.append(f"{skipped_count} skipped")
    message = f"{message} {', '.join(summary_parts).capitalize()}."

  return _success(message, data)


async def get_corpus_info(
    corpus_name: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
  """Returns public corpus details and numbered documents for follow-up tools.
  """
  if tool_context is None:
    return _error("Tool context is required.")
  try:
    display_name, resource_name = await _resolve_corpus(
        tool_context,
        corpus_name,
    )
    corpus = await _get_corpus(resource_name)
    files = await _list_files(resource_name)
  except (GoogleAPICallError, RetryError, ValueError) as err:
    LOGGER.exception("Failed to get corpus info.")
    return _error(f"Unable to get corpus information: {err}")

  _set_current_corpus(
      tool_context,
      display_name=display_name,
      resource_name=resource_name,
  )
  return _success(
      f"Loaded corpus information for '{display_name}'.",
      {
        "corpus": _describe_corpus(corpus, current_corpus=display_name),
        "document_count": len(files),
        "documents": [
          _describe_file(index, rag_file)
          for index, rag_file in enumerate(files, start=1)
        ],
      },
  )


async def delete_document(
    document_name: Optional[str] = None,
    document_number: Optional[int] = None,
    corpus_name: Optional[str] = None,
    confirm: bool = False,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
  """Deletes a numbered or named document after explicit confirmation."""
  if tool_context is None:
    return _error("Tool context is required.")
  if not confirm:
    return _error(
        "delete_document requires confirm=True before any deletion is allowed."
    )

  try:
    display_name, resource_name = await _resolve_corpus(
        tool_context,
        corpus_name,
    )
    files = await _list_files(resource_name)
    rag_file = _select_file(
        files=files,
        document_name=document_name,
        document_number=document_number,
    )
    await asyncio.to_thread(rag.delete_file, name=rag_file.name)
  except (GoogleAPICallError, RetryError, ValueError) as err:
    LOGGER.exception("Failed to delete document.")
    return _error(f"Unable to delete document: {err}")

  _set_current_corpus(
      tool_context,
      display_name=display_name,
      resource_name=resource_name,
  )
  display_value = getattr(rag_file, "display_name", "") or "selected document"
  return _success(
      f"Deleted '{display_value}' from '{display_name}'.",
      {"corpus": display_name, "document_name": display_value},
  )


async def delete_corpus(
    corpus_name: Optional[str] = None,
    confirm: bool = False,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
  """Deletes a corpus after explicit confirmation and clears active state."""
  if tool_context is None:
    return _error("Tool context is required.")
  if not confirm:
    return _error(
        "delete_corpus requires confirm=True before any deletion is allowed."
    )

  try:
    display_name, resource_name = await _resolve_corpus(
        tool_context,
        corpus_name,
    )
    await asyncio.to_thread(rag.delete_corpus, name=resource_name)
  except (GoogleAPICallError, RetryError, ValueError) as err:
    LOGGER.exception("Failed to delete corpus.")
    return _error(f"Unable to delete corpus: {err}")

  _remove_corpus_state(tool_context, display_name=display_name)
  return _success(
      f"Deleted corpus '{display_name}'.",
      {"corpus": display_name},
  )


async def rag_query(
    query: str,
    corpus_name: Optional[str] = None,
    top_k: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict[str, Any]:
  """Retrieves relevant RAG contexts for the active corpus and query."""
  if tool_context is None:
    return _error("Tool context is required.")
  if not query or not query.strip():
    return _error("query is required.")

  try:
    display_name, resource_name = await _resolve_corpus(
        tool_context,
        corpus_name,
    )
    selected_top_k = (
      top_k if top_k is not None else get_default_top_k()
    )
    selected_distance_threshold = (
      distance_threshold
      if distance_threshold is not None
      else get_default_distance_threshold()
    )
    if selected_top_k <= 0:
      raise ValueError("top_k must be greater than zero.")
    if selected_distance_threshold < 0:
      raise ValueError("distance_threshold must be non-negative.")

    response = await asyncio.to_thread(
        rag.retrieval_query,
        rag_resources=[rag.RagResource(rag_corpus=resource_name)],
        text=query.strip(),
        rag_retrieval_config=rag.RagRetrievalConfig(
            top_k=selected_top_k,
            filter=rag.Filter(
                vector_distance_threshold=selected_distance_threshold
            ),
        ),
    )
  except (GoogleAPICallError, RetryError, ValueError) as err:
    LOGGER.exception("Failed to run RAG query.")
    return _error(f"Unable to query corpus: {err}")

  _set_current_corpus(
      tool_context,
      display_name=display_name,
      resource_name=resource_name,
  )

  raw_contexts = (
    getattr(getattr(response, "contexts", None), "contexts", []) or []
  )
  contexts = [
    {
      "document_name": getattr(context, "source_display_name", "") or "",
      "text": getattr(context, "text", "") or "",
      "distance": getattr(context, "distance", None),
    }
    for context in raw_contexts
  ]

  message = (
    f"Retrieved {len(contexts)} context chunk(s) from '{display_name}'."
    if contexts
    else f"No matching context was found in '{display_name}'."
  )
  return _success(
      message,
      {
        "corpus": display_name,
        "query": query.strip(),
        "top_k": selected_top_k,
        "distance_threshold": selected_distance_threshold,
        "contexts": contexts,
      },
  )
