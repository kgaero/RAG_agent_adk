"""Runs deterministic local RAG evals against the ADK RAG agent."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = PROJECT_ROOT / "agents"

if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))
if str(AGENTS_ROOT) not in sys.path:
  sys.path.insert(0, str(AGENTS_ROOT))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from evals.rag_eval import GoldenCase
from evals.rag_eval import RetrievedContext
from evals.rag_eval import default_report_paths
from evals.rag_eval import load_golden_cases
from evals.rag_eval import score_case
from evals.rag_eval import write_json_report
from evals.rag_eval import write_markdown_report
from ragagent.bootstrap import get_default_chunk_overlap
from ragagent.bootstrap import get_default_chunk_size
from ragagent.bootstrap import get_default_distance_threshold
from ragagent.bootstrap import get_default_top_k


APP_NAME = "ragagent-local-eval"
USER_ID = "eval-user"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments for local RAG evaluation."""
  parser = argparse.ArgumentParser(
    description="Run deterministic local RAG evals against the ADK agent.",
  )
  parser.add_argument(
    "--dataset",
    default=str(PROJECT_ROOT / "evals" / "golden_qna.jsonl"),
    help="Path to the JSONL golden Q&A dataset.",
  )
  parser.add_argument(
    "--corpus-name",
    required=True,
    help=(
      "Display name of the existing Vertex RAG corpus that already contains "
      "the PRD source document."
    ),
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional maximum number of eval cases to run.",
  )
  parser.add_argument(
    "--output",
    default=None,
    help="Markdown report path. Defaults to evals/reports/rag_eval_*.md.",
  )
  parser.add_argument(
    "--json-output",
    default=None,
    help="JSON report path. Defaults to evals/reports/rag_eval_*.json.",
  )
  parser.add_argument(
    "--distance-threshold",
    type=float,
    default=None,
    help=(
      "Optional retrieval distance threshold override for eval questions. "
      "Defaults to DEFAULT_DISTANCE_THRESHOLD from .env."
    ),
  )
  parser.add_argument(
    "--case-retries",
    type=int,
    default=2,
    help="Retry count for transient per-case Vertex errors. Default: 2.",
  )
  parser.add_argument(
    "--retry-delay-seconds",
    type=float,
    default=60.0,
    help="Initial delay before retrying transient per-case errors.",
  )
  parser.add_argument(
    "--delay-between-cases",
    type=float,
    default=0.0,
    help="Optional pacing delay after each eval case.",
  )
  return parser.parse_args(argv)


async def _run_eval(args: argparse.Namespace) -> int:
  """Runs all selected eval cases and writes reports."""
  dataset_path = Path(args.dataset)
  cases = load_golden_cases(dataset_path)
  if args.limit is not None:
    if args.limit <= 0:
      raise ValueError("--limit must be greater than zero.")
    cases = cases[:args.limit]

  markdown_path, json_path = _resolve_report_paths(args)
  session_service = InMemorySessionService()
  from ragagent import agent as rag_agent_module

  runner = Runner(
    agent=rag_agent_module.root_agent,
    app_name=APP_NAME,
    session_service=session_service,
  )

  results = []
  for index, case in enumerate(cases, start=1):
    session_id = f"eval-{case.id}"
    await session_service.create_session(
      app_name=APP_NAME,
      user_id=USER_ID,
      session_id=session_id,
      state={"current_corpus": args.corpus_name, "eval_mode": True},
    )
    print(f"[{index}/{len(cases)}] {case.id}: {case.question}", flush=True)
    answer, contexts, latency = await _run_case_with_retries(
      runner,
      session_id=session_id,
      corpus_name=args.corpus_name,
      case=case,
      distance_threshold=args.distance_threshold,
      case_retries=args.case_retries,
      retry_delay_seconds=args.retry_delay_seconds,
    )
    results.append(score_case(
      case,
      answer=answer,
      contexts=contexts,
      latency_seconds=latency,
    ))
    if args.delay_between_cases > 0 and index < len(cases):
      await asyncio.sleep(args.delay_between_cases)

  metadata = _run_metadata(args, dataset_path)
  write_markdown_report(markdown_path, run_metadata=metadata, results=results)
  write_json_report(json_path, run_metadata=metadata, results=results)
  print(f"Wrote Markdown report: {markdown_path}")
  print(f"Wrote JSON report: {json_path}")
  return 0 if all(result.passed for result in results) else 1


async def _run_case_with_retries(
    runner: Runner,
    *,
    session_id: str,
    corpus_name: str,
    case: GoldenCase,
    distance_threshold: float | None = None,
    case_retries: int = 2,
    retry_delay_seconds: float = 60.0,
) -> tuple[str, list[RetrievedContext], float]:
  """Runs one eval case with retries for transient quota failures."""
  if case_retries < 0:
    raise ValueError("--case-retries must be zero or greater.")
  if retry_delay_seconds < 0:
    raise ValueError("--retry-delay-seconds must be zero or greater.")

  attempt = 0
  while True:
    try:
      return await _run_case(
        runner,
        session_id=session_id,
        corpus_name=corpus_name,
        case=case,
        distance_threshold=distance_threshold,
      )
    except Exception as err:
      if not _is_retryable_resource_error(err) or attempt >= case_retries:
        raise
      delay = retry_delay_seconds * (2 ** attempt)
      print(
        "Transient Vertex quota error for "
        f"{case.id}; retrying in {delay:.1f}s "
        f"({attempt + 1}/{case_retries}).",
        flush=True,
      )
      await asyncio.sleep(delay)
      attempt += 1


async def _run_case(
    runner: Runner,
    *,
    session_id: str,
    corpus_name: str,
    case: GoldenCase,
    distance_threshold: float | None = None,
) -> tuple[str, list[RetrievedContext], float]:
  """Runs one eval case through the local ADK runner."""
  prompt = (
    f"{case.question}\n\n"
    f"Use the existing corpus named '{corpus_name}'. "
    "For answerable questions about corpus content, call rag_query before "
    "answering."
  )
  if case.expected_behavior == "answer":
    prompt += _retrieval_hint(case)
  if distance_threshold is not None:
    prompt += (
      " When calling rag_query, pass "
      f"distance_threshold={distance_threshold}."
    )
  user_content = types.Content(
    role="user",
    parts=[types.Part.from_text(text=prompt)],
  )
  started_at = time.perf_counter()
  final_answer = ""
  contexts: list[RetrievedContext] = []

  async for event in runner.run_async(
      user_id=USER_ID,
      session_id=session_id,
      new_message=user_content,
  ):
    contexts.extend(_extract_contexts(event))
    if event.is_final_response() and event.content and event.content.parts:
      final_answer = "\n".join(
        part.text.strip()
        for part in event.content.parts
        if part.text and part.text.strip()
      )

  return final_answer, contexts, time.perf_counter() - started_at


def _is_retryable_resource_error(err: Exception) -> bool:
  """Returns whether an exception looks like a transient quota failure."""
  error_text = f"{type(err).__name__}: {err}"
  return (
    "RESOURCE_EXHAUSTED" in error_text
    or "429" in error_text
    or "_ResourceExhaustedError" in error_text
  )


def _extract_contexts(event: Any) -> list[RetrievedContext]:
  """Extracts rag_query contexts from ADK function response events."""
  contexts: list[RetrievedContext] = []
  content = getattr(event, "content", None)
  for part in getattr(content, "parts", []) or []:
    function_response = getattr(part, "function_response", None)
    if function_response is None:
      continue
    if getattr(function_response, "name", "") != "rag_query":
      continue
    response = getattr(function_response, "response", None) or {}
    if not isinstance(response, dict):
      continue
    data = response.get("data") or {}
    for raw_context in data.get("contexts") or []:
      if not isinstance(raw_context, dict):
        continue
      contexts.append(RetrievedContext(
        document_name=str(raw_context.get("document_name") or ""),
        text=str(raw_context.get("text") or ""),
        distance=raw_context.get("distance"),
      ))
  return contexts


def _retrieval_hint(case: GoldenCase) -> str:
  """Builds a retrieval-only hint without exposing the expected answer."""
  if not case.minimum_required_terms:
    return ""
  return (
    " Search for these topic terms: "
    f"{', '.join(case.minimum_required_terms)}."
  )


def _resolve_report_paths(args: argparse.Namespace) -> tuple[Path, Path]:
  """Resolves explicit or timestamped report output paths."""
  default_markdown, default_json = default_report_paths(
    PROJECT_ROOT / "evals" / "reports"
  )
  markdown_path = Path(args.output) if args.output else default_markdown
  json_path = Path(args.json_output) if args.json_output else default_json
  return markdown_path, json_path


def _run_metadata(
    args: argparse.Namespace,
    dataset_path: Path,
) -> dict[str, Any]:
  """Builds report metadata for an eval run."""
  return {
    "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    "agent": "local agents.ragagent.root_agent",
    "dataset": str(dataset_path),
    "corpus_name": args.corpus_name,
    "chunk_size": get_default_chunk_size(),
    "chunk_overlap": get_default_chunk_overlap(),
    "top_k": get_default_top_k(),
    "distance_threshold": _selected_distance_threshold(args),
    "evaluator": "deterministic local scoring; no judge LLM",
  }


def _selected_distance_threshold(args: argparse.Namespace) -> float:
  """Returns the eval threshold setting recorded in report metadata."""
  if args.distance_threshold is not None:
    return args.distance_threshold
  return get_default_distance_threshold()


def main(argv: list[str] | None = None) -> None:
  """Runs the local RAG evaluation CLI."""
  raise SystemExit(asyncio.run(_run_eval(_parse_args(argv))))


if __name__ == "__main__":
  main()
