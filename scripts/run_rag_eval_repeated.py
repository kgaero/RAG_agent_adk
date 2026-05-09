#!/usr/bin/env python3
"""Runs repeated local RAG evals and plots aggregate uncertainty."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from evals.rag_eval import aggregate_summaries
from evals.rag_eval import write_aggregate_json_report
from evals.rag_eval import write_aggregate_markdown_report
from evals.rag_eval import write_aggregate_metric_svg_plot
from run_rag_eval import _run_eval


PLOTTED_METRICS = [
  "pass_rate",
  "hit_rate_at_k",
  "recall_at_k",
  "context_precision_at_k",
  "answer_correctness_rate",
  "behavior_pass_rate",
]

COST_LATENCY_METRICS = [
  "latency_seconds_avg",
  "latency_seconds_max",
  "llm_call_count_proxy",
  "retrieved_context_count",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parses CLI arguments for repeated local RAG evaluation."""
  parser = argparse.ArgumentParser(
    description=(
      "Run the deterministic local RAG eval repeatedly and generate "
      "aggregate reports plus an SVG uncertainty plot."
    ),
  )
  parser.add_argument(
    "--dataset",
    default=str(PROJECT_ROOT / "evals" / "golden_qna.jsonl"),
    help="Path to the JSONL golden Q&A dataset.",
  )
  parser.add_argument(
    "--corpus-name",
    required=True,
    help="Display name of the existing Vertex RAG corpus to evaluate.",
  )
  parser.add_argument(
    "--runs",
    type=int,
    default=3,
    help="Number of repeated eval runs. Default: 3.",
  )
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional maximum number of eval cases per run.",
  )
  parser.add_argument(
    "--distance-threshold",
    type=float,
    default=None,
    help="Optional retrieval distance threshold override.",
  )
  parser.add_argument(
    "--reports-dir",
    default=str(PROJECT_ROOT / "evals" / "reports"),
    help="Directory for per-run and aggregate reports.",
  )
  parser.add_argument(
    "--output-prefix",
    default=None,
    help="Optional filename prefix. Defaults to repeated_rag_eval_<timestamp>.",
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
  parser.add_argument(
    "--delay-between-runs",
    type=float,
    default=0.0,
    help="Optional pacing delay after each completed run.",
  )
  return parser.parse_args(argv)


async def _run_repeated_eval(args: argparse.Namespace) -> int:
  """Runs repeated evals, aggregates summaries, and writes plot/report files."""
  if args.runs <= 0:
    raise ValueError("--runs must be greater than zero.")

  reports_dir = Path(args.reports_dir)
  reports_dir.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
  prefix = args.output_prefix or f"repeated_rag_eval_{timestamp}"
  run_reports: list[dict[str, Any]] = []
  failed_runs = 0

  for run_index in range(1, args.runs + 1):
    run_prefix = f"{prefix}_run_{run_index:02d}"
    markdown_path = reports_dir / f"{run_prefix}.md"
    json_path = reports_dir / f"{run_prefix}.json"
    if json_path.exists():
      print(
        f"=== Repeated eval run {run_index}/{args.runs} already exists; "
        "reusing report ===",
        flush=True,
      )
      run_report = _read_json_report(json_path)
      run_report["report_paths"] = {
        "markdown": str(markdown_path),
        "json": str(json_path),
      }
      run_report["run_index"] = run_index
      run_report["exit_code"] = 0 if _report_passed(run_report) else 1
      if run_report["exit_code"] != 0:
        failed_runs += 1
      run_reports.append(run_report)
      continue

    print(f"=== Repeated eval run {run_index}/{args.runs} ===", flush=True)
    run_args = argparse.Namespace(
      dataset=args.dataset,
      corpus_name=args.corpus_name,
      limit=args.limit,
      output=str(markdown_path),
      json_output=str(json_path),
      distance_threshold=args.distance_threshold,
      case_retries=args.case_retries,
      retry_delay_seconds=args.retry_delay_seconds,
      delay_between_cases=args.delay_between_cases,
    )
    exit_code = await _run_eval(run_args)
    if exit_code != 0:
      failed_runs += 1
    run_report = _read_json_report(json_path)
    run_report["report_paths"] = {
      "markdown": str(markdown_path),
      "json": str(json_path),
    }
    run_report["run_index"] = run_index
    run_report["exit_code"] = exit_code
    run_reports.append(run_report)
    if args.delay_between_runs > 0 and run_index < args.runs:
      await asyncio.sleep(args.delay_between_runs)

  aggregate = aggregate_summaries([
    run_report["summary"] for run_report in run_reports
  ])
  aggregate_markdown_path = reports_dir / f"{prefix}_aggregate.md"
  aggregate_json_path = reports_dir / f"{prefix}_aggregate.json"
  plot_path = reports_dir / f"{prefix}_metrics.svg"
  cost_latency_plot_path = reports_dir / f"{prefix}_cost_latency.svg"
  metadata = _aggregate_metadata(args, timestamp, failed_runs)

  write_aggregate_metric_svg_plot(
    plot_path,
    aggregate=aggregate,
    metric_names=PLOTTED_METRICS,
    title="Repeated RAG Eval Score Metrics With 95% CI",
  )
  write_aggregate_metric_svg_plot(
    cost_latency_plot_path,
    aggregate=aggregate,
    metric_names=COST_LATENCY_METRICS,
    title="Repeated RAG Eval Cost And Latency Proxies With 95% CI",
  )
  write_aggregate_markdown_report(
    aggregate_markdown_path,
    run_metadata=metadata,
    aggregate=aggregate,
    plot_path=plot_path,
    cost_latency_plot_path=cost_latency_plot_path,
  )
  write_aggregate_json_report(
    aggregate_json_path,
    run_metadata=metadata,
    run_reports=run_reports,
    aggregate=aggregate,
    plot_path=plot_path,
    cost_latency_plot_path=cost_latency_plot_path,
  )
  print(f"Wrote aggregate Markdown report: {aggregate_markdown_path}")
  print(f"Wrote aggregate JSON report: {aggregate_json_path}")
  print(f"Wrote SVG metrics plot: {plot_path}")
  print(f"Wrote SVG cost/latency plot: {cost_latency_plot_path}")
  return 0 if failed_runs == 0 else 1


def _read_json_report(path: Path) -> dict[str, Any]:
  """Reads one single-run JSON report."""
  return json.loads(path.read_text(encoding="utf-8"))


def _report_passed(run_report: dict[str, Any]) -> bool:
  """Returns whether a loaded run report passed all cases."""
  results = run_report.get("results", [])
  return bool(results) and all(result.get("passed") for result in results)


def _aggregate_metadata(
    args: argparse.Namespace,
    timestamp: str,
    failed_runs: int,
) -> dict[str, Any]:
  """Builds metadata for repeated-run aggregate reports."""
  return {
    "timestamp_utc": timestamp,
    "agent": "local agents.ragagent.root_agent",
    "dataset": args.dataset,
    "corpus_name": args.corpus_name,
    "runs": args.runs,
    "failed_runs": failed_runs,
    "limit": args.limit,
    "distance_threshold": args.distance_threshold,
    "case_retries": args.case_retries,
    "retry_delay_seconds": args.retry_delay_seconds,
    "delay_between_cases": args.delay_between_cases,
    "delay_between_runs": args.delay_between_runs,
    "evaluator": "deterministic local scoring; no judge LLM",
  }


def main(argv: list[str] | None = None) -> None:
  """Runs the repeated local RAG evaluation CLI."""
  raise SystemExit(asyncio.run(_run_repeated_eval(_parse_args(argv))))


if __name__ == "__main__":
  main()
