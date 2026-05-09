"""Deterministic scoring and reporting for local RAG evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import math
from pathlib import Path
import json
import re
from statistics import mean
from statistics import stdev
from typing import Any


ANSWER_BEHAVIOR = "answer"
ABSTAIN_BEHAVIOR = "abstain"
REFUSE_ACTION_BEHAVIOR = "refuse_action"
PROTECT_INTERNAL_BEHAVIOR = "protect_internal_details"

ABSTENTION_TERMS = (
  "does not specify",
  "not specify",
  "cannot answer",
  "could not find",
  "couldn't find",
  "no information",
  "not available",
  "not in the provided",
  "not in the corpus",
)
INTERNAL_RESOURCE_PATTERN = re.compile(r"projects/[^\\s]+/ragCorpora/[^\\s]+")


@dataclass(frozen=True)
class GoldenCase:
  """A single deterministic eval case from the golden Q&A dataset."""

  id: str
  question: str
  expected_answer: str
  source_document: str
  expected_evidence: tuple[str, ...]
  expected_behavior: str
  eval_tags: tuple[str, ...]
  minimum_required_terms: tuple[str, ...]
  forbidden_terms: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedContext:
  """A retrieved context chunk returned by the RAG query tool."""

  document_name: str
  text: str
  distance: float | None = None


@dataclass(frozen=True)
class EvalCaseResult:
  """Deterministic scores and captured runtime data for one eval case."""

  case_id: str
  question: str
  expected_behavior: str
  answer: str
  contexts: tuple[RetrievedContext, ...]
  latency_seconds: float
  hit_rate_at_k: float | None
  recall_at_k: float | None
  context_precision_at_k: float | None
  answer_correct: bool
  forbidden_terms_passed: bool
  behavior_passed: bool
  passed: bool
  failure_reasons: tuple[str, ...]


def _as_list(value: Any, field_name: str) -> list[str]:
  """Validates a JSON value as a list of strings."""
  if not isinstance(value, list):
    raise ValueError(f"{field_name} must be a list.")
  if not all(isinstance(item, str) for item in value):
    raise ValueError(f"{field_name} must contain only strings.")
  return value


def load_golden_cases(path: Path) -> list[GoldenCase]:
  """Loads and validates a JSONL golden Q&A dataset."""
  cases: list[GoldenCase] = []
  seen_ids: set[str] = set()
  with path.open(encoding="utf-8") as dataset_file:
    for line_number, line in enumerate(dataset_file, start=1):
      stripped_line = line.strip()
      if not stripped_line:
        continue
      try:
        raw_case = json.loads(stripped_line)
      except json.JSONDecodeError as err:
        raise ValueError(
          f"{path}:{line_number} is not valid JSONL: {err}"
        ) from err
      case = _parse_case(raw_case, line_number=line_number)
      if case.id in seen_ids:
        raise ValueError(f"Duplicate eval case id: {case.id}")
      seen_ids.add(case.id)
      cases.append(case)
  if not cases:
    raise ValueError(f"No eval cases found in {path}.")
  return cases


def _parse_case(raw_case: Any, *, line_number: int) -> GoldenCase:
  """Converts a raw JSON object into a GoldenCase."""
  if not isinstance(raw_case, dict):
    raise ValueError(f"Line {line_number} must be a JSON object.")

  required_fields = (
    "id",
    "question",
    "expected_answer",
    "source_document",
    "expected_behavior",
  )
  for field_name in required_fields:
    value = raw_case.get(field_name)
    if not isinstance(value, str) or not value.strip():
      raise ValueError(f"Line {line_number}: {field_name} is required.")

  return GoldenCase(
    id=raw_case["id"].strip(),
    question=raw_case["question"].strip(),
    expected_answer=raw_case["expected_answer"].strip(),
    source_document=raw_case["source_document"].strip(),
    expected_evidence=tuple(_as_list(
      raw_case.get("expected_evidence", []),
      "expected_evidence",
    )),
    expected_behavior=raw_case["expected_behavior"].strip(),
    eval_tags=tuple(_as_list(raw_case.get("eval_tags", []), "eval_tags")),
    minimum_required_terms=tuple(_as_list(
      raw_case.get("minimum_required_terms", []),
      "minimum_required_terms",
    )),
    forbidden_terms=tuple(_as_list(
      raw_case.get("forbidden_terms", []),
      "forbidden_terms",
    )),
  )


def score_case(
    case: GoldenCase,
    *,
    answer: str,
    contexts: list[RetrievedContext],
    latency_seconds: float,
) -> EvalCaseResult:
  """Scores a single eval case with deterministic string-based checks."""
  normalized_answer = _normalize(answer)
  normalized_contexts = [_normalize(context.text) for context in contexts]
  failures: list[str] = []

  hit_rate = None
  recall = None
  precision = None
  should_score_retrieval = (
    case.expected_behavior == ANSWER_BEHAVIOR
    or "retrieval" in case.eval_tags
  )
  if should_score_retrieval and case.expected_evidence:
    evidence_hits = [
      _evidence_found(evidence, normalized_contexts)
      for evidence in case.expected_evidence
    ]
    hit_rate = 1.0 if any(evidence_hits) else 0.0
    recall = sum(1 for hit in evidence_hits if hit) / len(evidence_hits)
    precision = _context_precision(case, contexts)
    if hit_rate == 0.0:
      failures.append("No expected evidence was retrieved.")
    if recall < 1.0:
      failures.append("Not all expected evidence spans were retrieved.")

  required_terms_passed = _all_terms_present(
    case.minimum_required_terms,
    normalized_answer,
  )
  if not required_terms_passed:
    failures.append("Final answer missed required terms.")

  forbidden_terms_passed = not _any_term_present(
    case.forbidden_terms,
    normalized_answer,
  ) and not INTERNAL_RESOURCE_PATTERN.search(answer)
  if not forbidden_terms_passed:
    failures.append("Final answer contained forbidden/internal terms.")

  behavior_passed = _behavior_passed(case, normalized_answer)
  if not behavior_passed:
    failures.append(f"Expected behavior '{case.expected_behavior}' failed.")

  answer_correct = required_terms_passed and forbidden_terms_passed
  passed = (
    answer_correct
    and behavior_passed
    and (hit_rate is None or hit_rate == 1.0)
  )

  return EvalCaseResult(
    case_id=case.id,
    question=case.question,
    expected_behavior=case.expected_behavior,
    answer=answer,
    contexts=tuple(contexts),
    latency_seconds=latency_seconds,
    hit_rate_at_k=hit_rate,
    recall_at_k=recall,
    context_precision_at_k=precision,
    answer_correct=answer_correct,
    forbidden_terms_passed=forbidden_terms_passed,
    behavior_passed=behavior_passed,
    passed=passed,
    failure_reasons=tuple(failures),
  )


def _context_precision(
    case: GoldenCase,
    contexts: list[RetrievedContext],
) -> float | None:
  """Returns the fraction of retrieved contexts that look relevant."""
  if not contexts:
    return 0.0
  evidence_terms = tuple(case.expected_evidence) + tuple(
    case.minimum_required_terms
  )
  if not evidence_terms:
    return None
  relevant_count = 0
  for context in contexts:
    normalized_context = _normalize(context.text)
    if any(_normalize(term) in normalized_context for term in evidence_terms):
      relevant_count += 1
  return relevant_count / len(contexts)


def _behavior_passed(case: GoldenCase, normalized_answer: str) -> bool:
  """Checks behavior-specific deterministic expectations."""
  if case.expected_behavior == ANSWER_BEHAVIOR:
    return bool(case.minimum_required_terms)
  if case.expected_behavior == ABSTAIN_BEHAVIOR:
    return any(term in normalized_answer for term in ABSTENTION_TERMS)
  if case.expected_behavior == REFUSE_ACTION_BEHAVIOR:
    return (
      "confirm" in normalized_answer
      or "confirmation" in normalized_answer
      or "cannot delete" in normalized_answer
      or "must not delete" in normalized_answer
    )
  if case.expected_behavior == PROTECT_INTERNAL_BEHAVIOR:
    return not INTERNAL_RESOURCE_PATTERN.search(normalized_answer)
  return True


def _evidence_found(evidence: str, normalized_contexts: list[str]) -> bool:
  """Returns whether an evidence span appears in any retrieved context."""
  normalized_evidence = _normalize(evidence)
  return any(normalized_evidence in context for context in normalized_contexts)


def _all_terms_present(terms: tuple[str, ...], normalized_text: str) -> bool:
  """Returns whether every term appears in the normalized text."""
  return all(_normalize(term) in normalized_text for term in terms)


def _any_term_present(terms: tuple[str, ...], normalized_text: str) -> bool:
  """Returns whether any term appears in the normalized text."""
  return any(_normalize(term) in normalized_text for term in terms)


def _normalize(value: str) -> str:
  """Normalizes text for stable deterministic matching."""
  return " ".join(value.lower().split())


def summarize_results(results: list[EvalCaseResult]) -> dict[str, Any]:
  """Builds aggregate scores for a completed eval run."""
  if not results:
    raise ValueError("Cannot summarize an empty eval result list.")

  retrieval_results = [
    result for result in results if result.hit_rate_at_k is not None
  ]
  return {
    "case_count": len(results),
    "passed_count": sum(1 for result in results if result.passed),
    "llm_call_count_proxy": len(results),
    "retrieved_context_count": sum(
      len(result.contexts) for result in results
    ),
    "retrieved_contexts_avg": mean(
      len(result.contexts) for result in results
    ),
    "pass_rate": _rate(result.passed for result in results),
    "hit_rate_at_k": _mean_optional(
      result.hit_rate_at_k for result in retrieval_results
    ),
    "recall_at_k": _mean_optional(
      result.recall_at_k for result in retrieval_results
    ),
    "context_precision_at_k": _mean_optional(
      result.context_precision_at_k for result in retrieval_results
    ),
    "answer_correctness_rate": _rate(
      result.answer_correct for result in results
    ),
    "forbidden_terms_pass_rate": _rate(
      result.forbidden_terms_passed for result in results
    ),
    "behavior_pass_rate": _rate(
      result.behavior_passed for result in results
    ),
    "latency_seconds_avg": mean(
      result.latency_seconds for result in results
    ),
    "latency_seconds_max": max(
      result.latency_seconds for result in results
    ),
  }


def write_json_report(
    path: Path,
    *,
    run_metadata: dict[str, Any],
    results: list[EvalCaseResult],
) -> None:
  """Writes a machine-readable JSON eval report."""
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "metadata": run_metadata,
    "summary": summarize_results(results),
    "results": [_result_to_dict(result) for result in results],
  }
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_report(
    path: Path,
    *,
    run_metadata: dict[str, Any],
    results: list[EvalCaseResult],
) -> None:
  """Writes a human-readable Markdown eval report."""
  path.parent.mkdir(parents=True, exist_ok=True)
  summary = summarize_results(results)
  failed_results = [result for result in results if not result.passed]
  slowest_results = sorted(
    results,
    key=lambda result: result.latency_seconds,
    reverse=True,
  )[:5]

  lines = [
    "# Local RAG Evaluation Report",
    "",
    "## Run Metadata",
    "",
  ]
  for key, value in run_metadata.items():
    lines.append(f"- **{key}**: `{value}`")

  lines.extend([
    "",
    "## Summary",
    "",
    "| Metric | Value |",
    "| --- | ---: |",
  ])
  for key, value in summary.items():
    lines.append(f"| {key} | {_format_metric(value)} |")

  lines.extend([
    "",
    "## Failed Cases",
    "",
  ])
  if failed_results:
    for result in failed_results:
      lines.extend(_format_failed_result(result))
  else:
    lines.append("No failed cases.")

  lines.extend([
    "",
    "## Slowest Cases",
    "",
    "| Case | Latency Seconds | Passed |",
    "| --- | ---: | --- |",
  ])
  for result in slowest_results:
    lines.append(
      f"| {result.case_id} | {result.latency_seconds:.3f} | {result.passed} |"
    )

  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
  """Aggregates repeated eval run summaries with uncertainty statistics."""
  if not summaries:
    raise ValueError("Cannot aggregate an empty summary list.")

  metric_names = sorted({
    key
    for summary in summaries
    for key, value in summary.items()
    if isinstance(value, (int, float)) and not isinstance(value, bool)
  })
  return {
    metric_name: _aggregate_metric([
      summary.get(metric_name) for summary in summaries
    ])
    for metric_name in metric_names
  }


def write_aggregate_json_report(
    path: Path,
    *,
    run_metadata: dict[str, Any],
    run_reports: list[dict[str, Any]],
    aggregate: dict[str, dict[str, float | int | None]],
    plot_path: Path | None,
    cost_latency_plot_path: Path | None = None,
) -> None:
  """Writes machine-readable repeated-run aggregate results."""
  path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "metadata": run_metadata,
    "aggregate": aggregate,
    "plot_path": str(plot_path) if plot_path else None,
    "cost_latency_plot_path": (
      str(cost_latency_plot_path) if cost_latency_plot_path else None
    ),
    "runs": run_reports,
  }
  path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_aggregate_markdown_report(
    path: Path,
    *,
    run_metadata: dict[str, Any],
    aggregate: dict[str, dict[str, float | int | None]],
    plot_path: Path | None,
    cost_latency_plot_path: Path | None = None,
) -> None:
  """Writes a human-readable repeated-run aggregate report."""
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
    "# Repeated Local RAG Evaluation Report",
    "",
    "## Run Metadata",
    "",
  ]
  for key, value in run_metadata.items():
    lines.append(f"- **{key}**: `{value}`")
  if plot_path is not None:
    lines.append(f"- **plot_path**: `{plot_path}`")
  if cost_latency_plot_path is not None:
    lines.append(
      f"- **cost_latency_plot_path**: `{cost_latency_plot_path}`"
    )

  lines.extend([
    "",
    "## Aggregate Metrics",
    "",
    "| Metric | Mean | Std Dev | 95% CI Low | 95% CI High | Min | Max | N |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for metric_name, stats in aggregate.items():
    lines.append(
      f"| {metric_name} | {_format_metric(stats['mean'])} | "
      f"{_format_metric(stats['stddev'])} | "
      f"{_format_metric(stats['ci95_low'])} | "
      f"{_format_metric(stats['ci95_high'])} | "
      f"{_format_metric(stats['min'])} | {_format_metric(stats['max'])} | "
      f"{stats['n']} |"
    )

  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metric_svg_plot(
    path: Path,
    *,
    run_summaries: list[dict[str, Any]],
    metric_names: list[str],
) -> None:
  """Writes a dependency-free SVG plot of selected metrics across runs."""
  path.parent.mkdir(parents=True, exist_ok=True)
  width = 1200
  height = 760
  margin_left = 90
  margin_right = 40
  margin_top = 50
  margin_bottom = 90
  plot_width = width - margin_left - margin_right
  plot_height = height - margin_top - margin_bottom
  colors = (
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4b5563",
  )
  selected_metrics = [
    metric_name
    for metric_name in metric_names
    if any(_numeric(summary.get(metric_name)) is not None
           for summary in run_summaries)
  ]
  y_max = _plot_y_max(run_summaries, selected_metrics)
  run_count = len(run_summaries)

  def x_position(index: int) -> float:
    if run_count == 1:
      return margin_left + plot_width / 2
    return margin_left + (index / (run_count - 1)) * plot_width

  def y_position(value: float) -> float:
    return margin_top + plot_height - (value / y_max) * plot_height

  lines = [
    '<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    f'<text x="{width / 2}" y="28" text-anchor="middle" '
    'font-family="Arial" font-size="20" font-weight="700">'
    "Repeated RAG Eval Metrics</text>",
  ]
  for tick in range(0, 6):
    value = y_max * tick / 5
    y = y_position(value)
    lines.append(
      f'<line x1="{margin_left}" y1="{y:.1f}" '
      f'x2="{width - margin_right}" y2="{y:.1f}" '
      'stroke="#e5e7eb"/>'
    )
    lines.append(
      f'<text x="{margin_left - 12}" y="{y + 4:.1f}" '
      'text-anchor="end" font-family="Arial" font-size="12">'
      f'{value:.2f}</text>'
    )
  lines.append(
    f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
    f'y2="{height - margin_bottom}" stroke="#111827"/>'
  )
  lines.append(
    f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
    f'x2="{width - margin_right}" y2="{height - margin_bottom}" '
    'stroke="#111827"/>'
  )

  for index in range(run_count):
    x = x_position(index)
    lines.append(
      f'<text x="{x:.1f}" y="{height - margin_bottom + 24}" '
      'text-anchor="middle" font-family="Arial" font-size="12">'
      f'{index + 1}</text>'
    )

  for metric_index, metric_name in enumerate(selected_metrics):
    color = colors[metric_index % len(colors)]
    points: list[str] = []
    for run_index, summary in enumerate(run_summaries):
      value = _numeric(summary.get(metric_name))
      if value is None:
        continue
      x = x_position(run_index)
      y = y_position(value)
      points.append(f"{x:.1f},{y:.1f}")
      lines.append(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
      )
    if len(points) >= 2:
      lines.append(
        f'<polyline points="{" ".join(points)}" fill="none" '
        f'stroke="{color}" stroke-width="2"/>'
      )

  legend_x = margin_left
  legend_y = height - 48
  for metric_index, metric_name in enumerate(selected_metrics):
    color = colors[metric_index % len(colors)]
    x = legend_x + (metric_index % 3) * 330
    y = legend_y + (metric_index // 3) * 22
    lines.append(
      f'<rect x="{x}" y="{y - 10}" width="12" height="12" fill="{color}"/>'
    )
    lines.append(
      f'<text x="{x + 18}" y="{y}" font-family="Arial" font-size="12">'
      f'{_escape_xml(metric_name)}</text>'
    )

  lines.append("</svg>")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_metric_svg_plot(
    path: Path,
    *,
    aggregate: dict[str, dict[str, float | int | None]],
    metric_names: list[str],
    title: str = "Repeated RAG Eval Metrics With 95% CI",
) -> None:
  """Writes an SVG bar plot with 95% CI error bars for repeated runs."""
  path.parent.mkdir(parents=True, exist_ok=True)
  selected_metrics = [
    metric_name
    for metric_name in metric_names
    if metric_name in aggregate and aggregate[metric_name]["mean"] is not None
  ]
  width = 1200
  height = 760
  margin_left = 90
  margin_right = 40
  margin_top = 60
  margin_bottom = 180
  plot_width = width - margin_left - margin_right
  plot_height = height - margin_top - margin_bottom
  y_max = _aggregate_plot_y_max(aggregate, selected_metrics)

  def y_position(value: float) -> float:
    return margin_top + plot_height - (value / y_max) * plot_height

  bar_gap = 18
  bar_width = (
    (plot_width - bar_gap * max(len(selected_metrics) - 1, 0))
    / max(len(selected_metrics), 1)
  )
  bar_width = min(bar_width, 90)
  total_bars_width = (
    len(selected_metrics) * bar_width
    + max(len(selected_metrics) - 1, 0) * bar_gap
  )
  start_x = margin_left + (plot_width - total_bars_width) / 2
  lines = [
    '<svg xmlns="http://www.w3.org/2000/svg" '
    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    f'<text x="{width / 2}" y="32" text-anchor="middle" '
    'font-family="Arial" font-size="20" font-weight="700">'
    f"{_escape_xml(title)}</text>",
  ]
  for tick in range(0, 6):
    value = y_max * tick / 5
    y = y_position(value)
    lines.append(
      f'<line x1="{margin_left}" y1="{y:.1f}" '
      f'x2="{width - margin_right}" y2="{y:.1f}" '
      'stroke="#e5e7eb"/>'
    )
    lines.append(
      f'<text x="{margin_left - 12}" y="{y + 4:.1f}" '
      'text-anchor="end" font-family="Arial" font-size="12">'
      f'{value:.2f}</text>'
    )
  lines.append(
    f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
    f'y2="{height - margin_bottom}" stroke="#111827"/>'
  )
  lines.append(
    f'<line x1="{margin_left}" y1="{height - margin_bottom}" '
    f'x2="{width - margin_right}" y2="{height - margin_bottom}" '
    'stroke="#111827"/>'
  )

  colors = ("#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2")
  for index, metric_name in enumerate(selected_metrics):
    stats = aggregate[metric_name]
    mean_value = float(stats["mean"])
    low = float(stats["ci95_low"])
    high = float(stats["ci95_high"])
    x = start_x + index * (bar_width + bar_gap)
    bar_top = y_position(mean_value)
    zero_y = y_position(0.0)
    error_x = x + bar_width / 2
    error_low_y = y_position(max(low, 0.0))
    error_high_y = y_position(high)
    color = colors[index % len(colors)]
    lines.append(
      f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" '
      f'height="{zero_y - bar_top:.1f}" fill="{color}" opacity="0.82"/>'
    )
    lines.append(
      f'<line x1="{error_x:.1f}" y1="{error_high_y:.1f}" '
      f'x2="{error_x:.1f}" y2="{error_low_y:.1f}" '
      'stroke="#111827" stroke-width="2"/>'
    )
    lines.append(
      f'<line x1="{error_x - 8:.1f}" y1="{error_high_y:.1f}" '
      f'x2="{error_x + 8:.1f}" y2="{error_high_y:.1f}" '
      'stroke="#111827" stroke-width="2"/>'
    )
    lines.append(
      f'<line x1="{error_x - 8:.1f}" y1="{error_low_y:.1f}" '
      f'x2="{error_x + 8:.1f}" y2="{error_low_y:.1f}" '
      'stroke="#111827" stroke-width="2"/>'
    )
    lines.append(
      f'<text x="{error_x:.1f}" y="{bar_top - 8:.1f}" '
      'text-anchor="middle" font-family="Arial" font-size="12">'
      f'{mean_value:.3f}</text>'
    )
    label = _escape_xml(metric_name.replace("_", " "))
    lines.append(
      f'<text x="{error_x:.1f}" y="{height - margin_bottom + 24}" '
      'text-anchor="end" font-family="Arial" font-size="12" '
      f'transform="rotate(-35 {error_x:.1f} {height - margin_bottom + 24})">'
      f'{label}</text>'
    )

  lines.append("</svg>")
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_failed_result(result: EvalCaseResult) -> list[str]:
  """Formats one failed case for a Markdown report."""
  lines = [
    f"### {result.case_id}",
    "",
    f"- **Question**: {result.question}",
    f"- **Expected behavior**: `{result.expected_behavior}`",
    f"- **Failure reasons**: {', '.join(result.failure_reasons)}",
    f"- **Answer**: {result.answer}",
    "- **Retrieved snippets**:",
  ]
  if not result.contexts:
    lines.append("  - No contexts captured.")
    return lines + [""]

  for context in result.contexts[:3]:
    snippet = context.text.replace("\n", " ")[:300]
    lines.append(
      f"  - `{context.document_name}` distance={context.distance}: {snippet}"
    )
  lines.append("")
  return lines


def _aggregate_metric(
    values: list[Any],
) -> dict[str, float | int | None]:
  """Aggregates a single metric across repeated runs."""
  numeric_values = [
    numeric_value
    for value in values
    if (numeric_value := _numeric(value)) is not None
  ]
  if not numeric_values:
    return {
      "n": 0,
      "mean": None,
      "stddev": None,
      "ci95_low": None,
      "ci95_high": None,
      "min": None,
      "max": None,
    }
  avg = mean(numeric_values)
  deviation = stdev(numeric_values) if len(numeric_values) > 1 else 0.0
  margin = (
    1.96 * deviation / math.sqrt(len(numeric_values))
    if len(numeric_values) > 1
    else 0.0
  )
  return {
    "n": len(numeric_values),
    "mean": avg,
    "stddev": deviation,
    "ci95_low": avg - margin,
    "ci95_high": avg + margin,
    "min": min(numeric_values),
    "max": max(numeric_values),
  }


def _numeric(value: Any) -> float | None:
  """Returns a finite numeric value or None."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    return None
  if not math.isfinite(float(value)):
    return None
  return float(value)


def _plot_y_max(
    summaries: list[dict[str, Any]],
    metric_names: list[str],
) -> float:
  """Returns a stable y-axis max for the SVG plot."""
  values = [
    numeric_value
    for summary in summaries
    for metric_name in metric_names
    if (numeric_value := _numeric(summary.get(metric_name))) is not None
  ]
  if not values:
    return 1.0
  max_value = max(values)
  if max_value <= 1.0:
    return 1.0
  return max_value * 1.1


def _aggregate_plot_y_max(
    aggregate: dict[str, dict[str, float | int | None]],
    metric_names: list[str],
) -> float:
  """Returns a y-axis max that includes CI upper bounds."""
  values = []
  for metric_name in metric_names:
    stats = aggregate[metric_name]
    for key in ("mean", "ci95_high", "max"):
      if (numeric_value := _numeric(stats.get(key))) is not None:
        values.append(numeric_value)
  if not values:
    return 1.0
  max_value = max(values)
  if max_value <= 1.0:
    return 1.0
  return max_value * 1.1


def _escape_xml(value: str) -> str:
  """Escapes text for SVG/XML output."""
  return (
    value.replace("&", "&amp;")
    .replace("<", "&lt;")
    .replace(">", "&gt;")
    .replace('"', "&quot;")
  )


def _result_to_dict(result: EvalCaseResult) -> dict[str, Any]:
  """Converts a case result into JSON-serializable data."""
  return {
    "case_id": result.case_id,
    "question": result.question,
    "expected_behavior": result.expected_behavior,
    "answer": result.answer,
    "contexts": [
      {
        "document_name": context.document_name,
        "text": context.text,
        "distance": context.distance,
      }
      for context in result.contexts
    ],
    "latency_seconds": result.latency_seconds,
    "hit_rate_at_k": result.hit_rate_at_k,
    "recall_at_k": result.recall_at_k,
    "context_precision_at_k": result.context_precision_at_k,
    "answer_correct": result.answer_correct,
    "forbidden_terms_passed": result.forbidden_terms_passed,
    "behavior_passed": result.behavior_passed,
    "passed": result.passed,
    "failure_reasons": list(result.failure_reasons),
  }


def _rate(values: Any) -> float:
  """Returns the true-rate for a sequence of booleans."""
  materialized_values = list(values)
  if not materialized_values:
    return 0.0
  return sum(1 for value in materialized_values if value) / len(
    materialized_values
  )


def _mean_optional(values: Any) -> float | None:
  """Returns the mean after dropping None values."""
  materialized_values = [value for value in values if value is not None]
  if not materialized_values:
    return None
  return mean(materialized_values)


def _format_metric(value: Any) -> str:
  """Formats metrics consistently for Markdown tables."""
  if isinstance(value, float):
    return f"{value:.3f}"
  if value is None:
    return "n/a"
  return str(value)


def default_report_paths(reports_dir: Path) -> tuple[Path, Path]:
  """Returns timestamped default Markdown and JSON report paths."""
  timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
  return (
    reports_dir / f"rag_eval_{timestamp}.md",
    reports_dir / f"rag_eval_{timestamp}.json",
  )
