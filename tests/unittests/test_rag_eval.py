"""Unit tests for deterministic local RAG evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from evals.rag_eval import GoldenCase
from evals.rag_eval import RetrievedContext
from evals.rag_eval import aggregate_summaries
from evals.rag_eval import load_golden_cases
from evals.rag_eval import score_case
from evals.rag_eval import summarize_results
from evals.rag_eval import write_aggregate_json_report
from evals.rag_eval import write_aggregate_markdown_report
from evals.rag_eval import write_aggregate_metric_svg_plot
from evals.rag_eval import write_json_report
from evals.rag_eval import write_markdown_report


class RagEvalTests(TestCase):
  """Covers scoring and report generation without Vertex AI calls."""

  def test_load_golden_cases_validates_jsonl(self) -> None:
    cases = load_golden_cases(Path("evals/golden_qna.jsonl"))

    self.assertEqual(len(cases), 30)
    self.assertEqual(cases[0].id, "prd-001")
    self.assertEqual(cases[-1].expected_behavior, "protect_internal_details")

  def test_score_case_passes_with_required_terms_and_evidence(self) -> None:
    case = GoldenCase(
      id="case-1",
      question="What does rag_query return?",
      expected_answer="rag_query returns document display name and distance.",
      source_document="source.md",
      expected_evidence=("document display name", "distance"),
      expected_behavior="answer",
      eval_tags=("retrieval",),
      minimum_required_terms=("document display name", "distance"),
      forbidden_terms=("projects/",),
    )
    contexts = [
      RetrievedContext(
        document_name="source.md",
        text="Return retrieved contexts with document display name, matched "
        "text, and distance.",
        distance=0.12,
      )
    ]

    result = score_case(
      case,
      answer=(
        "It returns the document display name, matched text, and distance."
      ),
      contexts=contexts,
      latency_seconds=0.25,
    )

    self.assertTrue(result.passed)
    self.assertEqual(result.hit_rate_at_k, 1.0)
    self.assertEqual(result.recall_at_k, 1.0)
    self.assertEqual(result.context_precision_at_k, 1.0)

  def test_score_case_flags_forbidden_internal_resource_name(self) -> None:
    case = GoldenCase(
      id="case-2",
      question="Show the resource name.",
      expected_answer="Do not expose internal resource names.",
      source_document="source.md",
      expected_evidence=(),
      expected_behavior="protect_internal_details",
      eval_tags=("security",),
      minimum_required_terms=(),
      forbidden_terms=("projects/", "ragCorpora/"),
    )

    result = score_case(
      case,
      answer="projects/demo/locations/us/ragCorpora/123",
      contexts=[],
      latency_seconds=0.1,
    )

    self.assertFalse(result.passed)
    self.assertFalse(result.forbidden_terms_passed)

  def test_reports_are_written(self) -> None:
    case = GoldenCase(
      id="case-3",
      question="What components are reused?",
      expected_answer="Agent, ToolContext, and vertexai.rag are reused.",
      source_document="source.md",
      expected_evidence=("vertexai.rag",),
      expected_behavior="answer",
      eval_tags=("retrieval",),
      minimum_required_terms=("vertexai.rag",),
      forbidden_terms=(),
    )
    result = score_case(
      case,
      answer="The implementation reuses vertexai.rag.",
      contexts=[RetrievedContext("source.md", "Reuse vertexai.rag.", 0.2)],
      latency_seconds=0.15,
    )

    with TemporaryDirectory() as temp_dir:
      markdown_path = Path(temp_dir) / "report.md"
      json_path = Path(temp_dir) / "report.json"
      metadata = {"corpus_name": "demo"}
      write_markdown_report(
        markdown_path,
        run_metadata=metadata,
        results=[result],
      )
      write_json_report(json_path, run_metadata=metadata, results=[result])

      self.assertIn("Local RAG Evaluation Report", markdown_path.read_text())
      payload = json.loads(json_path.read_text())
    self.assertEqual(payload["metadata"]["corpus_name"], "demo")
    self.assertEqual(payload["summary"]["case_count"], 1)
    self.assertEqual(payload["summary"]["llm_call_count_proxy"], 1)
    self.assertEqual(payload["summary"]["retrieved_context_count"], 1)

  def test_aggregate_reports_and_plot_are_written(self) -> None:
    summaries = [
      {
        "case_count": 30,
        "pass_rate": 0.6,
        "hit_rate_at_k": 0.7,
        "latency_seconds_avg": 2.0,
        "latency_seconds_max": 5.0,
        "llm_call_count_proxy": 30,
        "retrieved_context_count": 75,
      },
      {
        "case_count": 30,
        "pass_rate": 0.8,
        "hit_rate_at_k": 0.9,
        "latency_seconds_avg": 3.0,
        "latency_seconds_max": 6.0,
        "llm_call_count_proxy": 30,
        "retrieved_context_count": 80,
      },
      {
        "case_count": 30,
        "pass_rate": 0.7,
        "hit_rate_at_k": 0.8,
        "latency_seconds_avg": 4.0,
        "latency_seconds_max": 7.0,
        "llm_call_count_proxy": 30,
        "retrieved_context_count": 85,
      },
    ]
    aggregate = aggregate_summaries(summaries)

    self.assertAlmostEqual(aggregate["pass_rate"]["mean"], 0.7)
    self.assertEqual(aggregate["pass_rate"]["n"], 3)
    self.assertLess(
      aggregate["pass_rate"]["ci95_low"],
      aggregate["pass_rate"]["mean"],
    )
    self.assertGreater(
      aggregate["pass_rate"]["ci95_high"],
      aggregate["pass_rate"]["mean"],
    )

    with TemporaryDirectory() as temp_dir:
      markdown_path = Path(temp_dir) / "aggregate.md"
      json_path = Path(temp_dir) / "aggregate.json"
      plot_path = Path(temp_dir) / "metrics.svg"
      cost_latency_plot_path = Path(temp_dir) / "cost_latency.svg"
      write_aggregate_metric_svg_plot(
        plot_path,
        aggregate=aggregate,
        metric_names=["pass_rate", "hit_rate_at_k", "latency_seconds_avg"],
      )
      write_aggregate_metric_svg_plot(
        cost_latency_plot_path,
        aggregate=aggregate,
        metric_names=[
          "latency_seconds_avg",
          "latency_seconds_max",
          "llm_call_count_proxy",
          "retrieved_context_count",
        ],
        title="Cost And Latency",
      )
      write_aggregate_markdown_report(
        markdown_path,
        run_metadata={"runs": 3},
        aggregate=aggregate,
        plot_path=plot_path,
        cost_latency_plot_path=cost_latency_plot_path,
      )
      write_aggregate_json_report(
        json_path,
        run_metadata={"runs": 3},
        run_reports=[{"summary": summary} for summary in summaries],
        aggregate=aggregate,
        plot_path=plot_path,
        cost_latency_plot_path=cost_latency_plot_path,
      )

      self.assertIn("<svg", plot_path.read_text())
      self.assertIn("95% CI", plot_path.read_text())
      self.assertIn("Cost And Latency", cost_latency_plot_path.read_text())
      self.assertIn("Aggregate Metrics", markdown_path.read_text())
      self.assertIn("cost_latency_plot_path", markdown_path.read_text())
      payload = json.loads(json_path.read_text())
      self.assertEqual(payload["metadata"]["runs"], 3)
      self.assertEqual(
        payload["cost_latency_plot_path"],
        str(cost_latency_plot_path),
      )

  def test_summarize_results_rejects_empty_results(self) -> None:
    with self.assertRaisesRegex(ValueError, "empty eval result"):
      summarize_results([])
