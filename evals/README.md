# Local RAG Evaluation

This folder contains a repeatable, deterministic evaluation suite for the
local ADK RAG agent.

The suite does not use Ragas, DeepEval, Giskard, Agent Platform Evaluation, or
any separate LLM judge. Running the live eval still uses the local agent, so it
can incur normal Vertex/Gemini and Vertex RAG retrieval costs.

## Files

- `golden_qna.jsonl` - 30-row golden Q&A dataset grounded in the PRD source.
- `sources/prd_eval_source.md` - source snapshot for ingestion/reference.
- `reports/` - timestamped Markdown and JSON reports.
- `../scripts/run_rag_eval.py` - repeatable live eval entrypoint.
- `../scripts/run_rag_eval_repeated.py` - repeated live eval entrypoint that
  writes aggregate reports, a score SVG plot, and a cost/latency SVG plot with
  95% CI error bars.

## Before Running

Ingest `evals/sources/prd_eval_source.md` or `docs/ai_docs/PRD.md` into an
existing Vertex RAG corpus. The script does not create, ingest into, or delete
corpora.

## Run

```bash
python scripts/run_rag_eval.py --corpus-name <existing-corpus-display-name>
```

Optional arguments:

```bash
python scripts/run_rag_eval.py \
  --corpus-name <existing-corpus-display-name> \
  --dataset evals/golden_qna.jsonl \
  --limit 12 \
  --distance-threshold 0.7
```

## Run Repeated Eval With Plot

The repeated runner defaults to `--runs 3`, which is the smallest useful value
for a rough uncertainty estimate without tripling beyond that cost baseline.
Each run still calls the local agent and live Vertex RAG retrieval.

```bash
python scripts/run_rag_eval_repeated.py \
  --corpus-name <existing-corpus-display-name> \
  --distance-threshold 0.7
```

Optional arguments:

```bash
python scripts/run_rag_eval_repeated.py \
  --corpus-name <existing-corpus-display-name> \
  --runs 3 \
  --limit 12 \
  --case-retries 3 \
  --retry-delay-seconds 90 \
  --delay-between-cases 5 \
  --reports-dir evals/reports
```

If a repeated run is interrupted after writing some per-run JSON reports, rerun
with the same `--output-prefix`. Existing completed run reports are reused and
only missing runs are executed.

## Metrics

- `Hit Rate@K`: at least one expected evidence span was retrieved.
- `Recall@K`: expected evidence spans found divided by expected spans.
- `Context Precision@K`: retrieved contexts matching expected evidence or
  required answer terms divided by retrieved contexts.
- `Answer Correctness`: required terms are present and forbidden terms are
  absent in the final response.
- `No-answer / Abstention`: unanswerable prompts include abstention language.
- `Safety`: unconfirmed destructive requests are refused.
- `Security`: internal resource names are not exposed.
- `Latency`: elapsed seconds per case and aggregate latency.
- `Cost proxy counts`: deterministic proxy counts only. The reports include
  eval-case count as `llm_call_count_proxy` and total retrieved contexts as
  `retrieved_context_count`; these are not billing amounts.
- Repeated-run uncertainty: mean, standard deviation, min, max, and normal
  approximation 95% CI across N runs.
