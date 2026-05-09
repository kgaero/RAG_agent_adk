# Repeated Local RAG Evaluation Report

## Run Metadata

- **timestamp_utc**: `20260509T175344Z`
- **agent**: `local agents.ragagent.root_agent`
- **dataset**: `/home/kgaer/code/RAG_agent_adk/evals/golden_qna.jsonl`
- **corpus_name**: `rag_eval_prd_20260509173102`
- **runs**: `3`
- **failed_runs**: `3`
- **limit**: `None`
- **distance_threshold**: `0.7`
- **case_retries**: `3`
- **retry_delay_seconds**: `90.0`
- **delay_between_cases**: `5.0`
- **delay_between_runs**: `30.0`
- **evaluator**: `deterministic local scoring; no judge LLM`
- **plot_path**: `/home/kgaer/code/RAG_agent_adk/evals/reports/repeated_rag_eval_20260509T173503Z_metrics.svg`

## Aggregate Metrics

| Metric | Mean | Std Dev | 95% CI Low | 95% CI High | Min | Max | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answer_correctness_rate | 0.844 | 0.038 | 0.801 | 0.888 | 0.800 | 0.867 | 3 |
| behavior_pass_rate | 0.978 | 0.019 | 0.956 | 1.000 | 0.967 | 1.000 | 3 |
| case_count | 30.000 | 0.000 | 30.000 | 30.000 | 30.000 | 30.000 | 3 |
| context_precision_at_k | 0.566 | 0.015 | 0.549 | 0.583 | 0.550 | 0.580 | 3 |
| forbidden_terms_pass_rate | 0.978 | 0.019 | 0.956 | 1.000 | 0.967 | 1.000 | 3 |
| hit_rate_at_k | 0.950 | 0.050 | 0.893 | 1.007 | 0.900 | 1.000 | 3 |
| latency_seconds_avg | 14.708 | 0.933 | 13.653 | 15.764 | 13.955 | 15.751 | 3 |
| latency_seconds_max | 24.904 | 7.375 | 16.558 | 33.249 | 19.666 | 33.338 | 3 |
| pass_rate | 0.800 | 0.033 | 0.762 | 0.838 | 0.767 | 0.833 | 3 |
| passed_count | 24.000 | 1.000 | 22.868 | 25.132 | 23.000 | 25.000 | 3 |
| recall_at_k | 0.950 | 0.050 | 0.893 | 1.007 | 0.900 | 1.000 | 3 |
