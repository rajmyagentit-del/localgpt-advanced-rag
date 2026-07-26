"""
CLI entrypoint for the RAG evaluation suite (Improvement #13).

Usage:
    pip install -r eval/requirements.txt
    ollama pull qwen3:8b
    ollama pull nomic-embed-text
    python -m eval.run_eval [--json-out results.json] [--md-out results.md]

For each question in the golden dataset (eval/golden_dataset.py), this:
  1. Runs the REAL agent (rag_system.agent.loop.Agent) to get an answer
     and the retrieved source documents - not a mock, the actual pipeline
  2. Scores that answer against the reference using Ragas metrics,
     judged by a local Ollama model (see eval/metrics_runner.py)
  3. Aggregates results and checks pass/fail thresholds (eval/report.py)
  4. Prints a markdown summary and optionally writes JSON/markdown files

This is designed to be run in CI (see roadmap item on GitHub Actions) to
catch retrieval/answer-quality regressions automatically, the same way
the pytest suite catches code regressions.
"""

import argparse
import asyncio
import sys

from eval.golden_dataset import GOLDEN_DATASET
from eval.metrics_runner import RunConfig, score_case
from eval.report import CaseResult, build_report


async def _get_agent_response(agent, question: str) -> tuple[str, list[str]]:
    """
    Runs the real agent for a single question and extracts the answer
    text plus retrieved context strings, in the shape metrics_runner
    expects.
    """
    result = await agent._run_async(question)
    answer = result.get("answer", "")
    retrieved_contexts = [doc.get("text", "") for doc in result.get("source_documents", [])]
    return answer, retrieved_contexts


async def run_evaluation(agent, config: RunConfig | None = None) -> list[CaseResult]:
    config = config or RunConfig()
    results = []
    for case in GOLDEN_DATASET:
        try:
            answer, retrieved_contexts = await _get_agent_response(agent, case.question)
        except Exception as e:
            results.append(CaseResult(question=case.question, scores={}, error=f"Agent run failed: {e}"))
            continue

        result = await score_case(
            question=case.question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=case.reference,
            config=config,
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the RAG evaluation suite")
    parser.add_argument("--json-out", help="Write machine-readable results to this JSON file")
    parser.add_argument("--md-out", help="Write human-readable results to this markdown file")
    parser.add_argument("--judge-model", default="qwen3:8b", help="Ollama model to use as judge")
    args = parser.parse_args()

    # Imported here, not at module level, so `python -m eval.report`-style
    # unit tests (which don't need a live agent) never pay the cost of
    # importing the full ML stack.
    from rag_system.main import get_agent

    agent = get_agent()
    config = RunConfig(judge_model=args.judge_model)

    case_results = asyncio.run(run_evaluation(agent, config))
    report = build_report(case_results)

    print(report.to_markdown())

    if args.json_out:
        import json

        with open(args.json_out, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nWrote JSON results to {args.json_out}")

    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(report.to_markdown())
        print(f"Wrote markdown results to {args.md_out}")

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
