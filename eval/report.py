"""
Aggregation and pass/fail threshold logic for RAG evaluation results.

Deliberately kept separate from metrics_runner.py (which needs a live LLM
to actually compute scores): everything in this file is pure data
transformation - given a list of already-computed per-test-case scores,
produce a summary and decide pass/fail. This means it's fully unit
testable without any LLM/network dependency, unlike the scoring itself.
"""

from dataclasses import dataclass, field
from statistics import mean


# Default thresholds - a metric average below this fails the suite.
# These are intentionally conservative starting points; tighten them as
# you build confidence in your actual RAG pipeline's real performance.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.7,
    "context_precision": 0.6,
    "context_recall": 0.6,
}


@dataclass
class CaseResult:
    """Scores for a single evaluated question."""

    question: str
    scores: dict[str, float]  # metric name -> score (0.0 to 1.0)
    error: str | None = None  # set if this case failed to evaluate at all


@dataclass
class EvalReport:
    case_results: list[CaseResult]
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))

    @property
    def metric_averages(self) -> dict[str, float]:
        """Mean score per metric, across all cases that didn't error."""
        successful = [c for c in self.case_results if c.error is None]
        if not successful:
            return {}
        all_metrics = {m for c in successful for m in c.scores}
        return {
            metric: mean(c.scores[metric] for c in successful if metric in c.scores)
            for metric in all_metrics
        }

    @property
    def failed_cases(self) -> list[CaseResult]:
        """Cases that errored out during evaluation (LLM call failed, etc)."""
        return [c for c in self.case_results if c.error is not None]

    @property
    def metrics_below_threshold(self) -> dict[str, float]:
        """Which metric averages fall below their configured threshold."""
        averages = self.metric_averages
        return {
            metric: avg
            for metric, avg in averages.items()
            if metric in self.thresholds and avg < self.thresholds[metric]
        }

    @property
    def passed(self) -> bool:
        """
        Overall pass/fail for the suite: no cases errored, and no metric
        average fell below its threshold. This is what a CI pipeline
        would check to decide whether to block a merge (see roadmap
        item on CI/CD integration).
        """
        return not self.failed_cases and not self.metrics_below_threshold

    def to_dict(self) -> dict:
        """Machine-readable summary - what a CI pipeline would parse."""
        return {
            "passed": self.passed,
            "num_cases": len(self.case_results),
            "num_failed_cases": len(self.failed_cases),
            "metric_averages": self.metric_averages,
            "metrics_below_threshold": self.metrics_below_threshold,
            "thresholds": self.thresholds,
        }

    def to_markdown(self) -> str:
        """Human-readable summary, suitable for a PR comment or console output."""
        lines = [
            f"# RAG Evaluation Report",
            "",
            f"**Overall: {'✅ PASSED' if self.passed else '❌ FAILED'}**",
            f"- Cases evaluated: {len(self.case_results)}",
            f"- Cases errored: {len(self.failed_cases)}",
            "",
            "## Metric Averages",
            "",
            "| Metric | Score | Threshold | Status |",
            "|---|---|---|---|",
        ]
        averages = self.metric_averages
        for metric in sorted(averages):
            score = averages[metric]
            threshold = self.thresholds.get(metric)
            if threshold is None:
                status = "—"
            elif score >= threshold:
                status = "✅"
            else:
                status = "❌"
            threshold_str = f"{threshold:.2f}" if threshold is not None else "—"
            lines.append(f"| {metric} | {score:.3f} | {threshold_str} | {status} |")

        if self.failed_cases:
            lines += ["", "## Errored Cases", ""]
            for c in self.failed_cases:
                lines.append(f"- **{c.question}**: {c.error}")

        return "\n".join(lines)


def build_report(
    case_results: list[CaseResult], thresholds: dict[str, float] | None = None
) -> EvalReport:
    """Convenience constructor - see EvalReport for details."""
    return EvalReport(
        case_results=case_results,
        thresholds=dict(thresholds) if thresholds is not None else dict(DEFAULT_THRESHOLDS),
    )
