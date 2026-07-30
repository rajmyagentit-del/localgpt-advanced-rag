"""
Unit tests for eval/report.py (Improvement #13).

Deliberately does NOT test actual Ragas metric computation (that needs a
live LLM - see eval/metrics_runner.py's own module docstring for why
that part is integration-tested with a real Ollama server, not here).
This tests the pure aggregation/pass-fail logic that decides whether a
suite of already-computed scores passes or fails - the part that would
actually gate a CI pipeline.
"""

from eval.report import DEFAULT_THRESHOLDS, CaseResult, EvalReport, build_report


class TestMetricAverages:
    def test_averages_computed_correctly_across_cases(self):
        cases = [
            CaseResult("Q1", {"faithfulness": 1.0}),
            CaseResult("Q2", {"faithfulness": 0.0}),
        ]
        report = build_report(cases)
        assert report.metric_averages["faithfulness"] == 0.5

    def test_errored_cases_excluded_from_averages(self):
        cases = [
            CaseResult("Q1", {"faithfulness": 1.0}),
            CaseResult("Q2", {}, error="timed out"),
        ]
        report = build_report(cases)
        # Only Q1's score of 1.0 should count - Q2 errored and contributes nothing
        assert report.metric_averages["faithfulness"] == 1.0

    def test_empty_case_list_returns_empty_averages(self):
        report = build_report([])
        assert report.metric_averages == {}


class TestPassFail:
    def test_all_scores_above_threshold_passes(self):
        cases = [CaseResult("Q1", {"faithfulness": 0.95, "answer_relevancy": 0.9})]
        report = build_report(cases)
        assert report.passed is True

    def test_one_metric_below_threshold_fails_whole_suite(self):
        cases = [CaseResult("Q1", {"faithfulness": 0.3, "answer_relevancy": 0.9})]
        report = build_report(cases)
        assert report.passed is False
        assert "faithfulness" in report.metrics_below_threshold
        assert "answer_relevancy" not in report.metrics_below_threshold

    def test_any_errored_case_fails_the_suite_even_with_good_scores(self):
        cases = [
            CaseResult("Q1", {"faithfulness": 0.99}),
            CaseResult("Q2", {}, error="connection refused"),
        ]
        report = build_report(cases)
        assert report.passed is False

    def test_score_exactly_at_threshold_passes(self):
        """Boundary condition: >= threshold, not > threshold."""
        cases = [CaseResult("Q1", {"faithfulness": DEFAULT_THRESHOLDS["faithfulness"]})]
        report = build_report(cases)
        assert "faithfulness" not in report.metrics_below_threshold

    def test_custom_thresholds_override_defaults(self):
        cases = [CaseResult("Q1", {"faithfulness": 0.65})]
        # Default threshold (0.7) would fail this; a custom lower one passes it
        report = build_report(cases, thresholds={"faithfulness": 0.5})
        assert report.passed is True


class TestOutputFormats:
    def test_to_dict_contains_expected_keys(self):
        cases = [CaseResult("Q1", {"faithfulness": 0.9})]
        report = build_report(cases)
        d = report.to_dict()
        for key in ("passed", "num_cases", "num_failed_cases", "metric_averages", "thresholds"):
            assert key in d

    def test_to_markdown_reflects_pass_status(self):
        passing = build_report([CaseResult("Q1", {"faithfulness": 0.95})])
        failing = build_report([CaseResult("Q1", {"faithfulness": 0.1})])
        assert "PASSED" in passing.to_markdown()
        assert "FAILED" in failing.to_markdown()

    def test_to_markdown_lists_errored_case_questions(self):
        cases = [CaseResult("What is the refund window?", {}, error="LLM timeout")]
        report = build_report(cases)
        md = report.to_markdown()
        assert "What is the refund window?" in md
        assert "LLM timeout" in md
