"""
Real end-to-end test of Agent's self-correcting verification loop
(Improvement #23), invoking the ACTUAL _run_async() method - not a unit
test of an isolated helper. Only the LLM calls (triage, verification,
reformulation) and the retrieval pipeline are mocked, since those
genuinely need a live model or a real vector index; everything else
(the retry control flow, the confidence-tracking, the reformulation
wiring) is real production code from rag_system/agent/loop.py.

This is deliberately a heavier-weight test than the rest of
tests/test_agent_pure_logic.py - it's the only way to verify the retry
loop actually reformulates, actually re-retrieves, and actually keeps
the best-scoring attempt, rather than just asserting that a helper
function was called.
"""

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

# Stub the two heavy modules this test doesn't need to exercise for real
if "rag_system.pipelines.retrieval_pipeline" not in sys.modules:
    fake_rp = types.ModuleType("rag_system.pipelines.retrieval_pipeline")
    fake_rp.RetrievalPipeline = type("RetrievalPipeline", (), {})
    sys.modules["rag_system.pipelines.retrieval_pipeline"] = fake_rp
if "rag_system.retrieval.retrievers" not in sys.modules:
    fake_r = types.ModuleType("rag_system.retrieval.retrievers")
    fake_r.GraphRetriever = type("GraphRetriever", (), {})
    fake_r.MultiVectorRetriever = type("MultiVectorRetriever", (), {})
    sys.modules["rag_system.retrieval.retrievers"] = fake_r

from rag_system.agent.loop import Agent


class FakeLLMClient:
    """
    Handles both triage (called via generate_completion, synchronous
    wrapper used inside asyncio.to_thread by _triage_query_async) and
    reformulation (called via generate_completion_async) with
    deterministic, scripted responses.
    """

    def __init__(self):
        self.reformulation_calls = []

    def generate_completion(self, model, prompt, **kwargs):
        # Triage: always route to the RAG path for this test
        return {"response": '{"decision": "rag_query"}'}

    async def generate_completion_async(self, model, prompt, **kwargs):
        # Reformulation call
        self.reformulation_calls.append(prompt)
        return {"response": "What is the refund policy for annual subscriptions specifically?"}


class ScriptedVerifier:
    """Returns a scripted sequence of verification results across calls -
    first low-confidence (triggering a retry), then high-confidence
    (succeeding)."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    async def verify_async(self, query, context, answer):
        self.calls.append(query)
        return self.results.pop(0)


class FakeVerificationResult:
    def __init__(self, is_grounded, confidence_score, reasoning="test reasoning"):
        self.is_grounded = is_grounded
        self.confidence_score = confidence_score
        self.reasoning = reasoning


class ScriptedRetrievalPipeline:
    """Returns a different result depending on which query it receives -
    lets the test verify the SECOND (reformulated) query actually
    reaches retrieval, not just that reformulation was called."""

    def __init__(self, responses_by_query_substring):
        self.responses = responses_by_query_substring
        self.config = {}
        self.calls = []

    def run(self, query, table_name, window_override, event_callback=None):
        self.calls.append(query)
        for substring, response in self.responses.items():
            if substring in query:
                return response
        return {"answer": "no matching canned response", "source_documents": []}

    def _get_text_embedder(self):
        return None  # bypasses semantic cache lookup entirely for this test


def build_agent(retrieval_pipeline, verifier, llm_client, max_retries=2):
    agent = Agent.__new__(Agent)
    agent.chat_histories = {}
    agent.llm_client = llm_client
    agent.ollama_config = {"generation_model": "fake-model"}
    agent.doc_overviews = {}
    agent.semantic_cache_threshold = 0.98
    agent.retrieval_pipeline = retrieval_pipeline
    agent.verifier = verifier
    agent.pipeline_configs = {}
    agent._query_cache = {}
    return agent, max_retries


class TestSelfCorrectingVerificationLoop:
    def test_first_attempt_succeeds_no_retry_needed(self):
        """When verification passes immediately, no reformulation should
        happen at all - retries are for failures, not routine overhead."""
        pipeline = ScriptedRetrievalPipeline(
            {
                "refund policy": {
                    "answer": "Refunds within 30 days.",
                    "source_documents": [{"text": "Refund policy: 30 days."}],
                }
            }
        )
        verifier = ScriptedVerifier([FakeVerificationResult(is_grounded=True, confidence_score=95)])
        llm = FakeLLMClient()
        agent, max_retries = build_agent(pipeline, verifier, llm)

        result = asyncio.run(
            agent._run_async(
                "What is the refund policy?", query_decompose=False, max_retries=max_retries
            )
        )

        assert "Refunds within 30 days" in result["answer"]
        assert "[Confidence: 95%]" in result["answer"]
        assert (
            len(pipeline.calls) == 1
        ), "Should not have retried when the first attempt already passed"
        assert (
            len(llm.reformulation_calls) == 0
        ), "Should not reformulate when verification already passed"

    def test_failed_verification_triggers_reformulation_and_retry(self):
        """The core behavior this improvement adds: a failed
        verification should actually cause a SECOND retrieval with a
        reformulated query, not just a warning tag on the same answer."""
        pipeline = ScriptedRetrievalPipeline(
            {
                "What is the refund policy?": {
                    "answer": "We have various policies.",
                    "source_documents": [{"text": "Unrelated content about shipping."}],
                },
                "annual subscriptions specifically": {
                    "answer": "Annual subscriptions can be refunded within 30 days.",
                    "source_documents": [{"text": "Annual subscription refund policy: 30 days."}],
                },
            }
        )
        verifier = ScriptedVerifier(
            [
                FakeVerificationResult(
                    is_grounded=False,
                    confidence_score=20,
                    reasoning="context is about shipping, not refunds",
                ),
                FakeVerificationResult(is_grounded=True, confidence_score=90),
            ]
        )
        llm = FakeLLMClient()
        agent, max_retries = build_agent(pipeline, verifier, llm, max_retries=2)

        result = asyncio.run(
            agent._run_async(
                "What is the refund policy?", query_decompose=False, max_retries=max_retries
            )
        )

        # Retrieval was called TWICE - once with the original query, once
        # with the reformulated one
        assert len(pipeline.calls) == 2
        assert pipeline.calls[0] == "What is the refund policy?"
        assert "annual subscriptions specifically" in pipeline.calls[1]

        # Reformulation actually used the verifier's stated reasoning
        assert len(llm.reformulation_calls) == 1
        assert "context is about shipping, not refunds" in llm.reformulation_calls[0]

        # The final answer is the SECOND (successful) attempt, not the first
        assert "Annual subscriptions can be refunded" in result["answer"]
        assert "[Confidence: 90%]" in result["answer"]
        assert "Warning: Low confidence" not in result["answer"]

    def test_exhausts_retries_and_returns_best_attempt_with_warning(self):
        """When every attempt fails verification, the loop should give up
        after max_retries and return the BEST-scoring attempt seen (not
        just the last one), honestly flagged with a low-confidence warning."""
        pipeline = ScriptedRetrievalPipeline(
            {
                "What is the refund policy?": {
                    "answer": "Answer A - somewhat relevant.",
                    "source_documents": [{"text": "Loosely related content."}],
                },
                "annual subscriptions specifically": {
                    "answer": "Answer B - even less relevant.",
                    "source_documents": [{"text": "Barely related content."}],
                },
            }
        )
        verifier = ScriptedVerifier(
            [
                FakeVerificationResult(
                    is_grounded=False, confidence_score=35, reasoning="somewhat off-topic"
                ),
                FakeVerificationResult(
                    is_grounded=False, confidence_score=15, reasoning="very off-topic"
                ),
            ]
        )
        llm = FakeLLMClient()
        agent, max_retries = build_agent(pipeline, verifier, llm, max_retries=2)

        result = asyncio.run(
            agent._run_async(
                "What is the refund policy?", query_decompose=False, max_retries=max_retries
            )
        )

        # Exactly 2 attempts (max_retries), no more
        assert len(pipeline.calls) == 2

        # The BEST attempt (35% confidence, Answer A) should win, not the
        # last one tried (15%, Answer B) - proving best-attempt tracking
        # actually works, not just "return whatever happened last"
        assert "Answer A" in result["answer"]
        assert "[Confidence: 35%]" in result["answer"]
        assert "Warning: Low confidence" in result["answer"]

    def test_retrieval_returning_no_documents_on_retry_stops_gracefully(self):
        """If a reformulated query retrieves nothing at all, the loop
        should fall back to the best previous attempt rather than
        returning an empty/broken result."""
        pipeline = ScriptedRetrievalPipeline(
            {
                "What is the refund policy?": {
                    "answer": "Answer A.",
                    "source_documents": [{"text": "Some content."}],
                },
                # No entry matching the reformulated query -> falls
                # through to the "no matching canned response" default,
                # which has empty source_documents
            }
        )
        verifier = ScriptedVerifier(
            [
                FakeVerificationResult(
                    is_grounded=False, confidence_score=30, reasoning="not quite right"
                )
            ]
        )
        llm = FakeLLMClient()
        agent, max_retries = build_agent(pipeline, verifier, llm, max_retries=3)

        result = asyncio.run(
            agent._run_async(
                "What is the refund policy?", query_decompose=False, max_retries=max_retries
            )
        )

        assert "Answer A" in result["answer"]
        assert "[Confidence: 30%]" in result["answer"]

    def test_max_retries_of_one_means_no_retry_attempts(self):
        """max_retries=1 should mean exactly one attempt, zero retries -
        confirms the off-by-one handling (max(1, max_retries)) is correct."""
        pipeline = ScriptedRetrievalPipeline(
            {
                "What is the refund policy?": {
                    "answer": "Answer A.",
                    "source_documents": [{"text": "content"}],
                }
            }
        )
        verifier = ScriptedVerifier(
            [FakeVerificationResult(is_grounded=False, confidence_score=10, reasoning="bad")]
        )
        llm = FakeLLMClient()
        agent, _ = build_agent(pipeline, verifier, llm, max_retries=1)

        result = asyncio.run(
            agent._run_async("What is the refund policy?", query_decompose=False, max_retries=1)
        )

        assert len(pipeline.calls) == 1
        assert len(llm.reformulation_calls) == 0
        assert "Warning: Low confidence" in result["answer"]
