"""
Runs the actual Ragas metrics against real (question, answer, retrieved
context) triples, using a LOCAL Ollama model as the judge LLM.

WHY THIS NEEDS A REAL LLM CONNECTION (and isn't unit-tested like
eval/report.py): every metric here (faithfulness, answer relevancy,
context precision/recall) works by asking an LLM to judge the RAG
system's output against the question and reference answer. There's no
meaningful way to fake that judgment without a real model - mocking it
would just mean testing our own mock, not anything real. See
eval/report.py for the part of this suite that IS fully unit tested
(the pass/fail aggregation logic that consumes these scores).

WHY OLLAMA VIA THE OPENAI-COMPATIBLE ENDPOINT (not a Ragas-specific
Ollama integration, and not LangChain): Ollama exposes an
OpenAI-compatible API at /v1, so we can use the plain `openai` Python
package (lightweight, no LangChain dependency tree) pointed at a local
URL. This keeps the eval suite consistent with the project's "100%
local" positioning - no API keys, no data leaving the machine - and
avoids a real, current dependency conflict between `ragas` and the
modern LangChain ecosystem (see the "About This Fork" section of the
README for the specific pinned versions needed to make `ragas` import
cleanly: ragas + langchain-community==0.3.19).

REQUIRES (not in the main requirements.txt - see eval/requirements.txt):
    pip install ragas openai "langchain-community==0.3.19"
And a running Ollama server with a capable model pulled, e.g.:
    ollama pull qwen3:8b
"""

from dataclasses import dataclass

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from eval.report import CaseResult


@dataclass
class RunConfig:
    ollama_base_url: str = "http://localhost:11434/v1"
    judge_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"


def _build_llm(config: RunConfig):
    # AsyncOpenAI, not OpenAI - we call the async .ascore()/.agenerate()
    # methods throughout this module, which require an async-capable client.
    client = AsyncOpenAI(base_url=config.ollama_base_url, api_key="ollama")
    return llm_factory(config.judge_model, client=client)


def _build_embeddings(config: RunConfig):
    client = AsyncOpenAI(base_url=config.ollama_base_url, api_key="ollama")
    return OpenAIEmbeddings(client=client, model=config.embedding_model)


async def score_case(
    question: str,
    response: str,
    retrieved_contexts: list[str],
    reference: str,
    config: RunConfig | None = None,
) -> CaseResult:
    """
    Score a single (question, answer, retrieved context, reference)
    tuple against all four metrics. Any failure (LLM unreachable, model
    not pulled, malformed judge output) is caught and recorded on the
    CaseResult rather than raised, so one bad case doesn't kill an
    entire evaluation run.
    """
    config = config or RunConfig()
    try:
        llm = _build_llm(config)
        embeddings = _build_embeddings(config)

        faithfulness = Faithfulness(llm=llm)
        answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        context_precision = ContextPrecision(llm=llm)
        context_recall = ContextRecall(llm=llm)

        faithfulness_result = await faithfulness.ascore(
            user_input=question, response=response, retrieved_contexts=retrieved_contexts
        )
        relevancy_result = await answer_relevancy.ascore(user_input=question, response=response)
        precision_result = await context_precision.ascore(
            user_input=question, reference=reference, retrieved_contexts=retrieved_contexts
        )
        recall_result = await context_recall.ascore(
            user_input=question, retrieved_contexts=retrieved_contexts, reference=reference
        )

        return CaseResult(
            question=question,
            scores={
                "faithfulness": faithfulness_result.value,
                "answer_relevancy": relevancy_result.value,
                "context_precision": precision_result.value,
                "context_recall": recall_result.value,
            },
        )
    except Exception as e:
        return CaseResult(question=question, scores={}, error=str(e))
