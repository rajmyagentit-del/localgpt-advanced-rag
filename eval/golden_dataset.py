"""
Golden dataset for automated RAG evaluation (Improvement #13).

IMPORTANT: this is a STARTER example, not a real evaluation set. A golden
dataset is inherently specific to whatever documents you've actually
indexed - there is no generic set of questions that means anything
without knowing what's in your index. Replace these with real
question/reference-answer pairs drawn from documents you've indexed
locally, ideally written by someone who knows the correct answer
independently of the RAG system (so you're not just checking that the
system agrees with itself).

Each test case:
    question:  the user's query, exactly as they'd type it
    reference: the correct/expected answer, written by a human - this is
               what "context_precision", "context_recall", and answer
               quality get measured against
    reference_contexts: (optional) the specific passages that SHOULD be
               retrieved for this question, if you want to check
               retrieval quality independent of the LLM's answer

A good golden dataset covers: simple factual lookups, questions that
require combining info from multiple chunks, questions with no answer
in the corpus (to check the system doesn't hallucinate), and edge cases
specific to your documents' structure (tables, footnotes, etc).
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    question: str
    reference: str
    reference_contexts: list[str] = field(default_factory=list)


# --- EXAMPLE cases - replace with your own document-specific Q&A pairs ---
GOLDEN_DATASET: list[EvalCase] = [
    EvalCase(
        question="What is LocalGPT?",
        reference=(
            "LocalGPT is a fully private, on-premise document intelligence "
            "platform that lets users ask questions about their documents "
            "using local AI models, without any data leaving their machine."
        ),
    ),
    EvalCase(
        question="What license is this project released under?",
        reference="The project is released under the MIT License.",
    ),
    EvalCase(
        question="Does LocalGPT send my documents to any external server?",
        reference=(
            "No. LocalGPT is designed to run entirely locally - documents "
            "and queries are processed on-device and never leave the "
            "user's machine."
        ),
    ),
]
