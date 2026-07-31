"""
Regression tests for real bugs found while setting up CI (Improvement
#11) - caught by `ruff check` on pre-existing code, not by any test
that existed before this point.

Bug 1: backend/database.py's inspect_and_populate_index_metadata() had
a local `from datetime import datetime, timedelta` deep inside the
function that shadowed the module-level `datetime` import for the
ENTIRE function scope - causing an UnboundLocalError on the earlier use
of `datetime.now()` at line ~612, every single time this function ran.
It was silently swallowed by a bare `except: pass` a few lines later
(also fixed - narrowed to `except Exception:`), so the function just
quietly returned an empty/incomplete result instead of ever crashing
visibly. Caught by ruff rule F823.

Bug 2: rag_system/agent/loop.py's Agent._route_via_overviews() computed
the real document overviews into a local variable, but the prompt sent
to the LLM hardcoded generic placeholder text instead of using them -
meaning the LLM router never actually saw what documents exist, for any
project whose content wasn't literally the original developer's demo
data. Caught by ruff rule F841 (unused variable) - the unused variable
was the tell that something downstream wasn't using real data.

Bug 3 (found later, during Improvement #19 - GraphRAG completion):
rag_system/retrieval/retrievers.py's MultiVectorRetriever.retrieve() -
the MAIN vector/hybrid search method used for every non-graph RAG
query - had the exact same local-variable-shadowing bug as Bug 1, this
time with `logger` instead of `datetime`. A redundant local
`logger = logging.getLogger(__name__)` shadowed the module-level
logger for the whole method, so the earlier `logger.info(...)` call
would raise UnboundLocalError on every single call. Caught by ruff
rule F823, same as Bug 1.
"""

import ast
import os
import tempfile

from backend.database import ChatDatabase


def _find_local_reassignments_of_module_global(
    source_path: str, class_name: str, method_name: str, name: str
):
    """
    AST-based check for a specific, real bug pattern found twice this
    session (once with `datetime` in backend/database.py, once with
    `logger` in rag_system/retrieval/retrievers.py): a local
    `x = ...` assignment anywhere inside a function body makes Python
    treat `x` as a local variable for the ENTIRE function, so any
    earlier use of the module-level `x` in that same function raises
    UnboundLocalError - even though `x` "looks like" it should just
    refer to the module-level name.

    This check doesn't need the module to actually be importable (it
    works on heavy-ML-dependency files like retrievers.py without
    needing torch/transformers installed), since it's pure source-text
    analysis via Python's own ast module - the same kind of check ruff's
    F823 rule does, made explicit and permanent as a regression test.
    """
    with open(source_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == name:
                                    return stmt.lineno
    return None


def test_multivector_retriever_retrieve_does_not_shadow_module_logger():
    """
    Regression test for a real bug found while setting up Improvement
    #19 (GraphRAG completion) and running ruff over rag_system/retrieval/:
    MultiVectorRetriever.retrieve() had a redundant local
    `logger = logging.getLogger(__name__)` statement, which shadowed the
    module-level `logger` for the whole method - meaning the EARLIER
    `logger.info(...)` call a few lines above it would raise
    UnboundLocalError on every single call. This is the main
    vector/hybrid retrieval path used for every non-graph RAG query, so
    this was a serious bug, not a cosmetic one. Fixed by removing the
    redundant local assignment (the module-level logger already covers
    it, same fix pattern as the datetime bug below).
    """
    line = _find_local_reassignments_of_module_global(
        "rag_system/retrieval/retrievers.py", "MultiVectorRetriever", "retrieve", "logger"
    )
    assert line is None, (
        f"Found a local 'logger = ...' assignment at line {line} inside "
        f"MultiVectorRetriever.retrieve() - this shadows the module-level "
        f"logger for the whole method and will cause UnboundLocalError on "
        f"any earlier logger.* call in the same method."
    )


def test_inspect_and_populate_index_metadata_does_not_crash():
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(tmp_fd)
    try:
        db = ChatDatabase(db_path=tmp_path)
        index_id = db.create_index("Regression Test Index", "desc", metadata={})

        # Before the fix, this raised UnboundLocalError internally
        # (silently caught by the function's own error handling, but a
        # real crash nonetheless - datetime.now() literally could not
        # be called). After the fix, it returns a real, useful dict.
        result = db.inspect_and_populate_index_metadata(index_id)

        assert isinstance(result, dict)
        assert result != {}, (
            "Got an empty dict back - this is the exact symptom of the "
            "original bug (the UnboundLocalError was silently swallowed "
            "and no metadata was actually inferred)"
        )
        assert "metadata_inferred_at" in result
    finally:
        os.unlink(tmp_path)


def test_agent_route_via_overviews_includes_real_document_overviews_in_prompt():
    """
    Regression test for a second real bug found via `ruff check` (F841,
    unused variable) while setting up CI: Agent._route_via_overviews()
    computed the actual document overviews into `overviews_block`, but
    the prompt sent to the LLM hardcoded a generic placeholder
    ("Invoices, DeepSeek-V3 research papers" - leftover demo/test data)
    instead of using the real overviews. This meant the LLM router was
    never actually shown what documents exist for any project whose
    content wasn't literally invoices and DeepSeek papers.
    """
    import sys
    import types

    # Stub the two heavy modules Agent's import chain needs, same
    # pattern as tests/conftest.py
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

    captured_prompts = []

    class FakeLLMClient:
        def generate_completion(self, model, prompt, **kwargs):
            captured_prompts.append(prompt)
            return {"response": '{"category": "rag_query"}'}

    agent = Agent.__new__(Agent)
    agent.llm_client = FakeLLMClient()
    agent.ollama_config = {"generation_model": "fake-model"}
    agent.doc_overviews = [
        "Q3 2024 financial report for Acme Corp",
        "Employee handbook - remote work policy",
    ]

    agent._route_via_overviews("What is the remote work policy?")
    prompt_sent = captured_prompts[0]

    assert "Acme Corp" in prompt_sent, "Real document overview missing from the router prompt"
    assert "remote work policy" in prompt_sent
    assert "DeepSeek-V3" not in prompt_sent, "Old hardcoded placeholder text is still present"
