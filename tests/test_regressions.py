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
"""

import os
import tempfile

from backend.database import ChatDatabase


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
