"""
Chat routing and query-handling logic, extracted from the old
http.server-based RequestHandler into a standalone service class.

WHY THIS EXTRACTION MATTERS (Improvement #9 - FastAPI migration): the
original logic lived as methods on a class that also handled raw HTTP
parsing (reading Content-Length, decoding JSON by hand). That coupling
meant this business logic could only ever be exercised through a real
HTTP request. Pulling it out into a plain class - constructed with its
dependencies (an OllamaClient), with plain Python method signatures -
means it can be tested directly, reused from any transport (FastAPI,
a CLI script, a background job), and now that it doesn't need `self.rfile`
or `self.headers`, it's testable without spinning up a server at all.

The actual routing heuristics (should_use_rag, the overview-based LLM
router, the pattern-matching fallback) are preserved EXACTLY as they
were in the original - this is a structural extraction, not a rewrite of
logic we can't fully re-verify without a live LLM.
"""

import json
import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, ollama_client, rag_api_url: str = "http://localhost:8001"):
        self.ollama_client = ollama_client
        self.rag_api_url = rag_api_url

    # --- Routing decision: RAG pipeline vs direct LLM ---

    def should_use_rag(self, message: str, idx_ids: list[str]) -> bool:
        """Determine if a query should use the RAG pipeline using document overviews."""
        if not idx_ids:
            return False

        try:
            doc_overviews = self.load_document_overviews(idx_ids)
            if doc_overviews:
                return self.route_using_overviews(message, doc_overviews)
        except Exception as e:
            logger.warning(f"⚠️ Overview-based routing failed, falling back to simple routing: {e}")

        return self.simple_pattern_routing(message, idx_ids)

    def load_document_overviews(self, idx_ids: list[str]) -> list[str]:
        """Load and aggregate overviews for the given index IDs."""
        aggregated: list[str] = []

        for idx in idx_ids:
            candidate_paths = [
                f"../index_store/overviews/{idx}.jsonl",
                f"index_store/overviews/{idx}.jsonl",
                f"./index_store/overviews/{idx}.jsonl",
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    logger.debug(f"📖 Loading overviews from: {p}")
                    try:
                        with open(p, encoding="utf-8") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    record = json.loads(line)
                                    overview = record.get("overview", "").strip()
                                    if overview:
                                        aggregated.append(overview)
                                except json.JSONDecodeError:
                                    continue
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Error reading {p}: {e}")
                        break

        if not aggregated:
            legacy_paths = [
                "../index_store/overviews/overviews.jsonl",
                "index_store/overviews/overviews.jsonl",
                "./index_store/overviews/overviews.jsonl",
            ]
            for p in legacy_paths:
                if os.path.exists(p):
                    logger.warning(f"⚠️ Falling back to legacy overviews file: {p}")
                    try:
                        with open(p, encoding="utf-8") as f:
                            for line in f:
                                if not line.strip():
                                    continue
                                try:
                                    record = json.loads(line)
                                    overview = record.get("overview", "").strip()
                                    if overview:
                                        aggregated.append(overview)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        logger.warning(f"⚠️ Error reading legacy overviews file {p}: {e}")
                    break

        if aggregated:
            logger.info(f"✅ Loaded {len(aggregated)} document overviews from {len(idx_ids)} index(es)")
        else:
            logger.warning(f"⚠️ No overviews found for indices {idx_ids}")
        return aggregated[:40]

    def route_using_overviews(self, query: str, overviews: list[str]) -> bool:
        """Use document overviews and an LLM to make an intelligent routing decision."""
        if not overviews:
            return False

        overviews_block = "\n".join(f"[{i+1}] {ov}" for i, ov in enumerate(overviews))

        router_prompt = f"""You are an AI router deciding whether a user question should be answered via:
• "USE_RAG" – search the user's private documents (described below)  
• "DIRECT_LLM" – reply from general knowledge (greetings, public facts, unrelated topics)

CRITICAL PRINCIPLE: When documents exist in the KB, strongly prefer USE_RAG unless the query is purely conversational or completely unrelated to any possible document content.

RULES:
1. If ANY overview clearly relates to the question (entities, numbers, addresses, dates, amounts, companies, technical terms) → USE_RAG
2. For document operations (summarize, analyze, explain, extract, find) → USE_RAG  
3. For greetings only ("Hi", "Hello", "Thanks") → DIRECT_LLM
4. For pure math/world knowledge clearly unrelated to documents → DIRECT_LLM
5. When in doubt → USE_RAG

DOCUMENT OVERVIEWS:
{overviews_block}

DECISION EXAMPLES:
• "What invoice amounts are mentioned?" → USE_RAG (document-specific)
• "Who is PromptX AI LLC?" → USE_RAG (entity in documents)  
• "What is the DeepSeek model?" → USE_RAG (mentioned in documents)
• "Summarize the research paper" → USE_RAG (document operation)
• "What is 2+2?" → DIRECT_LLM (pure math)
• "Hi there" → DIRECT_LLM (greeting only)

USER QUERY: "{query}"

Respond with exactly one word: USE_RAG or DIRECT_LLM"""

        try:
            response = self.ollama_client.chat(
                message=router_prompt,
                model="qwen3:0.6b",
                enable_thinking=False
            )
            decision = response.strip().upper()

            if "USE_RAG" in decision:
                logger.debug(f"🎯 Overview-based routing: USE_RAG for query: '{query[:50]}...'")
                return True
            elif "DIRECT_LLM" in decision:
                logger.debug(f"⚡ Overview-based routing: DIRECT_LLM for query: '{query[:50]}...'")
                return False
            else:
                logger.warning(f"⚠️ Unclear routing decision '{decision}', defaulting to RAG")
                return True

        except Exception as e:
            logger.warning(f"❌ LLM routing failed: {e}, falling back to pattern matching")
            return self.simple_pattern_routing(query, [])

    def simple_pattern_routing(self, message: str, idx_ids: list[str]) -> bool:
        """Fallback: simple pattern-based routing (no LLM call needed)."""
        message_lower = message.lower()

        greeting_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how do you do', 'nice to meet', 'pleasure to meet',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'talk to you later',
            'test', 'testing', 'check', 'ping', 'just saying', 'nevermind',
            'ok', 'okay', 'alright', 'got it', 'understood', 'i see'
        ]
        for pattern in greeting_patterns:
            if pattern in message_lower:
                return False

        rag_indicators = [
            'document', 'doc', 'file', 'pdf', 'text', 'content', 'page',
            'according to', 'based on', 'mentioned', 'states', 'says',
            'what does', 'summarize', 'summary', 'analyze', 'analysis',
            'quote', 'citation', 'reference', 'source', 'evidence',
            'explain from', 'extract', 'find in', 'search for'
        ]
        for indicator in rag_indicators:
            if indicator in message_lower:
                return True

        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        starts_with_question = any(message_lower.startswith(word) for word in question_words)
        if starts_with_question and len(message) > 40:
            return True

        if len(message.strip()) < 20:
            return False

        return False

    # --- Query execution ---

    def handle_direct_llm_query(self, session_id: str, message: str, session: dict, db) -> tuple[str, list]:
        """Handle query using direct Ollama client with thinking disabled for speed."""
        try:
            conversation_history = db.get_conversation_history(session_id)
            model = session.get('model', 'qwen3:8b')

            response_text = self.ollama_client.chat(
                message=message,
                model=model,
                conversation_history=conversation_history,
                enable_thinking=False
            )
            return response_text, []

        except Exception as e:
            logger.error(f"❌ Direct LLM error: {e}")
            return f"Error processing query: {str(e)}", []

    def handle_rag_query(
        self, session_id: str, message: str, data: dict, idx_ids: list[str]
    ) -> tuple[str, list[dict]]:
        """Handle query using the full RAG pipeline (delegates to the advanced RAG API)."""
        response_text = ""
        source_docs: list[dict] = []

        rag_api_url = f"{self.rag_api_url}/chat"
        table_name = f"text_pages_{idx_ids[-1]}" if idx_ids else None
        payload: dict[str, Any] = {
            "query": message,
            "session_id": session_id,
        }
        if table_name:
            payload["table_name"] = table_name

        optional_params: dict[str, tuple[type, str]] = {
            "compose_sub_answers": (bool, "compose_sub_answers"),
            "query_decompose": (bool, "query_decompose"),
            "ai_rerank": (bool, "ai_rerank"),
            "context_expand": (bool, "context_expand"),
            "verify": (bool, "verify"),
            "retrieval_k": (int, "retrieval_k"),
            "context_window_size": (int, "context_window_size"),
            "reranker_top_k": (int, "reranker_top_k"),
            "search_type": (str, "search_type"),
            "dense_weight": (float, "dense_weight"),
            "provence_prune": (bool, "provence_prune"),
            "provence_threshold": (float, "provence_threshold"),
        }
        for key, (caster, payload_key) in optional_params.items():
            val = data.get(key)
            if val is not None:
                try:
                    payload[payload_key] = caster(val)
                except Exception:
                    payload[payload_key] = val

        try:
            rag_response = requests.post(rag_api_url, json=payload)
            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                response_text = rag_data.get("answer", "No answer found.")
                source_docs = rag_data.get("source_documents", [])
            else:
                response_text = f"Error from RAG API ({rag_response.status_code}): {rag_response.text}"
                logger.error(f"❌ RAG API error: {response_text}")
        except requests.exceptions.ConnectionError:
            response_text = "Could not connect to the RAG API server. Please ensure it is running."
            logger.error("❌ Connection to RAG API failed (port 8001).")
        except Exception as e:
            response_text = f"Error processing RAG query: {str(e)}"
            logger.error(f"❌ RAG processing error: {e}")

        response_text = re.sub(
            r'<(think|thinking)>.*?</\1>', '', response_text, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        return response_text, source_docs
