import cgi
import http.server
import json
import logging
import os
import socketserver
import sys
import uuid
from datetime import datetime
from urllib.parse import urlparse

import requests  # 🆕 Import requests for making HTTP calls

# Add parent directory to path so we can import rag_system modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# This is the entrypoint for the backend server, so this is where logging
# gets configured (once). See rag_system/utils/logging_utils.py for details
# on how LOG_LEVEL / LOG_FORMAT env vars control behavior.
from rag_system.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Import RAG system modules for complete metadata
try:
    from rag_system.main import PIPELINE_CONFIGS
    RAG_SYSTEM_AVAILABLE = True
    logger.info("✅ RAG system modules accessible from backend")
except ImportError as e:
    PIPELINE_CONFIGS = {}
    RAG_SYSTEM_AVAILABLE = False
    logger.warning(f"⚠️ RAG system modules not available: {e}")

import re
from typing import Any

import simple_pdf_processor as pdf_module
from database import db, generate_session_title
from validation import is_valid_id, validate_chat_message, MAX_FILE_SIZE_BYTES, MAX_TOTAL_UPLOAD_BYTES
from rate_limiter import chat_rate_limiter, upload_rate_limiter
from ollama_client import OllamaClient


# 🆕 Reusable TCPServer with address reuse enabled
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

class ChatHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.ollama_client = OllamaClient()
        super().__init__(*args, **kwargs)
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _valid_id_or_400(self, candidate: str) -> str | None:
        """
        Validate a session/index ID extracted from the URL path.
        Sends a 400 response and returns None if it's not a well-formed
        UUID - callers should check for None and return immediately.
        """
        if not is_valid_id(candidate):
            self.send_json_response({"error": "Invalid or malformed ID in URL"}, status_code=400)
            return None
        return candidate

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.handle_health_check()
        elif parsed_path.path == '/sessions':
            self.handle_get_sessions()
        elif parsed_path.path == '/sessions/cleanup':
            self.handle_cleanup_sessions()
        elif parsed_path.path == '/models':
            self.handle_get_models()
        elif parsed_path.path == '/indexes':
            self.handle_get_indexes()
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = self._valid_id_or_400(parsed_path.path.split('/')[-1])
            if index_id is None:
                return
            self.handle_get_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/documents'):
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if session_id is None:
                return
            self.handle_get_session_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/indexes'):
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if session_id is None:
                return
            self.handle_get_session_indexes(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-1])
            if session_id is None:
                return
            self.handle_get_session(session_id)
        else:
            self.send_response(404)
            self.end_headers()
    
    def _rate_limited(self, limiter) -> bool:
        """
        Checks the rate limiter for the requesting client's IP. If over
        the limit, sends a 429 response (with Retry-After) and returns
        True - callers should check the return value and return
        immediately if True.
        """
        client_ip = self.client_address[0]
        allowed, retry_after = limiter.is_allowed(client_ip)
        if not allowed:
            self.send_response(429)
            self.send_header('Retry-After', str(retry_after))
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Rate limit exceeded. Please slow down.",
                "retry_after_seconds": retry_after,
            }).encode('utf-8'))
            return True
        return False

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/chat':
            if self._rate_limited(chat_rate_limiter):
                return
            self.handle_chat()
        elif parsed_path.path == '/sessions':
            self.handle_create_session()
        elif parsed_path.path == '/indexes':
            self.handle_create_index()
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/upload'):
            if self._rate_limited(upload_rate_limiter):
                return
            index_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if index_id is None:
                return
            self.handle_index_file_upload(index_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.endswith('/build'):
            index_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if index_id is None:
                return
            self.handle_build_index(index_id)
        elif parsed_path.path.startswith('/sessions/') and '/indexes/' in parsed_path.path:
            parts = parsed_path.path.split('/')
            session_id = self._valid_id_or_400(parts[2])
            if session_id is None:
                return
            index_id = self._valid_id_or_400(parts[4])
            if index_id is None:
                return
            self.handle_link_index_to_session(session_id, index_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/messages'):
            if self._rate_limited(chat_rate_limiter):
                return
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if session_id is None:
                return
            self.handle_session_chat(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/upload'):
            if self._rate_limited(upload_rate_limiter):
                return
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if session_id is None:
                return
            self.handle_file_upload(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/index'):
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if session_id is None:
                return
            self.handle_index_documents(session_id)
        elif parsed_path.path.startswith('/sessions/') and parsed_path.path.endswith('/rename'):
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-2])
            if session_id is None:
                return
            self.handle_rename_session(session_id)
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        """Handle DELETE requests"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path.startswith('/sessions/') and parsed_path.path.count('/') == 2:
            session_id = self._valid_id_or_400(parsed_path.path.split('/')[-1])
            if session_id is None:
                return
            self.handle_delete_session(session_id)
        elif parsed_path.path.startswith('/indexes/') and parsed_path.path.count('/') == 2:
            index_id = self._valid_id_or_400(parsed_path.path.split('/')[-1])
            if index_id is None:
                return
            self.handle_delete_index(index_id)
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_chat(self):
        """Handle legacy chat requests (without sessions)"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            message = data.get('message', '')
            model = data.get('model', 'llama3.2:latest')
            conversation_history = data.get('conversation_history', [])
            
            is_valid, error = validate_chat_message(message)
            if not is_valid:
                self.send_json_response({"error": error}, status_code=400)
                return
            
            # Check if Ollama is running
            if not self.ollama_client.is_ollama_running():
                self.send_json_response({
                    "error": "Ollama is not running. Please start Ollama first."
                }, status_code=503)
                return
            
            # Get response from Ollama
            response = self.ollama_client.chat(message, model, conversation_history)
            
            self.send_json_response({
                "response": response,
                "model": model,
                "message_count": len(conversation_history) + 1
            })
            
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Server error: {str(e)}"
            }, status_code=500)
    
    def handle_health_check(self):
        """
        Real health check: actually verifies each dependency instead of
        unconditionally returning 200 OK.

        Returns:
          - 200 if every dependency is healthy
          - 503 (Service Unavailable) if any REQUIRED dependency is down -
            this is the status code a load balancer / Kubernetes readiness
            probe looks at to decide whether to route traffic here.

        Each check is wrapped individually so one failing dependency
        doesn't crash the whole endpoint or hide the status of the others.
        """
        checks = {}
        overall_healthy = True

        # --- Ollama (required: no LLM generation works without it) ---
        try:
            ollama_up = self.ollama_client.is_ollama_running()
            checks["ollama"] = {
                "status": "up" if ollama_up else "down",
                "required": True,
            }
            if not ollama_up:
                overall_healthy = False
        except Exception as e:
            checks["ollama"] = {"status": "error", "required": True, "detail": str(e)}
            overall_healthy = False

        # --- SQLite (required: sessions/chat history depend on it) ---
        try:
            stats = db.get_stats()
            checks["database"] = {"status": "up", "required": True, "stats": stats}
        except Exception as e:
            checks["database"] = {"status": "error", "required": True, "detail": str(e)}
            overall_healthy = False

        # --- LanceDB (required only once at least one index exists;
        #     treated as informational rather than failing the whole
        #     health check, since a fresh install has no indexes yet) ---
        try:
            import os as _os

            from rag_system.config import settings
            lancedb_path = settings.lancedb_path
            checks["lancedb"] = {
                "status": "up" if _os.path.isdir(lancedb_path) else "not_initialized",
                "required": False,
                "path": lancedb_path,
            }
        except Exception as e:
            checks["lancedb"] = {"status": "error", "required": False, "detail": str(e)}

        response_body = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "checks": checks,
        }
        self.send_json_response(response_body, status_code=200 if overall_healthy else 503)

    def handle_get_sessions(self):
        """Get all chat sessions"""
        try:
            sessions = db.get_sessions()
            self.send_json_response({
                "sessions": sessions,
                "total": len(sessions)
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to get sessions: {str(e)}"
            }, status_code=500)
    
    def handle_cleanup_sessions(self):
        """Clean up empty sessions"""
        try:
            cleanup_count = db.cleanup_empty_sessions()
            self.send_json_response({
                "message": f"Cleaned up {cleanup_count} empty sessions",
                "cleanup_count": cleanup_count
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to cleanup sessions: {str(e)}"
            }, status_code=500)
    
    def handle_get_session(self, session_id: str):
        """Get a specific session with its messages"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({
                    "error": "Session not found"
                }, status_code=404)
                return
            
            messages = db.get_messages(session_id)
            
            self.send_json_response({
                "session": session,
                "messages": messages
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to get session: {str(e)}"
            }, status_code=500)
    
    def handle_get_session_documents(self, session_id: str):
        """Return documents and basic info for a session."""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            docs = db.get_documents_for_session(session_id)

            # Extract original filenames from stored paths
            filenames = [os.path.basename(p).split('_', 1)[-1] if '_' in os.path.basename(p) else os.path.basename(p) for p in docs]

            self.send_json_response({
                "session": session,
                "files": filenames,
                "file_count": len(docs)
            })
        except Exception as e:
            self.send_json_response({"error": f"Failed to get documents: {str(e)}"}, status_code=500)
    
    def handle_create_session(self):
        """Create a new chat session"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            title = data.get('title', 'New Chat')
            model = data.get('model', 'llama3.2:latest')
            
            session_id = db.create_session(title, model)
            session = db.get_session(session_id)
            
            self.send_json_response({
                "session": session,
                "session_id": session_id
            }, status_code=201)
            
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            self.send_json_response({
                "error": f"Failed to create session: {str(e)}"
            }, status_code=500)
    
    def handle_session_chat(self, session_id: str):
        """
        Handle chat within a specific session.
        Intelligently routes between direct LLM (fast) and RAG pipeline (document-aware).
        """
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            message = data.get('message', '')

            is_valid, error = validate_chat_message(message)
            if not is_valid:
                self.send_json_response({"error": error}, status_code=400)
                return

            if session['message_count'] == 0:
                title = generate_session_title(message)
                db.update_session_title(session_id, title)

            # Add user message to database first
            user_message_id = db.add_message(session_id, message, "user")
            
            # 🎯 SMART ROUTING: Decide between direct LLM vs RAG
            idx_ids = db.get_indexes_for_session(session_id)
            force_rag = bool(data.get("force_rag", False))
            use_rag = True if force_rag else self._should_use_rag(message, idx_ids)
            
            if use_rag:
                # 🔍 --- Use RAG Pipeline for Document-Related Queries ---
                logger.debug(f"🔍 Using RAG pipeline for document query: '{message[:50]}...'")
                response_text, source_docs = self._handle_rag_query(session_id, message, data, idx_ids)
            else:
                # ⚡ --- Use Direct LLM for General Queries (FAST) ---
                logger.debug(f"⚡ Using direct LLM for general query: '{message[:50]}...'")
                response_text, source_docs = self._handle_direct_llm_query(session_id, message, session)

            # Add AI response to database
            ai_message_id = db.add_message(session_id, response_text, "assistant")
            
            updated_session = db.get_session(session_id)
            
            # Send response with proper error handling
            self.send_json_response({
                "response": response_text,
                "session": updated_session,
                "source_documents": source_docs,
                "used_rag": use_rag
            })
            
        except BrokenPipeError:
            # Client disconnected - this is normal for long queries, just log it
            logger.warning(f"⚠️  Client disconnected during RAG processing for query: '{message[:30]}...'")
        except json.JSONDecodeError:
            self.send_json_response({
                "error": "Invalid JSON"
            }, status_code=400)
        except Exception as e:
            logger.error(f"❌ Server error in session chat: {str(e)}")
            try:
                self.send_json_response({
                    "error": f"Server error: {str(e)}"
                }, status_code=500)
            except BrokenPipeError:
                logger.warning("⚠️  Client disconnected during error response")
    
    def _should_use_rag(self, message: str, idx_ids: list[str]) -> bool:
        """
        🧠 ENHANCED: Determine if a query should use RAG pipeline using document overviews.
        
        Args:
            message: The user's query
            idx_ids: List of index IDs associated with the session
            
        Returns:
            bool: True if should use RAG, False for direct LLM
        """
        # No indexes = definitely no RAG needed
        if not idx_ids:
            return False

        # Load document overviews for intelligent routing
        try:
            doc_overviews = self._load_document_overviews(idx_ids)
            if doc_overviews:
                return self._route_using_overviews(message, doc_overviews)
        except Exception as e:
            logger.warning(f"⚠️ Overview-based routing failed, falling back to simple routing: {e}")
        
        # Fallback to simple pattern matching if overviews unavailable
        return self._simple_pattern_routing(message, idx_ids)

    def _load_document_overviews(self, idx_ids: list[str]) -> list[str]:
        """Load and aggregate overviews for the given index IDs.
        
        Strategy:
        1. Attempt to load each index's dedicated overview file.
        2. Aggregate all overviews found across available files (deduplicated).
        3. If none of the index files exist, fall back to the legacy global overview file.
        """
        import json
        import os

        aggregated: list[str] = []

        # 1️⃣  Collect overviews from per-index files
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
                                    continue  # skip malformed lines
                        break  # Stop after the first existing path for this idx
                    except Exception as e:
                        logger.warning(f"⚠️ Error reading {p}: {e}")
                        break  # Don't keep trying other paths for this idx if read failed

        # 2️⃣  Fall back to legacy global file if no per-index overviews found
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

        # Limit for performance
        if aggregated:
            logger.info(f"✅ Loaded {len(aggregated)} document overviews from {len(idx_ids)} index(es)")
        else:
            logger.warning(f"⚠️ No overviews found for indices {idx_ids}")
        return aggregated[:40]

    def _route_using_overviews(self, query: str, overviews: list[str]) -> bool:
        """
        🎯 Use document overviews and LLM to make intelligent routing decisions.
        
        Returns True if RAG should be used, False for direct LLM.
        """
        if not overviews:
            return False
        
        # Format overviews for the routing prompt
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
            # Use Ollama to make the routing decision
            response = self.ollama_client.chat(
                message=router_prompt,
                model="qwen3:0.6b",  # Fast model for routing
                enable_thinking=False  # Fast routing
            )
            
            # The response is directly the text, not a dict
            decision = response.strip().upper()
            
            # Parse decision
            if "USE_RAG" in decision:
                logger.debug(f"🎯 Overview-based routing: USE_RAG for query: '{query[:50]}...'")
                return True
            elif "DIRECT_LLM" in decision:
                logger.debug(f"⚡ Overview-based routing: DIRECT_LLM for query: '{query[:50]}...'")
                return False
            else:
                logger.warning(f"⚠️ Unclear routing decision '{decision}', defaulting to RAG")
                return True  # Default to RAG when uncertain
                
        except Exception as e:
            logger.warning(f"❌ LLM routing failed: {e}, falling back to pattern matching")
            return self._simple_pattern_routing(query, [])

    def _simple_pattern_routing(self, message: str, idx_ids: list[str]) -> bool:
        """
        📝 FALLBACK: Simple pattern-based routing (original logic).
        """
        message_lower = message.lower()
        
        # Always use Direct LLM for greetings and casual conversation
        greeting_patterns = [
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'how do you do', 'nice to meet', 'pleasure to meet',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'talk to you later',
            'test', 'testing', 'check', 'ping', 'just saying', 'nevermind',
            'ok', 'okay', 'alright', 'got it', 'understood', 'i see'
        ]
        
        # Check for greeting patterns
        for pattern in greeting_patterns:
            if pattern in message_lower:
                return False  # Use Direct LLM for greetings
        
        # Keywords that strongly suggest document-related queries
        rag_indicators = [
            'document', 'doc', 'file', 'pdf', 'text', 'content', 'page',
            'according to', 'based on', 'mentioned', 'states', 'says',
            'what does', 'summarize', 'summary', 'analyze', 'analysis',
            'quote', 'citation', 'reference', 'source', 'evidence',
            'explain from', 'extract', 'find in', 'search for'
        ]
        
        # Check for strong RAG indicators
        for indicator in rag_indicators:
            if indicator in message_lower:
                return True
        
        # Question words + substantial length might benefit from RAG
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        starts_with_question = any(message_lower.startswith(word) for word in question_words)
        
        if starts_with_question and len(message) > 40:
            return True
        
        # Very short messages - use direct LLM
        if len(message.strip()) < 20:
            return False
        
        # Default to Direct LLM unless there's clear indication of document query
        return False
    
    def _handle_direct_llm_query(self, session_id: str, message: str, session: dict):
        """
        Handle query using direct Ollama client with thinking disabled for speed.
        
        Returns:
            tuple: (response_text, empty_source_docs)
        """
        try:
            # Get conversation history for context
            conversation_history = db.get_conversation_history(session_id)
            
            # Use the session's model or default
            model = session.get('model', 'qwen3:8b')  # Default to fast model
            
            # Direct Ollama call with thinking disabled for speed
            response_text = self.ollama_client.chat(
                message=message,
                model=model,
                conversation_history=conversation_history,
                enable_thinking=False  # ⚡ DISABLE THINKING FOR SPEED
            )
            
            return response_text, []  # No source docs for direct LLM
            
        except Exception as e:
            logger.error(f"❌ Direct LLM error: {e}")
            return f"Error processing query: {str(e)}", []
    
    def _handle_rag_query(self, session_id: str, message: str, data: dict, idx_ids: list[str]):
        """
        Handle query using the full RAG pipeline (delegates to the advanced RAG API running on port 8001).

        Returns:
            tuple[str, List[dict]]: (response_text, source_documents)
        """
        # Defaults
        response_text = ""
        source_docs: list[dict] = []

        # Build payload for RAG API
        rag_api_url = "http://localhost:8001/chat"
        table_name = f"text_pages_{idx_ids[-1]}" if idx_ids else None
        payload: dict[str, Any] = {
            "query": message,
            "session_id": session_id,
        }
        if table_name:
            payload["table_name"] = table_name

        # Copy optional parameters from the incoming request
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
                    payload[payload_key] = caster(val)  # type: ignore[arg-type]
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

        # Strip any <think>/<thinking> tags that might slip through
        response_text = re.sub(r'<(think|thinking)>.*?</\\1>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()

        return response_text, source_docs

    def handle_delete_session(self, session_id: str):
        """Delete a session and its messages"""
        try:
            deleted = db.delete_session(session_id)
            if deleted:
                self.send_json_response({'deleted': deleted})
            else:
                self.send_json_response({'error': 'Session not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_file_upload(self, session_id: str):
        """Handle file uploads, save them, and associate with the session."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
        except (TypeError, ValueError):
            content_length = 0
        if content_length > MAX_TOTAL_UPLOAD_BYTES:
            self.send_json_response(
                {"error": f"Upload exceeds maximum total size of {MAX_TOTAL_UPLOAD_BYTES // (1024*1024)}MB"},
                status_code=413,
            )
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']}
        )

        uploaded_files = []
        if 'files' in form:
            files = form['files']
            if not isinstance(files, list):
                files = [files]
            
            upload_dir = "shared_uploads"
            os.makedirs(upload_dir, exist_ok=True)

            for file_item in files:
                if file_item.filename:
                    # Create a unique filename to avoid overwrites
                    unique_filename = f"{uuid.uuid4()}_{file_item.filename}"
                    file_path = os.path.join(upload_dir, unique_filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(file_item.file.read())
                    
                    # Store the absolute path for the indexing service
                    absolute_file_path = os.path.abspath(file_path)
                    db.add_document_to_session(session_id, absolute_file_path)
                    uploaded_files.append({"filename": file_item.filename, "stored_path": absolute_file_path})

        if not uploaded_files:
            self.send_json_response({"error": "No files were uploaded"}, status_code=400)
            return
            
        self.send_json_response({
            "message": f"Successfully uploaded {len(uploaded_files)} files.",
            "uploaded_files": uploaded_files
        })

    def handle_index_documents(self, session_id: str):
        """Triggers indexing for all documents in a session."""
        logger.info(f"🔥 Received request to index documents for session {session_id[:8]}...")
        try:
            file_paths = db.get_documents_for_session(session_id)
            if not file_paths:
                self.send_json_response({"message": "No documents to index for this session."}, status_code=200)
                return

            logger.info(f"Found {len(file_paths)} documents to index. Sending to RAG API...")
            
            rag_api_url = "http://localhost:8001/index"
            rag_response = requests.post(rag_api_url, json={"file_paths": file_paths, "session_id": session_id})

            if rag_response.status_code == 200:
                logger.info("✅ RAG API successfully indexed documents.")
                # Merge key config values into index metadata
                idx_meta = {
                    "session_linked": True,
                    "retrieval_mode": "hybrid",
                }
                try:
                    db.update_index_metadata(session_id, idx_meta)  # session_id used as index_id in text table naming
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update index metadata for session index: {e}")
                self.send_json_response(rag_response.json())
            else:
                error_info = rag_response.text
                logger.error(f"❌ RAG API indexing failed ({rag_response.status_code}): {error_info}")
                self.send_json_response({"error": f"Indexing failed: {error_info}"}, status_code=500)

        except Exception as e:
            logger.error(f"❌ Exception during indexing: {str(e)}")
            self.send_json_response({"error": f"An unexpected error occurred: {str(e)}"}, status_code=500)
            
    def handle_pdf_upload(self, session_id: str):
        """
        Processes PDF files: extracts text and stores it in the database.
        DEPRECATED: This is the old method. Use handle_file_upload instead.
        """
        # This function is now deprecated in favor of the new indexing workflow
        # but is kept for potential legacy/compatibility reasons.
        # For new functionality, it should not be used.
        self.send_json_response({
            "warning": "This upload method is deprecated. Use the new file upload and indexing flow.",
            "message": "No action taken."
        }, status_code=410) # 410 Gone

    def handle_get_models(self):
        """Get available models from both Ollama and HuggingFace, grouped by capability"""
        try:
            generation_models = []
            embedding_models = []
            
            # Get Ollama models if available
            if self.ollama_client.is_ollama_running():
                all_ollama_models = self.ollama_client.list_models()
                
                # Very naive classification - same logic as RAG API server
                ollama_embedding_models = [m for m in all_ollama_models if any(k in m for k in ['embed','bge','embedding','text'])]
                ollama_generation_models = [m for m in all_ollama_models if m not in ollama_embedding_models]
                
                generation_models.extend(ollama_generation_models)
                embedding_models.extend(ollama_embedding_models)
            
            # Add supported HuggingFace embedding models
            huggingface_embedding_models = [
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-4B", 
                "Qwen/Qwen3-Embedding-8B"
            ]
            embedding_models.extend(huggingface_embedding_models)
            
            # Sort models for consistent ordering
            generation_models.sort()
            embedding_models.sort()
            
            self.send_json_response({
                "generation_models": generation_models,
                "embedding_models": embedding_models
            })
        except Exception as e:
            self.send_json_response({
                "error": f"Could not list models: {str(e)}"
            }, status_code=500)

    def handle_get_indexes(self):
        try:
            data = db.list_indexes()
            self.send_json_response({'indexes': data, 'total': len(data)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_get_index(self, index_id: str):
        try:
            data = db.get_index(index_id)
            if not data:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
                return
            self.send_json_response(data)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_create_index(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            name = data.get('name')
            description = data.get('description')
            metadata = data.get('metadata', {})
            
            if not name:
                self.send_json_response({'error': 'Name required'}, status_code=400)
                return
            
            # Add complete metadata from RAG system configuration if available
            if RAG_SYSTEM_AVAILABLE and PIPELINE_CONFIGS.get('default'):
                default_config = PIPELINE_CONFIGS['default']
                complete_metadata = {
                    'status': 'created',
                    'metadata_source': 'rag_system_config',
                    'created_at': json.loads(json.dumps(datetime.now().isoformat())),
                    'chunk_size': 512,  # From default config
                    'chunk_overlap': 64,  # From default config
                    'retrieval_mode': 'hybrid',  # From default config
                    'window_size': 5,  # From default config
                    'embedding_model': 'Qwen/Qwen3-Embedding-0.6B',  # From default config
                    'enrich_model': 'qwen3:0.6b',  # From default config
                    'overview_model': 'qwen3:0.6b',  # From default config
                    'enable_enrich': True,  # From default config
                    'latechunk': True,  # From default config
                    'docling_chunk': True,  # From default config
                    'note': 'Default configuration from RAG system'
                }
                # Merge with any provided metadata
                complete_metadata.update(metadata)
                metadata = complete_metadata
            
            idx_id = db.create_index(name, description, metadata)
            self.send_json_response({'index_id': idx_id}, status_code=201)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)
    
    def handle_index_file_upload(self, index_id: str):
        """Reuse file upload logic but store docs under index."""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
        except (TypeError, ValueError):
            content_length = 0
        if content_length > MAX_TOTAL_UPLOAD_BYTES:
            self.send_json_response(
                {"error": f"Upload exceeds maximum total size of {MAX_TOTAL_UPLOAD_BYTES // (1024*1024)}MB"},
                status_code=413,
            )
            return

        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD':'POST', 'CONTENT_TYPE': self.headers['Content-Type']})
        uploaded_files=[]
        if 'files' in form:
            files=form['files']
            if not isinstance(files, list):
                files=[files]
            upload_dir='shared_uploads'
            os.makedirs(upload_dir, exist_ok=True)
            for f in files:
                if f.filename:
                    unique=f"{uuid.uuid4()}_{f.filename}"
                    path=os.path.join(upload_dir, unique)
                    with open(path,'wb') as out: out.write(f.file.read())
                    db.add_document_to_index(index_id, f.filename, os.path.abspath(path))
                    uploaded_files.append({'filename':f.filename,'stored_path':os.path.abspath(path)})
        if not uploaded_files:
            self.send_json_response({'error':'No files uploaded'}, status_code=400); return
        self.send_json_response({'message':f"Uploaded {len(uploaded_files)} files","uploaded_files":uploaded_files})
    
    def handle_build_index(self, index_id: str):
        try:
            index=db.get_index(index_id)
            if not index:
                self.send_json_response({'error':'Index not found'}, status_code=404); return
            file_paths=[d['stored_path'] for d in index.get('documents',[])]
            if not file_paths:
                self.send_json_response({'error':'No documents to index'}, status_code=400); return

            # Parse request body for optional flags and configuration
            latechunk = False
            docling_chunk = False
            chunk_size = 512
            chunk_overlap = 64
            retrieval_mode = 'hybrid'
            window_size = 2
            enable_enrich = True
            embedding_model = None
            enrich_model = None
            batch_size_embed = 50
            batch_size_enrich = 25
            overview_model = None
            
            if 'Content-Length' in self.headers and int(self.headers['Content-Length']) > 0:
                try:
                    length = int(self.headers['Content-Length'])
                    body = self.rfile.read(length)
                    opts = json.loads(body.decode('utf-8'))
                    latechunk = bool(opts.get('latechunk', False))
                    docling_chunk = bool(opts.get('doclingChunk', False))
                    chunk_size = int(opts.get('chunkSize', 512))
                    chunk_overlap = int(opts.get('chunkOverlap', 64))
                    retrieval_mode = str(opts.get('retrievalMode', 'hybrid'))
                    window_size = int(opts.get('windowSize', 2))
                    enable_enrich = bool(opts.get('enableEnrich', True))
                    embedding_model = opts.get('embeddingModel')
                    enrich_model = opts.get('enrichModel')
                    batch_size_embed = int(opts.get('batchSizeEmbed', 50))
                    batch_size_enrich = int(opts.get('batchSizeEnrich', 25))
                    overview_model = opts.get('overviewModel')
                except Exception:
                    # Keep defaults on parse error
                    pass

            # Set per-index overview file path
            overview_path = f"index_store/overviews/{index_id}.jsonl"

            # Ensure config_override includes overview_path
            def ensure_overview_path(cfg: dict):
                cfg["overview_path"] = overview_path
            
            # we'll inject later when we build config_override

            # Delegate to advanced RAG API same as session indexing
            rag_api_url = "http://localhost:8001/index"

            import requests
            # Use the index's dedicated LanceDB table so retrieval matches
            table_name = index.get("vector_table_name")
            payload = {
                "file_paths": file_paths,
                "session_id": index_id,  # reuse index_id for progress tracking
                "table_name": table_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "retrieval_mode": retrieval_mode,
                "window_size": window_size,
                "enable_enrich": enable_enrich,
                "batch_size_embed": batch_size_embed,
                "batch_size_enrich": batch_size_enrich
            }
            if latechunk:
                payload["enable_latechunk"] = True
            if docling_chunk:
                payload["enable_docling_chunk"] = True
            if embedding_model:
                payload["embedding_model"] = embedding_model
            if enrich_model:
                payload["enrich_model"] = enrich_model
            if overview_model:
                payload["overview_model_name"] = overview_model
                
            rag_resp = requests.post(rag_api_url, json=payload)
            if rag_resp.status_code==200:
                meta_updates = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "retrieval_mode": retrieval_mode,
                    "window_size": window_size,
                    "enable_enrich": enable_enrich,
                    "latechunk": latechunk,
                    "docling_chunk": docling_chunk,
                }
                if embedding_model:
                    meta_updates["embedding_model"] = embedding_model
                if enrich_model:
                    meta_updates["enrich_model"] = enrich_model
                if overview_model:
                    meta_updates["overview_model"] = overview_model
                try:
                    db.update_index_metadata(index_id, meta_updates)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update index metadata: {e}")

                self.send_json_response({
                    "response": rag_resp.json(),
                    **meta_updates
                })
            else:
                # Gracefully handle scenario where table already exists (idempotent build)
                try:
                    err_json = rag_resp.json()
                except Exception:
                    err_json = {}
                err_text = err_json.get('error') if isinstance(err_json, dict) else rag_resp.text
                if err_text and 'already exists' in err_text:
                    # Treat as non-fatal; return message indicating index previously built
                    self.send_json_response({
                        "message": "Index already built – skipping rebuild.",
                        "note": err_text
                })
                else:
                    self.send_json_response({"error":f"RAG indexing failed: {rag_resp.text}"}, status_code=500)
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)
    
    def handle_link_index_to_session(self, session_id: str, index_id: str):
        try:
            db.link_index_to_session(session_id, index_id)
            self.send_json_response({'message':'Index linked to session'})
        except Exception as e:
            self.send_json_response({'error':str(e)}, status_code=500)

    def handle_get_session_indexes(self, session_id: str):
        try:
            idx_ids = db.get_indexes_for_session(session_id)
            indexes = []
            for idx_id in idx_ids:
                idx = db.get_index(idx_id)
                if idx:
                    # Try to populate metadata for older indexes that have empty metadata
                    if not idx.get('metadata') or len(idx['metadata']) == 0:
                        logger.debug(f"🔍 Attempting to infer metadata for index {idx_id[:8]}...")
                        inferred_metadata = db.inspect_and_populate_index_metadata(idx_id)
                        if inferred_metadata:
                            # Refresh the index data with the new metadata
                            idx = db.get_index(idx_id)
                    indexes.append(idx)
            self.send_json_response({'indexes': indexes, 'total': len(indexes)})
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_delete_index(self, index_id: str):
        """Remove an index, its documents, links, and the underlying LanceDB table."""
        try:
            deleted = db.delete_index(index_id)
            if deleted:
                self.send_json_response({'message': 'Index deleted successfully', 'index_id': index_id})
            else:
                self.send_json_response({'error': 'Index not found'}, status_code=404)
        except Exception as e:
            self.send_json_response({'error': str(e)}, status_code=500)

    def handle_rename_session(self, session_id: str):
        """Rename an existing session title"""
        try:
            session = db.get_session(session_id)
            if not session:
                self.send_json_response({"error": "Session not found"}, status_code=404)
                return

            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response({"error": "Request body required"}, status_code=400)
                return

            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            new_title: str = data.get('title', '').strip()

            if not new_title:
                self.send_json_response({"error": "Title cannot be empty"}, status_code=400)
                return

            db.update_session_title(session_id, new_title)
            updated_session = db.get_session(session_id)

            self.send_json_response({
                "message": "Session renamed successfully",
                "session": updated_session
            })

        except json.JSONDecodeError:
            self.send_json_response({"error": "Invalid JSON"}, status_code=400)
        except Exception as e:
            self.send_json_response({"error": f"Failed to rename session: {str(e)}"}, status_code=500)

    def send_json_response(self, data, status_code: int = 200):
        """Send a JSON (UTF-8) response with CORS headers. Safe against client disconnects."""
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.send_header('Access-Control-Allow-Credentials', 'true')
            self.end_headers()
        
            response_bytes = json.dumps(data, indent=2).encode('utf-8')
            self.wfile.write(response_bytes)
        except BrokenPipeError:
            # Client disconnected before we could finish sending
            logger.warning("⚠️  Client disconnected during response – ignoring.")
        except Exception as e:
            logger.error(f"❌ Error sending response: {e}")
    
    def log_message(self, format, *args):
        """Custom log format"""
        logger.info(f"[{self.date_time_string()}] {format % args}")

def main():
    """Main function to initialize and start the server"""
    PORT = 8000  # 🆕 Define port
    try:
        # Initialize the database
        logger.info("✅ Database initialized successfully")

        # Initialize the PDF processor
        try:
            pdf_module.initialize_simple_pdf_processor()
            logger.info("📄 Initializing simple PDF processing...")
            if pdf_module.simple_pdf_processor:
                logger.info("✅ Simple PDF processor initialized")
            else:
                logger.warning("⚠️ PDF processing could not be initialized.")
        except Exception as e:
            logger.error(f"❌ Error initializing PDF processor: {e}")
            logger.warning("⚠️ PDF processing disabled - server will run without RAG functionality")

        # Set a global reference to the initialized processor if needed elsewhere
        global pdf_processor
        pdf_processor = pdf_module.simple_pdf_processor
        if pdf_processor:
            logger.info("✅ Global PDF processor initialized")
        else:
            logger.warning("⚠️ PDF processing disabled - server will run without RAG functionality")
        
        # Cleanup empty sessions on startup
        logger.info("🧹 Cleaning up empty sessions...")
        cleanup_count = db.cleanup_empty_sessions()
        if cleanup_count > 0:
            logger.info(f"✨ Cleaned up {cleanup_count} empty sessions")
        else:
            logger.debug("✨ No empty sessions to clean up")

        # Start the server
        with ReusableTCPServer(("", PORT), ChatHandler) as httpd:
            logger.info(f"🚀 Starting localGPT backend server on port {PORT}")
            logger.info(f"📍 Chat endpoint: http://localhost:{PORT}/chat")
            logger.info(f"🔍 Health check: http://localhost:{PORT}/health")
            
            # Test Ollama connection
            client = OllamaClient()
            if client.is_ollama_running():
                models = client.list_models()
                logger.info(f"✅ Ollama is running with {len(models)} models")
                logger.info(f"📋 Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}")
            else:
                logger.warning("⚠️  Ollama is not running. Please start Ollama:")
                logger.warning("   Install: https://ollama.ai")
                logger.warning("   Run: ollama serve")
            
            logger.info(f"\n🌐 Frontend should connect to: http://localhost:{PORT}")
            logger.info("💬 Ready to chat!\n")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped")

if __name__ == "__main__":
    main() 