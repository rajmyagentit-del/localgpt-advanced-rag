import json
import logging

import requests

from rag_system.config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str | None = None):
        if base_url is None:
            base_url = settings.ollama_host
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> list[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code == 200:
                logger.info(f"Pulling model {model_name}...")
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            logger.info(f"Status: {data['status']}")
                        if data.get("status") == "success":
                            return True
                return True
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def chat(self, message: str, model: str = "llama3.2", conversation_history: list[dict] = None, enable_thinking: bool = True) -> str:
        """Send a chat message to Ollama"""
        if conversation_history is None:
            conversation_history = []
        
        # Add user message to conversation
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
            }
            
            # Multiple approaches to disable thinking tokens
            if not enable_thinking:
                payload.update({
                    "think": False,  # Native Ollama parameter
                    "options": {
                        "think": False,
                        "thinking": False,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                })
            else:
                payload["think"] = True
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["message"]["content"]
                
                # Additional cleanup: remove any thinking tokens that might slip through
                if not enable_thinking:
                    # Remove common thinking token patterns
                    import re
                    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
                    response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
                    response_text = response_text.strip()
                
                return response_text
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Connection error: {e}"
    
    def chat_stream(self, message: str, model: str = "llama3.2", conversation_history: list[dict] = None, enable_thinking: bool = True):
        """Stream chat response from Ollama"""
        if conversation_history is None:
            conversation_history = []
        
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            
            # Multiple approaches to disable thinking tokens
            if not enable_thinking:
                payload.update({
                    "think": False,  # Native Ollama parameter
                    "options": {
                        "think": False,
                        "thinking": False,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                })
            else:
                payload["think"] = True
            
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                
                                # Filter out thinking tokens in streaming mode
                                if not enable_thinking:
                                    # Skip content that looks like thinking tokens
                                    if '<think>' in content.lower() or '<thinking>' in content.lower():
                                        continue
                                
                                yield content
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            yield f"Connection error: {e}"

def main():
    """Test the Ollama client"""
    client = OllamaClient()
    
    # Check if Ollama is running
    if not client.is_ollama_running():
        logger.error("❌ Ollama is not running. Please start Ollama first.")
        logger.info("Install: https://ollama.ai")
        logger.info("Run: ollama serve")
        return
    
    logger.info("✅ Ollama is running!")
    
    # List available models
    models = client.list_models()
    logger.info(f"Available models: {models}")
    
    # Try to use llama3.2, pull if needed
    model_name = "llama3.2"
    if model_name not in [m.split(":")[0] for m in models]:
        logger.info(f"Model {model_name} not found. Pulling...")
        if client.pull_model(model_name):
            logger.info(f"✅ Model {model_name} pulled successfully!")
        else:
            logger.error(f"❌ Failed to pull model {model_name}")
            return
    
    # Test chat
    logger.info("\n🤖 Testing chat...")
    response = client.chat("Hello! Can you tell me a short joke?", model_name)
    logger.info(f"AI: {response}")

if __name__ == "__main__":
    main()    