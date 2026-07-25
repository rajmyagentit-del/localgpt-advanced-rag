#!/usr/bin/env python3

import os
import sys
import logging

logger = logging.getLogger(__name__)

def test_ollama_connectivity():
    """Test Ollama connectivity from within Docker container"""
    logger.info("🧪 Testing Ollama Connectivity")
    logger.info("=" * 40)
    
    ollama_host = os.getenv('OLLAMA_HOST', 'Not set')
    logger.info(f"OLLAMA_HOST environment variable: {ollama_host}")
    
    try:
        from ollama_client import OllamaClient
        client = OllamaClient()
        logger.info(f"OllamaClient base_url: {client.base_url}")
        
        is_running = client.is_ollama_running()
        logger.info(f"Ollama running: {is_running}")
        
        if is_running:
            models = client.list_models()
            logger.info(f"Available models: {models}")
            logger.info("✅ Ollama connectivity test passed!")
            return True
        else:
            logger.error("❌ Ollama connectivity test failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing Ollama connectivity: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_connectivity()
    sys.exit(0 if success else 1)
