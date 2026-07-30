#!/usr/bin/env python3
"""
Simple test script for the localGPT backend
"""

import logging

import requests

logger = logging.getLogger(__name__)


def test_health_endpoint():
    """Test the health endpoint"""
    logger.debug("🔍 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Health check passed")
            logger.info(f"   Ollama running: {data['ollama_running']}")
            logger.info(f"   Models available: {len(data['available_models'])}")
            return True
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


def test_chat_endpoint():
    """Test the chat endpoint"""
    logger.info("\n💬 Testing chat endpoint...")

    test_message = {"message": "Say 'Hello World' and nothing else.", "model": "llama3.2:latest"}

    try:
        response = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            json=test_message,
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            logger.info("✅ Chat test passed")
            logger.info(f"   Model: {data['model']}")
            logger.info(f"   Response: {data['response']}")
            logger.info(f"   Message count: {data['message_count']}")
            return True
        else:
            logger.error(f"❌ Chat test failed: {response.status_code}")
            logger.info(f"   Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Chat test failed: {e}")
        return False


def test_conversation_history():
    """Test conversation with history"""
    logger.info("\n🗨️  Testing conversation history...")

    # First message
    conversation = []

    message1 = {
        "message": "My name is Alice. Remember this.",
        "model": "llama3.2:latest",
        "conversation_history": conversation,
    }

    try:
        response1 = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            json=message1,
            timeout=30,
        )

        if response1.status_code == 200:
            data1 = response1.json()

            # Add to conversation history
            conversation.append({"role": "user", "content": "My name is Alice. Remember this."})
            conversation.append({"role": "assistant", "content": data1["response"]})

            # Second message asking about the name
            message2 = {
                "message": "What is my name?",
                "model": "llama3.2:latest",
                "conversation_history": conversation,
            }

            response2 = requests.post(
                "http://localhost:8000/chat",
                headers={"Content-Type": "application/json"},
                json=message2,
                timeout=30,
            )

            if response2.status_code == 200:
                data2 = response2.json()
                logger.info("✅ Conversation history test passed")
                logger.info(f"   First response: {data1['response']}")
                logger.info(f"   Second response: {data2['response']}")

                # Check if the AI remembered the name
                if "alice" in data2["response"].lower():
                    logger.info("✅ AI correctly remembered the name!")
                else:
                    logger.warning("⚠️  AI might not have remembered the name")
                return True
            else:
                logger.error(f"❌ Second message failed: {response2.status_code}")
                return False
        else:
            logger.error(f"❌ First message failed: {response1.status_code}")
            return False

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Conversation test failed: {e}")
        return False


def main():
    logger.info("🧪 Testing localGPT Backend")
    logger.info("=" * 40)

    # Test health endpoint
    health_ok = test_health_endpoint()
    if not health_ok:
        logger.error("\n❌ Backend server is not running or not healthy")
        logger.info("   Make sure to run: python server.py")
        return

    # Test basic chat
    chat_ok = test_chat_endpoint()
    if not chat_ok:
        logger.error("\n❌ Chat functionality is not working")
        return

    # Test conversation history
    conversation_ok = test_conversation_history()

    logger.info("\n" + "=" * 40)
    if health_ok and chat_ok and conversation_ok:
        logger.info("🎉 All tests passed! Backend is ready for frontend integration.")
    else:
        logger.error("⚠️  Some tests failed. Check the issues above.")

    logger.info("\n🔗 Ready to connect to frontend at http://localhost:3000")


if __name__ == "__main__":
    main()
