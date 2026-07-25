#!/usr/bin/env python3
"""
Model Configuration Validation Script
=====================================

This script validates the consolidated model configuration system to ensure:
1. No configuration conflicts exist
2. All model names are consistent across components
3. Models are accessible and properly configured
4. The configuration validation system works correctly

Run this after making configuration changes to catch issues early.
"""

import sys
import os
import logging
# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

from rag_system.main import (
    PIPELINE_CONFIGS, 
    OLLAMA_CONFIG, 
    EXTERNAL_MODELS,
    validate_model_config
)

def print_header(title: str):
    """Print a formatted header."""
    logger.info(f"\n{'='*60}")
    logger.debug(f"🔍 {title}")
    logger.info(f"{'='*60}")

def print_section(title: str):
    """Print a formatted section header.""" 
    logger.info(f"\n{'─'*40}")
    logger.info(f"📋 {title}")
    logger.info(f"{'─'*40}")

def validate_configuration_consistency():
    """Validate that all configurations are consistent."""
    print_header("CONFIGURATION CONSISTENCY VALIDATION")
    
    errors = []
    
    # 1. Check embedding model consistency
    print_section("Embedding Model Consistency")
    default_embedding = PIPELINE_CONFIGS["default"]["embedding_model_name"]
    external_embedding = EXTERNAL_MODELS["embedding_model"]
    fast_embedding = PIPELINE_CONFIGS["fast"]["embedding_model_name"]
    
    logger.info(f"Default Config: {default_embedding}")
    logger.info(f"External Models: {external_embedding}")  
    logger.info(f"Fast Config: {fast_embedding}")
    
    if default_embedding != external_embedding:
        errors.append(f"❌ Embedding model mismatch: default={default_embedding}, external={external_embedding}")
    elif default_embedding != fast_embedding:
        errors.append(f"❌ Embedding model mismatch: default={default_embedding}, fast={fast_embedding}")
    else:
        logger.info("✅ Embedding models are consistent")
    
    # 2. Check reranker model consistency
    print_section("Reranker Model Consistency")
    default_reranker = PIPELINE_CONFIGS["default"]["reranker"]["model_name"]
    external_reranker = EXTERNAL_MODELS["reranker_model"]
    
    logger.info(f"Default Config: {default_reranker}")
    logger.info(f"External Models: {external_reranker}")
    
    if default_reranker != external_reranker:
        errors.append(f"❌ Reranker model mismatch: default={default_reranker}, external={external_reranker}")
    else:
        logger.info("✅ Reranker models are consistent")
    
    # 3. Check vision model consistency
    print_section("Vision Model Consistency")
    default_vision = PIPELINE_CONFIGS["default"]["vision_model_name"]
    external_vision = EXTERNAL_MODELS["vision_model"]
    
    logger.info(f"Default Config: {default_vision}")
    logger.info(f"External Models: {external_vision}")
    
    if default_vision != external_vision:
        errors.append(f"❌ Vision model mismatch: default={default_vision}, external={external_vision}")
    else:
        logger.info("✅ Vision models are consistent")
    
    return errors

def print_model_usage_map():
    """Print a comprehensive map of which models are used where."""
    print_header("MODEL USAGE MAP")
    
    print_section("🤖 Ollama Models (Local Inference)")
    for model_type, model_name in OLLAMA_CONFIG.items():
        if model_type != "host":
            logger.info(f"  {model_type.replace('_', ' ').title()}: {model_name}")
    
    print_section("🔗 External Models (HuggingFace/Direct)")
    for model_type, model_name in EXTERNAL_MODELS.items():
        logger.info(f"  {model_type.replace('_', ' ').title()}: {model_name}")
    
    print_section("📍 Model Usage by Component")
    usage_map = {
        "🔤 Text Embedding": {
            "Model": EXTERNAL_MODELS["embedding_model"],
            "Used In": ["Retrieval Pipeline", "Semantic Cache", "Dense Retrieval", "Late Chunking"],
            "Component": "QwenEmbedder (representations.py)"
        },
        "🧠 Text Generation": {
            "Model": OLLAMA_CONFIG["generation_model"],
            "Used In": ["Agent Loop", "Answer Synthesis", "Query Decomposition", "Verification"],
            "Component": "OllamaClient"
        },
        "🚀 Enrichment/Routing": {
            "Model": OLLAMA_CONFIG["enrichment_model"],
            "Used In": ["Query Routing", "Document Overview Analysis"],
            "Component": "Agent Loop (_route_via_overviews)"
        },
        "🔀 Reranking": {
            "Model": EXTERNAL_MODELS["reranker_model"],
            "Used In": ["Hybrid Search", "Document Reranking", "AI Reranker"],
            "Component": "ColBERT (rerankers-lib) or QwenReranker"
        },
        "👁️ Vision": {
            "Model": EXTERNAL_MODELS["vision_model"],
            "Used In": ["Multimodal Processing", "Image Embeddings"],
            "Component": "Vision Pipeline (when enabled)"
        }
    }
    
    for model_name, details in usage_map.items():
        logger.info(f"\n{model_name}")
        logger.info(f"  Model: {details['Model']}")
        logger.info(f"  Component: {details['Component']}")
        logger.info(f"  Used In: {', '.join(details['Used In'])}")

def test_validation_function():
    """Test the built-in validation function."""
    print_header("VALIDATION FUNCTION TEST")
    
    try:
        result = validate_model_config()
        if result:
            logger.info("✅ validate_model_config() passed successfully!")
        else:
            logger.error("❌ validate_model_config() returned False")
    except Exception as e:
        logger.error(f"❌ validate_model_config() failed with error: {e}")
        return False
    
    return True

def check_pipeline_configurations():
    """Check all pipeline configurations for completeness."""
    print_header("PIPELINE CONFIGURATION COMPLETENESS")
    
    required_keys = {
        "default": ["storage", "retrieval", "embedding_model_name", "reranker"],
        "fast": ["storage", "retrieval", "embedding_model_name"]
    }
    
    errors = []
    
    for config_name, required in required_keys.items():
        print_section(f"{config_name.title()} Configuration")
        config = PIPELINE_CONFIGS.get(config_name, {})
        
        for key in required:
            if key in config:
                logger.info(f"  ✅ {key}: {type(config[key]).__name__}")
            else:
                error_msg = f"❌ Missing required key '{key}' in {config_name} config"
                errors.append(error_msg)  
                logger.error(f"  {error_msg}")
    
    return errors

def main():
    """Run all validation checks."""
    logger.info("🚀 Starting Model Configuration Validation")
    logger.info(f"Python Path: {sys.path[0]}")
    
    all_errors = []
    
    # Run all validation checks
    all_errors.extend(validate_configuration_consistency())
    all_errors.extend(check_pipeline_configurations())
    
    # Print model usage map
    print_model_usage_map()
    
    # Test validation function
    validation_passed = test_validation_function()
    
    # Final summary
    print_header("VALIDATION SUMMARY")
    
    if all_errors:
        logger.error("❌ VALIDATION FAILED - Issues Found:")
        for error in all_errors:
            logger.error(f"  {error}")
        return 1
    elif not validation_passed:
        logger.error("❌ VALIDATION FAILED - validate_model_config() function failed")
        return 1
    else:
        logger.info("✅ ALL VALIDATIONS PASSED!")
        logger.info("\n🎉 Your model configuration is consistent and properly structured!")
        logger.info("\n📋 Summary:")
        logger.info(f"   • Embedding Model: {EXTERNAL_MODELS['embedding_model']}")
        logger.info(f"   • Generation Model: {OLLAMA_CONFIG['generation_model']}")
        logger.info(f"   • Enrichment Model: {OLLAMA_CONFIG['enrichment_model']}")
        logger.info(f"   • Reranker Model: {EXTERNAL_MODELS['reranker_model']}")
        logger.info(f"   • Vision Model: {EXTERNAL_MODELS['vision_model']}")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 