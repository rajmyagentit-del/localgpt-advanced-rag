#!/usr/bin/env python3
"""
Demo Batch Indexing Script for LocalGPT RAG System

This script demonstrates how to perform batch indexing of multiple documents
using configuration files. It's designed to showcase the full capabilities
of the indexing pipeline with various configuration options.

Usage:
    python demo_batch_indexing.py --config batch_indexing_config.json
    python demo_batch_indexing.py --create-sample-config
    python demo_batch_indexing.py --help
"""

import os
import sys
import json
import argparse
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Add the project root to the path so we can import rag_system modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_system.main import PIPELINE_CONFIGS
    from rag_system.pipelines.indexing_pipeline import IndexingPipeline
    from rag_system.utils.ollama_client import OllamaClient
    from backend.database import ChatDatabase
except ImportError as e:
    logger.error(f"❌ Error importing required modules: {e}")
    logger.info("Please ensure you're running this script from the project root directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


class BatchIndexingDemo:
    """Demonstration of batch indexing capabilities."""
    
    def __init__(self, config_path: str):
        """Initialize the batch indexing demo."""
        self.config_path = config_path
        self.config = self._load_config()
        self.db = ChatDatabase()
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient()
        
        # Initialize pipeline with merged configuration
        self.pipeline_config = self._merge_configurations()
        self.pipeline = IndexingPipeline(
            self.pipeline_config,
            self.ollama_client,
            self.config.get("ollama_config", {
                "generation_model": "qwen3:0.6b",
                "embedding_model": "qwen3:0.6b"
            })
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load batch indexing configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def _merge_configurations(self) -> Dict[str, Any]:
        """Merge batch config with default pipeline config."""
        # Start with default pipeline configuration
        merged_config = PIPELINE_CONFIGS.get("default", {}).copy()
        
        # Override with batch-specific settings
        batch_settings = self.config.get("pipeline_settings", {})
        
        # Deep merge for nested dictionaries
        def deep_merge(base: dict, override: dict) -> dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(merged_config, batch_settings)
    
    def validate_documents(self, documents: List[str]) -> List[str]:
        """Validate and filter document paths."""
        valid_documents = []
        
        logger.info(f"📋 Validating {len(documents)} documents...")
        
        for doc_path in documents:
            # Handle relative paths
            if not os.path.isabs(doc_path):
                doc_path = os.path.abspath(doc_path)
            
            if os.path.exists(doc_path):
                # Check file extension
                ext = Path(doc_path).suffix.lower()
                if ext in ['.pdf', '.txt', '.docx', '.md', '.html', '.htm']:
                    valid_documents.append(doc_path)
                    logger.info(f"  ✅ {doc_path}")
                else:
                    logger.warning(f"  ⚠️  Unsupported file type: {doc_path}")
            else:
                logger.error(f"  ❌ File not found: {doc_path}")
        
        logger.info(f"📊 {len(valid_documents)} valid documents found")
        return valid_documents
    
    def create_indexes(self) -> List[str]:
        """Create multiple indexes based on configuration."""
        indexes = self.config.get("indexes", [])
        created_indexes = []
        
        for index_config in indexes:
            index_id = self.create_single_index(index_config)
            if index_id:
                created_indexes.append(index_id)
        
        return created_indexes
    
    def create_single_index(self, index_config: Dict[str, Any]) -> Optional[str]:
        """Create a single index from configuration."""
        try:
            # Extract index metadata
            index_name = index_config.get("name", "Unnamed Index")
            index_description = index_config.get("description", "")
            documents = index_config.get("documents", [])
            
            if not documents:
                logger.warning(f"⚠️  No documents specified for index '{index_name}', skipping...")
                return None
            
            # Validate documents
            valid_documents = self.validate_documents(documents)
            if not valid_documents:
                logger.error(f"❌ No valid documents found for index '{index_name}'")
                return None
            
            logger.info(f"\n🚀 Creating index: {index_name}")
            logger.info(f"📄 Processing {len(valid_documents)} documents")
            
            # Create index record in database
            index_metadata = {
                "created_by": "demo_batch_indexing.py",
                "created_at": datetime.now().isoformat(),
                "document_count": len(valid_documents),
                "config_used": index_config.get("processing_options", {})
            }
            
            index_id = self.db.create_index(
                name=index_name,
                description=index_description,
                metadata=index_metadata
            )
            
            # Add documents to index
            for doc_path in valid_documents:
                filename = os.path.basename(doc_path)
                self.db.add_document_to_index(index_id, filename, doc_path)
            
            # Process documents through pipeline
            start_time = time.time()
            self.pipeline.process_documents(valid_documents)
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Index '{index_name}' created successfully!")
            logger.info(f"   Index ID: {index_id}")
            logger.info(f"   Processing time: {processing_time:.2f} seconds")
            logger.info(f"   Documents processed: {len(valid_documents)}")
            
            return index_id
            
        except Exception as e:
            logger.error(f"❌ Error creating index '{index_name}': {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def demonstrate_features(self):
        """Demonstrate various indexing features."""
        logger.info("\n🎯 Batch Indexing Demo Features:")
        logger.info("=" * 50)
        
        # Show configuration
        logger.info(f"📋 Configuration file: {self.config_path}")
        logger.info(f"📊 Number of indexes to create: {len(self.config.get('indexes', []))}")
        
        # Show pipeline settings
        pipeline_settings = self.config.get("pipeline_settings", {})
        if pipeline_settings:
            logger.info("\n⚙️  Pipeline Settings:")
            for key, value in pipeline_settings.items():
                logger.info(f"   {key}: {value}")
        
        # Show model configuration
        ollama_config = self.config.get("ollama_config", {})
        if ollama_config:
            logger.info("\n🤖 Model Configuration:")
            for key, value in ollama_config.items():
                logger.info(f"   {key}: {value}")
    
    def run_demo(self):
        """Run the complete batch indexing demo."""
        logger.info("🚀 LocalGPT Batch Indexing Demo")
        logger.info("=" * 50)
        
        # Show demo features
        self.demonstrate_features()
        
        # Create indexes
        logger.info(f"\n📚 Starting batch indexing process...")
        start_time = time.time()
        
        created_indexes = self.create_indexes()
        
        total_time = time.time() - start_time
        
        # Summary
        logger.info(f"\n📊 Batch Indexing Summary")
        logger.info("=" * 50)
        logger.info(f"✅ Successfully created {len(created_indexes)} indexes")
        logger.info(f"⏱️  Total processing time: {total_time:.2f} seconds")
        
        if created_indexes:
            logger.info(f"\n📋 Created Indexes:")
            for i, index_id in enumerate(created_indexes, 1):
                index_info = self.db.get_index(index_id)
                if index_info:
                    logger.info(f"   {i}. {index_info['name']} ({index_id[:8]}...)")
                    logger.info(f"      Documents: {len(index_info.get('documents', []))}")
        
        logger.info(f"\n🎉 Demo completed successfully!")
        logger.info(f"💡 You can now use these indexes in the LocalGPT interface.")


def create_sample_config():
    """Create a comprehensive sample configuration file."""
    sample_config = {
        "description": "Demo batch indexing configuration showcasing various features",
        "pipeline_settings": {
            "embedding_model_name": "Qwen/Qwen3-Embedding-0.6B",
            "indexing": {
                "embedding_batch_size": 50,
                "enrichment_batch_size": 25,
                "enable_progress_tracking": True
            },
            "contextual_enricher": {
                "enabled": True,
                "window_size": 2,
                "model_name": "qwen3:0.6b"
            },
            "chunking": {
                "chunk_size": 512,
                "chunk_overlap": 64,
                "enable_latechunk": True,
                "enable_docling": True
            },
            "retrievers": {
                "dense": {
                    "enabled": True,
                    "lancedb_table_name": "demo_text_pages"
                },
                "bm25": {
                    "enabled": True,
                    "index_name": "demo_bm25_index"
                }
            },
            "storage": {
                "lancedb_uri": "./index_store/lancedb",
                "bm25_path": "./index_store/bm25"
            }
        },
        "ollama_config": {
            "generation_model": "qwen3:0.6b",
            "embedding_model": "qwen3:0.6b"
        },
        "indexes": [
            {
                "name": "Sample Invoice Collection",
                "description": "Demo index containing sample invoice documents",
                "documents": [
                    "./rag_system/documents/invoice_1039.pdf",
                    "./rag_system/documents/invoice_1041.pdf"
                ],
                "processing_options": {
                    "chunk_size": 512,
                    "enable_enrichment": True,
                    "retrieval_mode": "hybrid"
                }
            },
            {
                "name": "Research Papers Demo",
                "description": "Demo index for research papers and whitepapers",
                "documents": [
                    "./rag_system/documents/Newwhitepaper_Agents2.pdf"
                ],
                "processing_options": {
                    "chunk_size": 1024,
                    "enable_enrichment": True,
                    "retrieval_mode": "dense"
                }
            }
        ]
    }
    
    config_filename = "batch_indexing_config.json"
    with open(config_filename, "w") as f:
        json.dump(sample_config, f, indent=2)
    
    logger.info(f"✅ Sample configuration created: {config_filename}")
    logger.info(f"📝 Edit this file to customize your batch indexing setup")
    logger.info(f"🚀 Run: python demo_batch_indexing.py --config {config_filename}")


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="LocalGPT Batch Indexing Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_batch_indexing.py --config batch_indexing_config.json
  python demo_batch_indexing.py --create-sample-config
  
This demo showcases the advanced batch indexing capabilities of LocalGPT,
including multi-index creation, advanced configuration options, and
comprehensive processing pipelines.
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="batch_indexing_config.json",
        help="Path to batch indexing configuration file"
    )
    
    parser.add_argument(
        "--create-sample-config",
        action="store_true",
        help="Create a sample configuration file"
    )
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    if not os.path.exists(args.config):
        logger.error(f"❌ Configuration file not found: {args.config}")
        logger.info(f"💡 Create a sample config with: python {sys.argv[0]} --create-sample-config")
        sys.exit(1)
    
    try:
        demo = BatchIndexingDemo(args.config)
        demo.run_demo()
        
    except KeyboardInterrupt:
        logger.error("\n\n❌ Demo cancelled by user.")
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()  