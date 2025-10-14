#!/usr/bin/env python3
"""
Configuration settings for Graphiti integration with CPSC Regulation Agent
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
WORKSPACE_ROOT = Path("/workspace")
PROJECT_ROOT = WORKSPACE_ROOT / "graphiti_integration"
DATA_ROOT = WORKSPACE_ROOT

# Database configuration
DATABASE_CONFIG = {
    "sqlite_path": str(DATA_ROOT / "regulations.db"),
    "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
    "neo4j_password": os.getenv("NEO4J_PASSWORD", "password"),
    "neo4j_database": os.getenv("NEO4J_DATABASE", "cpsc_regulations")
}

# Graphiti configuration
GRAPHITI_CONFIG = {
    "group_id": "cpsc_regulations",
    "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "gpt-3.5-turbo",
    "max_episode_length": 1000,
    "similarity_threshold": 0.7,
    "confidence_threshold": 0.8
}

# OpenAI configuration
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-3.5-turbo",
    "temperature": 0.1,
    "max_tokens": 1000
}

# Analysis configuration
ANALYSIS_CONFIG = {
    "min_similarity_score": 0.7,
    "min_confidence_score": 0.8,
    "max_relationships_per_entity": 50,
    "hierarchical_weight": 0.3,
    "semantic_weight": 0.7,
    "batch_size": 100
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "max_nodes_display": 500,
    "max_edges_display": 1000,
    "node_size_range": (10, 50),
    "edge_width_range": (1, 5),
    "color_scheme": "viridis",
    "layout_algorithm": "force_directed"
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "title": "CPSC Regulation Knowledge Graph",
    "page_icon": "🕸️",
    "layout": "wide",
    "sidebar_state": "expanded",
    "theme": "light"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(PROJECT_ROOT / "logs" / "graphiti_integration.log")
}

# Create logs directory
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Entity type mappings
ENTITY_TYPES = {
    "regulation_section": "RegulationSection",
    "regulation_part": "RegulationPart", 
    "regulation_chapter": "RegulationChapter",
    "compliance_requirement": "ComplianceRequirement",
    "safety_standard": "SafetyStandard",
    "product_category": "ProductCategory",
    "hazard_type": "HazardType"
}

# Relationship types
RELATIONSHIP_TYPES = {
    "hierarchical": ["CONTAINS", "BELONGS_TO", "PART_OF"],
    "semantic": ["SIMILAR_TO", "RELATED_TO", "REFERENCES"],
    "compliance": ["REQUIRES", "PROHIBITS", "MANDATES"],
    "temporal": ["SUPERSEDES", "AMENDS", "REPLACES"],
    "cross_reference": ["CITES", "REFERENCES", "DEFINES"]
}

# Search configuration
SEARCH_CONFIG = {
    "default_limit": 20,
    "max_results": 100,
    "similarity_threshold": 0.7,
    "include_embeddings": True,
    "include_metadata": True
}

# Export configuration
EXPORT_CONFIG = {
    "formats": ["json", "csv", "graphml", "gexf"],
    "include_embeddings": False,
    "include_metadata": True,
    "compression": True
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "database": DATABASE_CONFIG,
        "graphiti": GRAPHITI_CONFIG,
        "openai": OPENAI_CONFIG,
        "analysis": ANALYSIS_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "dashboard": DASHBOARD_CONFIG,
        "logging": LOGGING_CONFIG,
        "entity_types": ENTITY_TYPES,
        "relationship_types": RELATIONSHIP_TYPES,
        "search": SEARCH_CONFIG,
        "export": EXPORT_CONFIG
    }

def validate_config() -> bool:
    """Validate configuration settings"""
    required_env_vars = ["OPENAI_API_KEY"]
    
    for var in required_env_vars:
        if not os.getenv(var):
            print(f"Warning: {var} not set in environment variables")
            return False
    
    # Check if SQLite database exists
    if not Path(DATABASE_CONFIG["sqlite_path"]).exists():
        print(f"Error: SQLite database not found at {DATABASE_CONFIG['sqlite_path']}")
        return False
    
    return True

if __name__ == "__main__":
    config = get_config()
    print("Configuration loaded successfully")
    print(f"SQLite database: {DATABASE_CONFIG['sqlite_path']}")
    print(f"Neo4j URI: {DATABASE_CONFIG['neo4j_uri']}")
    print(f"Group ID: {GRAPHITI_CONFIG['group_id']}")
    
    if validate_config():
        print("✅ Configuration validation passed")
    else:
        print("❌ Configuration validation failed")