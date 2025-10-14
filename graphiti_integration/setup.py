#!/usr/bin/env python3
"""
Setup script for CPSC Regulation Knowledge Graph Integration
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 10):
        logger.error("Python 3.10 or higher is required")
        return False
    logger.info(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directories...")
    
    directories = [
        "logs",
        "data",
        "exports",
        "examples/output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created: {directory}")
    
    logger.info("✅ Directories created successfully")
    return True

def setup_environment():
    """Set up environment configuration"""
    logger.info("🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as f:
            content = f.read()
        with open(env_file, 'w') as f:
            content = content.replace("your_openai_api_key_here", "")
            content = content.replace("your_neo4j_password_here", "password")
            f.write(content)
        logger.info("✅ Environment file created from example")
    elif not env_file.exists():
        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("""# Graphiti Integration Environment Configuration
OPENAI_API_KEY=
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=cpsc_regulations
LOG_LEVEL=INFO
""")
        logger.info("✅ Basic environment file created")
    
    return True

def check_database():
    """Check if the regulation database exists"""
    logger.info("🗄️ Checking database...")
    
    db_path = Path("/workspace/regulations.db")
    if db_path.exists():
        logger.info(f"✅ Database found: {db_path}")
        return True
    else:
        logger.error(f"❌ Database not found: {db_path}")
        logger.info("Please ensure the regulations.db file exists in the workspace root")
        return False

def test_imports():
    """Test if all imports work correctly"""
    logger.info("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import networkx as nx
        logger.info("✅ Basic dependencies imported successfully")
        
        # Test Graphiti imports (may fail if not installed)
        try:
            from graphiti_core import Graphiti
            logger.info("✅ Graphiti imported successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Graphiti import failed: {e}")
            logger.info("This is expected if Graphiti is not installed yet")
        
        # Test local imports
        from config import get_config
        from data_loader import DataLoader
        logger.info("✅ Local modules imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

def run_basic_tests():
    """Run basic functionality tests"""
    logger.info("🧪 Running basic tests...")
    
    try:
        # Test data loading
        from data_loader import DataLoader
        data_loader = DataLoader()
        data = data_loader.load_from_sqlite()
        
        logger.info(f"✅ Data loading test passed: {len(data.sections)} sections loaded")
        
        # Test configuration
        from config import get_config, validate_config
        config = get_config()
        logger.info("✅ Configuration test passed")
        
        # Test entity creation
        from entities import RegulationSection, EntityType
        test_section = RegulationSection(
            id="test_section",
            name="Test Section",
            section_number="§ TEST.1",
            subject="Test Subject",
            text="Test text content",
            part_id=1,
            chapter_id=1,
            subchapter_id=1
        )
        logger.info("✅ Entity creation test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic tests failed: {e}")
        return False

def create_launch_scripts():
    """Create convenient launch scripts"""
    logger.info("🚀 Creating launch scripts...")
    
    # Create dashboard launch script
    dashboard_script = """#!/bin/bash
# Launch the Graphiti Dashboard
cd "$(dirname "$0")"
streamlit run graph_dashboard.py
"""
    
    with open("launch_dashboard.sh", "w") as f:
        f.write(dashboard_script)
    
    # Make executable
    os.chmod("launch_dashboard.sh", 0o755)
    
    # Create example runner script
    example_script = """#!/bin/bash
# Run basic usage examples
cd "$(dirname "$0")"
python examples/basic_usage.py
"""
    
    with open("run_examples.sh", "w") as f:
        f.write(example_script)
    
    os.chmod("run_examples.sh", 0o755)
    
    logger.info("✅ Launch scripts created")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)
    print()
    print("Next steps:")
    print()
    print("1. 🔑 Configure API Keys:")
    print("   Edit .env file and add your OpenAI API key:")
    print("   OPENAI_API_KEY=your_actual_api_key_here")
    print()
    print("2. 🗄️ Set up Neo4j (optional):")
    print("   - Install Neo4j Desktop or use Neo4j AuraDB")
    print("   - Update Neo4j credentials in .env file")
    print("   - Or use the built-in fallback mode")
    print()
    print("3. 🚀 Launch the dashboard:")
    print("   ./launch_dashboard.sh")
    print("   # or")
    print("   streamlit run graph_dashboard.py")
    print()
    print("4. 🧪 Run examples:")
    print("   ./run_examples.sh")
    print("   # or")
    print("   python examples/basic_usage.py")
    print()
    print("5. 📚 Explore the code:")
    print("   - graph_dashboard.py: Interactive visualization")
    print("   - integration_api.py: Programmatic access")
    print("   - examples/: Usage examples")
    print()
    print("6. 🔧 Customize:")
    print("   - config.py: Configuration settings")
    print("   - entities.py: Data models")
    print("   - graph_builder.py: Graph construction logic")
    print()
    print("For more information, see README.md")
    print("="*70)

def main():
    """Main setup function"""
    print("🚀 CPSC Regulation Knowledge Graph Integration Setup")
    print("="*60)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Check database
    if not check_database():
        print("⚠️  Database check failed, but continuing with setup...")
    
    # Test imports
    if not test_imports():
        print("⚠️  Some imports failed, but continuing with setup...")
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Dependency installation failed, but continuing with setup...")
    
    # Run basic tests
    if not run_basic_tests():
        print("⚠️  Basic tests failed, but continuing with setup...")
    
    # Create launch scripts
    if not create_launch_scripts():
        print("⚠️  Launch script creation failed, but continuing with setup...")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()