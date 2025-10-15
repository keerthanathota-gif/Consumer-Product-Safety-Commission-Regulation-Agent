#!/usr/bin/env python3
"""
Test script to verify CPSC Knowledge Graph installation
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Test Python version"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Requires 3.8+")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'pandas', 'sklearn', 'pydantic', 'dotenv', 
        'streamlit', 'plotly', 'networkx', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'dotenv':
                import dotenv
            else:
                __import__(package)
            print(f"   ✅ {package} - OK")
        except ImportError:
            print(f"   ❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n   Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def test_database():
    """Test database availability"""
    print("\n🗄️ Testing database...")
    
    # Check for database in parent directory
    parent_db = Path("../regulations.db")
    current_db = Path("regulations.db")
    
    if parent_db.exists():
        print(f"   ✅ Database found: {parent_db.absolute()}")
        return str(parent_db.absolute())
    elif current_db.exists():
        print(f"   ✅ Database found: {current_db.absolute()}")
        return str(current_db.absolute())
    else:
        print("   ❌ Database not found")
        print("   Expected: regulations.db in current or parent directory")
        return None

def test_knowledge_graph():
    """Test knowledge graph functionality"""
    print("\n🕸️ Testing knowledge graph...")
    
    try:
        from simple_knowledge_graph import SimpleKnowledgeGraph
        print("   ✅ Import successful")
        
        # Test graph creation
        graph = SimpleKnowledgeGraph()
        print("   ✅ Graph object created")
        
        # Test graph building
        graph.build_graph()
        print("   ✅ Graph built successfully")
        
        # Test basic functionality
        stats = graph.get_graph_statistics()
        print(f"   ✅ Graph stats: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        
        # Test search
        results = graph.search("test", limit=1)
        print(f"   ✅ Search test: {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_dashboard():
    """Test dashboard functionality"""
    print("\n📊 Testing dashboard...")
    
    try:
        import streamlit
        print("   ✅ Streamlit available")
        
        # Check if dashboard file exists
        if Path("analysis_dashboard.py").exists():
            print("   ✅ Dashboard file found")
            return True
        else:
            print("   ❌ Dashboard file not found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 CPSC Knowledge Graph - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Database", lambda: test_database() is not None),
        ("Knowledge Graph", test_knowledge_graph),
        ("Dashboard", test_dashboard)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"   ❌ {test_name} test failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Run: python3 simple_knowledge_graph.py")
        print("2. Or run: ./launch_dashboard.sh")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check database location")
        print("3. Verify Python version (3.8+)")
    
    print("=" * 50)

if __name__ == "__main__":
    main()