#!/usr/bin/env python3
"""
Simple test script for CPSC Regulation Knowledge Graph Integration
Tests basic functionality without requiring API keys
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from entities import RegulationSection, EntityType
import json

def test_data_loading():
    """Test data loading functionality"""
    print("🧪 Testing data loading...")
    
    try:
        data_loader = DataLoader()
        data = data_loader.load_from_sqlite()
        
        print(f"✅ Data loaded successfully:")
        print(f"   Chapters: {len(data.chapters)}")
        print(f"   Parts: {len(data.parts)}")
        print(f"   Sections: {len(data.sections)}")
        
        # Test sample section
        if data.sections:
            sample = data.sections[0]
            print(f"   Sample section: {sample.section_number} - {sample.subject}")
            print(f"   Word count: {sample.word_count}")
            print(f"   Sentence count: {sample.sentence_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_entity_creation():
    """Test entity creation"""
    print("\n🧪 Testing entity creation...")
    
    try:
        # Create a test regulation section
        test_section = RegulationSection(
            id="test_section_1",
            name="§ TEST.1",
            description="Test Section",
            section_number="§ TEST.1",
            subject="Test Subject",
            text="This is a test regulation section for demonstration purposes.",
            citation="Test Citation",
            part_id=1,
            chapter_id=1,
            subchapter_id=1,
            word_count=12,
            sentence_count=1
        )
        
        print(f"✅ Entity created successfully:")
        print(f"   ID: {test_section.id}")
        print(f"   Name: {test_section.name}")
        print(f"   Type: {test_section.entity_type}")
        print(f"   Word count: {test_section.word_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Entity creation failed: {e}")
        return False

def test_hierarchical_structure():
    """Test hierarchical structure loading"""
    print("\n🧪 Testing hierarchical structure...")
    
    try:
        data_loader = DataLoader()
        structure = data_loader.get_hierarchical_structure()
        
        print(f"✅ Hierarchical structure loaded:")
        print(f"   Total sections: {structure['total_sections']}")
        print(f"   Total parts: {structure['total_parts']}")
        print(f"   Total chapters: {structure['total_chapters']}")
        
        # Show first chapter structure
        if structure['chapters']:
            first_chapter = structure['chapters'][0]
            print(f"   First chapter: {first_chapter['name']}")
            print(f"   Subchapters: {len(first_chapter['subchapters'])}")
            
            if first_chapter['subchapters']:
                first_subchapter = first_chapter['subchapters'][0]
                print(f"   First subchapter parts: {len(first_subchapter['parts'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical structure test failed: {e}")
        return False

def test_search_functionality():
    """Test search functionality"""
    print("\n🧪 Testing search functionality...")
    
    try:
        data_loader = DataLoader()
        
        # Test search by text content
        search_results = data_loader.search_sections("safety", limit=5)
        
        print(f"✅ Search completed:")
        print(f"   Found {len(search_results)} sections containing 'safety'")
        
        for i, result in enumerate(search_results[:3], 1):
            print(f"   {i}. {result.section_number}: {result.subject}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import get_config, validate_config
        
        config = get_config()
        print(f"✅ Configuration loaded:")
        print(f"   Database path: {config['database']['sqlite_path']}")
        print(f"   Group ID: {config['graphiti']['group_id']}")
        print(f"   Similarity threshold: {config['analysis']['min_similarity_score']}")
        
        # Test validation (will fail without API keys, but that's expected)
        validation_result = validate_config()
        if validation_result:
            print("   Configuration validation: ✅ Passed")
        else:
            print("   Configuration validation: ⚠️  Failed (expected without API keys)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_export_functionality():
    """Test export functionality"""
    print("\n🧪 Testing export functionality...")
    
    try:
        data_loader = DataLoader()
        data = data_loader.load_from_sqlite()
        
        # Test JSON export
        export_data = {
            "metadata": {
                "total_sections": len(data.sections),
                "total_parts": len(data.parts),
                "total_chapters": len(data.chapters),
                "export_timestamp": "2024-01-01T00:00:00Z"
            },
            "sections": [
                {
                    "id": section.id,
                    "section_number": section.section_number,
                    "subject": section.subject,
                    "word_count": section.word_count,
                    "sentence_count": section.sentence_count
                }
                for section in data.sections[:10]  # First 10 sections
            ]
        }
        
        json_output = json.dumps(export_data, indent=2)
        print(f"✅ Export test completed:")
        print(f"   JSON size: {len(json_output)} characters")
        print(f"   Sections exported: {len(export_data['sections'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CPSC Regulation Knowledge Graph - Simple Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_data_loading,
        test_entity_creation,
        test_hierarchical_structure,
        test_search_functionality,
        test_configuration,
        test_export_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The basic integration is working correctly.")
        print("\nNext steps:")
        print("1. Add your OpenAI API key to .env file")
        print("2. Set up Neo4j database (optional)")
        print("3. Run the full dashboard: streamlit run graph_dashboard.py")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()