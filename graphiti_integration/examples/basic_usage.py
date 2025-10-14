#!/usr/bin/env python3
"""
Basic Usage Examples for CPSC Regulation Knowledge Graph Integration
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from integration_api import GraphitiIntegration
from graph_builder import RegulationGraphBuilder
from data_loader import DataLoader
import json

def example_basic_search():
    """Example: Basic search functionality"""
    print("🔍 Example: Basic Search")
    print("=" * 50)
    
    try:
        # Initialize integration
        integration = GraphitiIntegration()
        
        # Search for safety-related regulations
        results = integration.search_regulations("safety requirements", limit=5)
        
        print(f"Found {len(results)} results for 'safety requirements':")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Content: {result['content'][:100]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_related_regulations():
    """Example: Finding related regulations"""
    print("🔗 Example: Related Regulations")
    print("=" * 50)
    
    try:
        integration = GraphitiIntegration()
        
        # Get related regulations for a specific section
        related = integration.get_related_regulations("§ 1000.1", limit=5)
        
        print(f"Regulations related to § 1000.1:")
        for i, rel in enumerate(related, 1):
            print(f"{i}. {rel['section_number']}: {rel['subject']}")
            print(f"   Relationship: {rel['relationship_type']} (weight: {rel['relationship_weight']:.3f})")
            print(f"   Context: {rel['context'] or 'N/A'}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_regulatory_clusters():
    """Example: Finding regulatory clusters"""
    print("📊 Example: Regulatory Clusters")
    print("=" * 50)
    
    try:
        integration = GraphitiIntegration()
        
        # Find clusters of related regulations
        clusters = integration.find_regulatory_clusters(min_cluster_size=3)
        
        print(f"Found {len(clusters)} regulatory clusters:")
        for i, cluster in enumerate(clusters, 1):
            print(f"Cluster {i}: {cluster['size']} regulations")
            for section in cluster['sections'][:3]:  # Show first 3
                print(f"  - {section['section_number']}: {section['subject']}")
            if len(cluster['sections']) > 3:
                print(f"  ... and {len(cluster['sections']) - 3} more")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_relationship_analysis():
    """Example: Analyzing relationships between regulations"""
    print("🔍 Example: Relationship Analysis")
    print("=" * 50)
    
    try:
        integration = GraphitiIntegration()
        
        # Analyze relationship between two specific sections
        analysis = integration.analyze_relationships("§ 1000.1", "§ 1000.2")
        
        print(f"Relationship analysis between § 1000.1 and § 1000.2:")
        print(f"Direct relationships: {len(analysis.get('direct_relationships', []))}")
        print(f"Common neighbors: {analysis.get('common_neighbors', 0)}")
        print(f"Similarity score: {analysis.get('similarity_score', 0):.3f}")
        print(f"Section 1 degree: {analysis.get('section1_degree', 0)}")
        print(f"Section 2 degree: {analysis.get('section2_degree', 0)}")
        
        if analysis.get('direct_relationships'):
            print("\nDirect relationships:")
            for rel in analysis['direct_relationships']:
                print(f"  - {rel['type']} (weight: {rel['weight']:.3f}, confidence: {rel['confidence']:.3f})")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def example_compliance_requirements():
    """Example: Extracting compliance requirements"""
    print("⚖️ Example: Compliance Requirements")
    print("=" * 50)
    
    try:
        integration = GraphitiIntegration()
        
        # Get compliance requirements for a specific section
        requirements = integration.get_compliance_requirements("§ 1000.1")
        
        print(f"Compliance requirements for § 1000.1:")
        if requirements:
            for i, req in enumerate(requirements, 1):
                print(f"{i}. [{req['type']}] {req['text']}")
        else:
            print("No compliance requirements found")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_network_analysis():
    """Example: Network analysis around a regulation"""
    print("🕸️ Example: Network Analysis")
    print("=" * 50)
    
    try:
        integration = GraphitiIntegration()
        
        # Get network around a specific regulation
        network = integration.get_regulation_network("§ 1000.1", depth=2)
        
        if 'error' in network:
            print(f"Error: {network['error']}")
            return
        
        print(f"Network around § 1000.1 (depth 2):")
        print(f"Nodes: {len(network['nodes'])}")
        print(f"Edges: {len(network['edges'])}")
        
        # Show nodes by depth
        depth_counts = {}
        for node in network['nodes']:
            depth = node['depth']
            if depth not in depth_counts:
                depth_counts[depth] = 0
            depth_counts[depth] += 1
        
        print("\nNodes by depth:")
        for depth in sorted(depth_counts.keys()):
            print(f"  Depth {depth}: {depth_counts[depth]} nodes")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_graph_statistics():
    """Example: Getting graph statistics"""
    print("📈 Example: Graph Statistics")
    print("=" * 50)
    
    try:
        integration = GraphitiIntegration()
        
        # Get comprehensive graph statistics
        stats = integration.get_graph_statistics()
        
        print("Knowledge Graph Statistics:")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Total edges: {stats['total_edges']}")
        print(f"Average degree: {stats['average_degree']:.2f}")
        print(f"Max degree: {stats['max_degree']}")
        print(f"Min degree: {stats['min_degree']}")
        print(f"Average weight: {stats['average_weight']:.3f}")
        print(f"Max weight: {stats['max_weight']:.3f}")
        print(f"Min weight: {stats['min_weight']:.3f}")
        
        print("\nNode types:")
        for node_type, count in stats['node_types'].items():
            print(f"  {node_type}: {count}")
        
        print("\nRelationship types:")
        for rel_type, count in stats['relationship_types'].items():
            print(f"  {rel_type}: {count}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_data_loading():
    """Example: Loading and exploring data"""
    print("📊 Example: Data Loading")
    print("=" * 50)
    
    try:
        # Load data directly
        data_loader = DataLoader()
        data = data_loader.load_from_sqlite()
        
        print(f"Loaded data:")
        print(f"  Chapters: {len(data.chapters)}")
        print(f"  Parts: {len(data.parts)}")
        print(f"  Sections: {len(data.sections)}")
        
        # Show sample section
        if data.sections:
            sample = data.sections[0]
            print(f"\nSample section:")
            print(f"  Number: {sample.section_number}")
            print(f"  Subject: {sample.subject}")
            print(f"  Text preview: {sample.text[:100]}...")
            print(f"  Word count: {sample.word_count}")
            print(f"  Sentence count: {sample.sentence_count}")
        
        # Show hierarchical structure
        structure = data_loader.get_hierarchical_structure()
        print(f"\nHierarchical structure:")
        print(f"  Total sections: {structure['total_sections']}")
        print(f"  Total parts: {structure['total_parts']}")
        print(f"  Total chapters: {structure['total_chapters']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all examples"""
    print("🚀 CPSC Regulation Knowledge Graph - Basic Usage Examples")
    print("=" * 70)
    print()
    
    examples = [
        example_data_loading,
        example_graph_statistics,
        example_basic_search,
        example_related_regulations,
        example_regulatory_clusters,
        example_relationship_analysis,
        example_compliance_requirements,
        example_network_analysis
    ]
    
    for example in examples:
        try:
            example()
            print()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print()
    
    print("✅ All examples completed!")

if __name__ == "__main__":
    main()