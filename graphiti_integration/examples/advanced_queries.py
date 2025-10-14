#!/usr/bin/env python3
"""
Advanced Graph Queries for CPSC Regulation Knowledge Graph
"""

import sys
from pathlib import Path
import json
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from integration_api import GraphitiIntegration
from graph_builder import RegulationGraphBuilder
from entities import KnowledgeGraph, RelationshipType

def find_regulatory_paths(integration: GraphitiIntegration, 
                         start_section: str, end_section: str, 
                         max_depth: int = 5) -> List[List[str]]:
    """Find all paths between two regulations"""
    try:
        # Get network for start section
        start_network = integration.get_regulation_network(start_section, depth=max_depth)
        if 'error' in start_network:
            return []
        
        # Build adjacency list
        adjacency = {}
        for edge in start_network['edges']:
            source = edge['source']
            target = edge['target']
            if source not in adjacency:
                adjacency[source] = []
            if target not in adjacency:
                adjacency[target] = []
            adjacency[source].append(target)
            adjacency[target].append(source)
        
        # Find all paths using BFS
        paths = []
        queue = [(start_section, [start_section])]
        visited = set()
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_section:
                paths.append(path)
                continue
            
            if len(path) >= max_depth:
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in adjacency.get(current, []):
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor]))
        
        return paths
        
    except Exception as e:
        print(f"Error finding paths: {e}")
        return []

def find_central_regulations(integration: GraphitiIntegration, 
                           top_k: int = 10) -> List[Dict[str, Any]]:
    """Find the most central regulations in the graph"""
    try:
        stats = integration.get_graph_statistics()
        
        # Get all sections with their degrees
        sections = []
        for node_id, node in integration.graph.nodes.items():
            if node.entity.entity_type == 'regulation_section':
                sections.append({
                    'section_number': node.entity.section_number,
                    'subject': node.entity.subject,
                    'degree': node.degree,
                    'centrality': node.centrality_score or 0
                })
        
        # Sort by degree and centrality
        sections.sort(key=lambda x: (x['degree'], x['centrality']), reverse=True)
        
        return sections[:top_k]
        
    except Exception as e:
        print(f"Error finding central regulations: {e}")
        return []

def find_regulatory_bridges(integration: GraphitiIntegration) -> List[Dict[str, Any]]:
    """Find regulations that act as bridges between different clusters"""
    try:
        # Get clusters
        clusters = integration.find_regulatory_clusters(min_cluster_size=2)
        
        if len(clusters) < 2:
            return []
        
        # Find sections that connect different clusters
        bridges = []
        
        for node_id, node in integration.graph.nodes.items():
            if node.entity.entity_type != 'regulation_section':
                continue
            
            # Get neighbors
            neighbors = node.neighbors
            if len(neighbors) < 2:
                continue
            
            # Check if neighbors belong to different clusters
            neighbor_clusters = set()
            for neighbor_id in neighbors:
                for cluster in clusters:
                    cluster_sections = [s['section_id'] for s in cluster['sections']]
                    if neighbor_id in cluster_sections:
                        neighbor_clusters.add(cluster['cluster_id'])
            
            if len(neighbor_clusters) > 1:
                bridges.append({
                    'section_number': node.entity.section_number,
                    'subject': node.entity.subject,
                    'degree': node.degree,
                    'connected_clusters': len(neighbor_clusters),
                    'neighbor_count': len(neighbors)
                })
        
        # Sort by number of connected clusters and degree
        bridges.sort(key=lambda x: (x['connected_clusters'], x['degree']), reverse=True)
        
        return bridges
        
    except Exception as e:
        print(f"Error finding regulatory bridges: {e}")
        return []

def analyze_regulatory_evolution(integration: GraphitiIntegration) -> Dict[str, Any]:
    """Analyze how regulations have evolved over time"""
    try:
        # This is a simplified analysis - in practice, you'd need temporal data
        analysis = {
            'total_sections': len([n for n in integration.graph.nodes.values() 
                                 if n.entity.entity_type == 'regulation_section']),
            'relationship_density': 0,
            'average_connectivity': 0,
            'regulatory_complexity': 0
        }
        
        # Calculate relationship density
        total_possible_edges = len(integration.graph.nodes) * (len(integration.graph.nodes) - 1) / 2
        if total_possible_edges > 0:
            analysis['relationship_density'] = len(integration.graph.edges) / total_possible_edges
        
        # Calculate average connectivity
        total_degree = sum(node.degree for node in integration.graph.nodes.values())
        if integration.graph.nodes:
            analysis['average_connectivity'] = total_degree / len(integration.graph.nodes)
        
        # Calculate regulatory complexity (based on text length and relationships)
        total_words = 0
        for node in integration.graph.nodes.values():
            if (node.entity.entity_type == 'regulation_section' and 
                hasattr(node.entity, 'word_count')):
                total_words += node.entity.word_count
        
        analysis['regulatory_complexity'] = total_words / max(analysis['total_sections'], 1)
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing regulatory evolution: {e}")
        return {}

def find_regulatory_gaps(integration: GraphitiIntegration) -> List[Dict[str, Any]]:
    """Find potential gaps in regulatory coverage"""
    try:
        gaps = []
        
        # Find isolated sections (low connectivity)
        isolated_sections = []
        for node_id, node in integration.graph.nodes.items():
            if (node.entity.entity_type == 'regulation_section' and 
                node.degree < 3):  # Threshold for isolation
                isolated_sections.append({
                    'section_number': node.entity.section_number,
                    'subject': node.entity.subject,
                    'degree': node.degree,
                    'text_preview': node.entity.text[:200] + "..." if len(node.entity.text) > 200 else node.entity.text
                })
        
        # Find sections with high similarity but no direct relationship
        # This would require more sophisticated analysis in practice
        potential_duplicates = []
        
        gaps.extend(isolated_sections)
        gaps.extend(potential_duplicates)
        
        return gaps
        
    except Exception as e:
        print(f"Error finding regulatory gaps: {e}")
        return []

def generate_regulatory_insights(integration: GraphitiIntegration) -> Dict[str, Any]:
    """Generate comprehensive regulatory insights"""
    try:
        insights = {
            'overview': {},
            'central_regulations': [],
            'regulatory_bridges': [],
            'regulatory_gaps': [],
            'evolution_analysis': {},
            'recommendations': []
        }
        
        # Overview
        stats = integration.get_graph_statistics()
        insights['overview'] = {
            'total_regulations': stats['total_nodes'],
            'total_relationships': stats['total_edges'],
            'average_connectivity': stats['average_degree'],
            'relationship_density': stats['total_edges'] / max(stats['total_nodes'] * (stats['total_nodes'] - 1) / 2, 1)
        }
        
        # Central regulations
        insights['central_regulations'] = find_central_regulations(integration, top_k=5)
        
        # Regulatory bridges
        insights['regulatory_bridges'] = find_regulatory_bridges(integration)
        
        # Regulatory gaps
        insights['regulatory_gaps'] = find_regulatory_gaps(integration)
        
        # Evolution analysis
        insights['evolution_analysis'] = analyze_regulatory_evolution(integration)
        
        # Generate recommendations
        recommendations = []
        
        if insights['regulatory_gaps']:
            recommendations.append({
                'type': 'coverage_gap',
                'description': f"Found {len(insights['regulatory_gaps'])} potentially isolated regulations that may need better integration",
                'priority': 'medium'
            })
        
        if insights['regulatory_bridges']:
            recommendations.append({
                'type': 'integration_opportunity',
                'description': f"Found {len(insights['regulatory_bridges'])} regulations that could serve as integration points",
                'priority': 'high'
            })
        
        if insights['overview']['relationship_density'] < 0.1:
            recommendations.append({
                'type': 'connectivity',
                'description': "Low relationship density suggests opportunities for better cross-referencing",
                'priority': 'high'
            })
        
        insights['recommendations'] = recommendations
        
        return insights
        
    except Exception as e:
        print(f"Error generating regulatory insights: {e}")
        return {}

def main():
    """Run advanced query examples"""
    print("🔬 Advanced Graph Queries for CPSC Regulation Knowledge Graph")
    print("=" * 70)
    print()
    
    try:
        # Initialize integration
        integration = GraphitiIntegration()
        print("✅ Integration initialized")
        
        # Example 1: Find regulatory paths
        print("\n1. Finding regulatory paths...")
        paths = find_regulatory_paths(integration, "§ 1000.1", "§ 1000.2", max_depth=3)
        print(f"Found {len(paths)} paths between § 1000.1 and § 1000.2")
        for i, path in enumerate(paths[:3]):  # Show first 3 paths
            print(f"  Path {i+1}: {' -> '.join(path)}")
        
        # Example 2: Find central regulations
        print("\n2. Finding central regulations...")
        central = find_central_regulations(integration, top_k=5)
        print(f"Top 5 most central regulations:")
        for i, reg in enumerate(central, 1):
            print(f"  {i}. {reg['section_number']}: {reg['subject']} (degree: {reg['degree']})")
        
        # Example 3: Find regulatory bridges
        print("\n3. Finding regulatory bridges...")
        bridges = find_regulatory_bridges(integration)
        print(f"Found {len(bridges)} regulatory bridges")
        for i, bridge in enumerate(bridges[:3], 1):
            print(f"  {i}. {bridge['section_number']}: {bridge['subject']} (connects {bridge['connected_clusters']} clusters)")
        
        # Example 4: Analyze regulatory evolution
        print("\n4. Analyzing regulatory evolution...")
        evolution = analyze_regulatory_evolution(integration)
        print(f"Evolution analysis:")
        for key, value in evolution.items():
            print(f"  {key}: {value}")
        
        # Example 5: Find regulatory gaps
        print("\n5. Finding regulatory gaps...")
        gaps = find_regulatory_gaps(integration)
        print(f"Found {len(gaps)} potential regulatory gaps")
        for i, gap in enumerate(gaps[:3], 1):
            print(f"  {i}. {gap['section_number']}: {gap['subject']} (degree: {gap['degree']})")
        
        # Example 6: Generate comprehensive insights
        print("\n6. Generating comprehensive insights...")
        insights = generate_regulatory_insights(integration)
        print(f"Generated insights with {len(insights['recommendations'])} recommendations")
        
        # Save insights to file
        with open('regulatory_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        print("Insights saved to regulatory_insights.json")
        
        print("\n✅ All advanced queries completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()