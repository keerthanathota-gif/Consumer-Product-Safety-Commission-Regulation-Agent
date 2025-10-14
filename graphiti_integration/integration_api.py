#!/usr/bin/env python3
"""
Integration API for CPSC Regulation Knowledge Graph
Provides clean interface for integrating graph capabilities with existing system
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from graph_builder import RegulationGraphBuilder
from data_loader import DataLoader
from entities import KnowledgeGraph, GraphNode, GraphEdge, RelationshipType
from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphitiIntegration:
    """Main integration class for CPSC Regulation Knowledge Graph"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        self.builder = None
        self.graph = None
        self.data_loader = DataLoader()
        self._initialize()
    
    def _initialize(self):
        """Initialize the integration"""
        try:
            self.builder = RegulationGraphBuilder(self.config)
            self.graph = self.builder.build_graph()
            logger.info("✅ GraphitiIntegration initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GraphitiIntegration: {e}")
            raise
    
    def get_related_regulations(self, section_number: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get regulations related to a specific section"""
        try:
            # Find the section in the graph
            section_id = None
            for node_id, node in self.graph.nodes.items():
                if (hasattr(node.entity, 'section_number') and 
                    node.entity.section_number == section_number):
                    section_id = node_id
                    break
            
            if not section_id:
                logger.warning(f"Section {section_number} not found in graph")
                return []
            
            # Get related sections
            related = self.builder.get_related_sections(section_id, limit)
            
            # Format results
            results = []
            for item in related:
                section = item['section']
                relationship = item['relationship']
                
                result = {
                    'section_number': section.section_number,
                    'subject': section.subject,
                    'text_preview': section.text[:200] + "..." if len(section.text) > 200 else section.text,
                    'relationship_type': relationship['type'] if relationship else 'unknown',
                    'relationship_weight': relationship['weight'] if relationship else 0.0,
                    'relationship_confidence': relationship['confidence'] if relationship else 0.0,
                    'context': relationship['context'] if relationship and relationship['context'] else None
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get related regulations for {section_number}: {e}")
            return []
    
    def find_regulatory_clusters(self, min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Find clusters of related regulations"""
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Prepare data for clustering
            sections = []
            section_ids = []
            
            for node_id, node in self.graph.nodes.items():
                if node.entity.entity_type == 'regulation_section':
                    sections.append(f"{node.entity.subject} {node.entity.text}")
                    section_ids.append(node_id)
            
            if len(sections) < min_cluster_size:
                return []
            
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sections)
            
            # Perform clustering
            clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size)
            cluster_labels = clustering.fit_predict(tfidf_matrix)
            
            # Group sections by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append({
                        'section_id': section_ids[i],
                        'section_number': self.graph.nodes[section_ids[i]].entity.section_number,
                        'subject': self.graph.nodes[section_ids[i]].entity.subject
                    })
            
            # Format results
            results = []
            for cluster_id, cluster_sections in clusters.items():
                if len(cluster_sections) >= min_cluster_size:
                    results.append({
                        'cluster_id': cluster_id,
                        'size': len(cluster_sections),
                        'sections': cluster_sections
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find regulatory clusters: {e}")
            return []
    
    def analyze_relationships(self, section1: str, section2: str) -> Dict[str, Any]:
        """Analyze relationships between two specific sections"""
        try:
            # Find both sections
            section1_id = None
            section2_id = None
            
            for node_id, node in self.graph.nodes.items():
                if (hasattr(node.entity, 'section_number')):
                    if node.entity.section_number == section1:
                        section1_id = node_id
                    elif node.entity.section_number == section2:
                        section2_id = node_id
            
            if not section1_id or not section2_id:
                return {'error': 'One or both sections not found'}
            
            # Find direct relationships
            direct_relationships = []
            for edge in self.graph.edges.values():
                if ((edge.source_node == section1_id and edge.target_node == section2_id) or
                    (edge.source_node == section2_id and edge.target_node == section1_id)):
                    direct_relationships.append({
                        'type': edge.relationship.relationship_type.value,
                        'weight': edge.weight,
                        'confidence': edge.relationship.confidence,
                        'context': edge.relationship.context
                    })
            
            # Find indirect relationships (through common neighbors)
            section1_neighbors = set(self.graph.nodes[section1_id].neighbors)
            section2_neighbors = set(self.graph.nodes[section2_id].neighbors)
            common_neighbors = section1_neighbors.intersection(section2_neighbors)
            
            # Calculate similarity metrics
            similarity_score = len(common_neighbors) / max(len(section1_neighbors), len(section2_neighbors), 1)
            
            return {
                'section1': section1,
                'section2': section2,
                'direct_relationships': direct_relationships,
                'common_neighbors': len(common_neighbors),
                'similarity_score': similarity_score,
                'section1_degree': self.graph.nodes[section1_id].degree,
                'section2_degree': self.graph.nodes[section2_id].degree
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze relationships between {section1} and {section2}: {e}")
            return {'error': str(e)}
    
    def search_regulations(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search regulations using semantic similarity"""
        try:
            # Use Graphiti search
            results = self.builder.search_graph(query, limit)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'content': result.get('content', ''),
                    'score': result.get('score', 0.0),
                    'metadata': result.get('metadata', {}),
                    'type': result.get('type', 'unknown')
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search regulations: {e}")
            return []
    
    def get_regulation_network(self, section_number: str, depth: int = 2) -> Dict[str, Any]:
        """Get the network around a specific regulation"""
        try:
            # Find the section
            section_id = None
            for node_id, node in self.graph.nodes.items():
                if (hasattr(node.entity, 'section_number') and 
                    node.entity.section_number == section_number):
                    section_id = node_id
                    break
            
            if not section_id:
                return {'error': 'Section not found'}
            
            # Build network at specified depth
            visited = set()
            current_level = {section_id}
            network = {
                'nodes': [],
                'edges': [],
                'depths': {}
            }
            
            for d in range(depth + 1):
                next_level = set()
                
                for node_id in current_level:
                    if node_id in visited:
                        continue
                    
                    visited.add(node_id)
                    node = self.graph.nodes[node_id]
                    
                    network['nodes'].append({
                        'id': node_id,
                        'name': node.entity.name,
                        'type': node.entity.entity_type,
                        'degree': node.degree,
                        'depth': d
                    })
                    
                    network['depths'][node_id] = d
                    
                    # Add edges to neighbors
                    for neighbor_id in node.neighbors:
                        if neighbor_id not in visited:
                            next_level.add(neighbor_id)
                            
                            # Add edge
                            edge = None
                            for e in self.graph.edges.values():
                                if ((e.source_node == node_id and e.target_node == neighbor_id) or
                                    (e.source_node == neighbor_id and e.target_node == node_id)):
                                    edge = e
                                    break
                            
                            if edge:
                                network['edges'].append({
                                    'source': node_id,
                                    'target': neighbor_id,
                                    'type': edge.relationship.relationship_type.value,
                                    'weight': edge.weight,
                                    'confidence': edge.relationship.confidence
                                })
                
                current_level = next_level
            
            return network
            
        except Exception as e:
            logger.error(f"Failed to get regulation network for {section_number}: {e}")
            return {'error': str(e)}
    
    def get_compliance_requirements(self, section_number: str) -> List[Dict[str, Any]]:
        """Extract compliance requirements from a regulation section"""
        try:
            # Find the section
            section_id = None
            for node_id, node in self.graph.nodes.items():
                if (hasattr(node.entity, 'section_number') and 
                    node.entity.section_number == section_number):
                    section_id = node_id
                    break
            
            if not section_id:
                return []
            
            section = self.graph.nodes[section_id].entity
            
            # Extract compliance requirements
            requirements = []
            if hasattr(section, 'compliance_requirements'):
                for i, req in enumerate(section.compliance_requirements):
                    requirements.append({
                        'id': f"{section_number}_req_{i}",
                        'text': req,
                        'section': section_number,
                        'type': 'compliance_requirement'
                    })
            
            # Extract safety standards
            if hasattr(section, 'safety_standards'):
                for i, std in enumerate(section.safety_standards):
                    requirements.append({
                        'id': f"{section_number}_std_{i}",
                        'text': std,
                        'section': section_number,
                        'type': 'safety_standard'
                    })
            
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to get compliance requirements for {section_number}: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            stats = {
                'total_nodes': len(self.graph.nodes),
                'total_edges': len(self.graph.edges),
                'node_types': {},
                'relationship_types': {},
                'average_degree': 0,
                'max_degree': 0,
                'min_degree': 0,
                'average_weight': 0,
                'max_weight': 0,
                'min_weight': 0
            }
            
            # Node type distribution
            for node in self.graph.nodes.values():
                entity_type = node.entity.entity_type
                if entity_type not in stats['node_types']:
                    stats['node_types'][entity_type] = 0
                stats['node_types'][entity_type] += 1
            
            # Relationship type distribution
            for edge in self.graph.edges.values():
                rel_type = edge.relationship.relationship_type.value
                if rel_type not in stats['relationship_types']:
                    stats['relationship_types'][rel_type] = 0
                stats['relationship_types'][rel_type] += 1
            
            # Degree statistics
            if self.graph.nodes:
                degrees = [node.degree for node in self.graph.nodes.values()]
                stats['average_degree'] = sum(degrees) / len(degrees)
                stats['max_degree'] = max(degrees)
                stats['min_degree'] = min(degrees)
            
            # Weight statistics
            if self.graph.edges:
                weights = [edge.weight for edge in self.graph.edges.values()]
                stats['average_weight'] = sum(weights) / len(weights)
                stats['max_weight'] = max(weights)
                stats['min_weight'] = min(weights)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {'error': str(e)}
    
    def export_graph_data(self, format: str = 'json') -> str:
        """Export graph data in specified format"""
        try:
            if format == 'json':
                return self.graph.model_dump_json(indent=2)
            elif format == 'graphml':
                return self.builder.export_graph('graphml')
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Failed to export graph data: {e}")
            return ""

def main():
    """Test the integration API"""
    try:
        # Initialize integration
        integration = GraphitiIntegration()
        
        # Test basic functionality
        print("✅ GraphitiIntegration initialized")
        
        # Test graph statistics
        stats = integration.get_graph_statistics()
        print(f"✅ Graph statistics: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
        
        # Test search
        search_results = integration.search_regulations("safety requirements", limit=5)
        print(f"✅ Search test: Found {len(search_results)} results")
        
        # Test related regulations
        related = integration.get_related_regulations("§ 1000.1", limit=3)
        print(f"✅ Related regulations test: Found {len(related)} related sections")
        
        # Test clusters
        clusters = integration.find_regulatory_clusters(min_cluster_size=2)
        print(f"✅ Clusters test: Found {len(clusters)} clusters")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Integration API test failed: {e}")

if __name__ == "__main__":
    main()