#!/usr/bin/env python3
"""
Simplified Knowledge Graph for CPSC Regulations
A working implementation without complex dependencies
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegulationNode:
    """Represents a node in the knowledge graph"""
    id: str
    name: str
    node_type: str  # 'section', 'part', 'chapter'
    content: str
    metadata: Dict[str, Any]
    neighbors: List[str] = None
    degree: int = 0
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []

@dataclass
class RegulationEdge:
    """Represents an edge in the knowledge graph"""
    id: str
    source: str
    target: str
    relationship_type: str
    weight: float
    confidence: float
    context: str = ""

class SimpleKnowledgeGraph:
    """Simplified knowledge graph for CPSC regulations"""
    
    def __init__(self, db_path: str = "/workspace/regulations.db"):
        self.db_path = db_path
        self.nodes: Dict[str, RegulationNode] = {}
        self.edges: Dict[str, RegulationEdge] = {}
        self.section_texts: List[str] = []
        self.section_ids: List[str] = []
        
    def build_graph(self) -> None:
        """Build the knowledge graph from the database"""
        logger.info("🚀 Building simplified knowledge graph...")
        
        # Load data from database
        self._load_data()
        
        # Build hierarchical relationships
        self._build_hierarchical_relationships()
        
        # Build semantic relationships
        self._build_semantic_relationships()
        
        # Build compliance relationships
        self._build_compliance_relationships()
        
        logger.info(f"✅ Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _load_data(self) -> None:
        """Load data from SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # Load chapters
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chapters")
            for row in cursor.fetchall():
                node = RegulationNode(
                    id=f"chapter_{row['chapter_id']}",
                    name=row['chapter_name'],
                    node_type='chapter',
                    content=row['chapter_name'],
                    metadata={'chapter_id': row['chapter_id']}
                )
                self.nodes[node.id] = node
            
            # Load parts
            cursor.execute("""
                SELECT p.*, sc.chapter_id 
                FROM parts p 
                JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
            """)
            for row in cursor.fetchall():
                node = RegulationNode(
                    id=f"part_{row['part_id']}",
                    name=row['heading'],
                    node_type='part',
                    content=row['heading'],
                    metadata={
                        'part_id': row['part_id'],
                        'subchapter_id': row['subchapter_id'],
                        'chapter_id': row['chapter_id']
                    }
                )
                self.nodes[node.id] = node
            
            # Load sections
            cursor.execute("""
                SELECT s.*, p.subchapter_id, sc.chapter_id
                FROM sections s
                JOIN parts p ON s.part_id = p.part_id
                JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
            """)
            for row in cursor.fetchall():
                node = RegulationNode(
                    id=f"section_{row['section_id']}",
                    name=row['section_number'],
                    node_type='section',
                    content=row['text'] or "",
                    metadata={
                        'section_id': row['section_id'],
                        'section_number': row['section_number'],
                        'subject': row['subject'],
                        'part_id': row['part_id'],
                        'subchapter_id': row['subchapter_id'],
                        'chapter_id': row['chapter_id'],
                        'word_count': len((row['text'] or "").split()),
                        'char_count': len(row['text'] or "")
                    }
                )
                self.nodes[node.id] = node
                self.section_texts.append(f"{row['subject']} {row['text'] or ''}")
                self.section_ids.append(node.id)
                
        finally:
            conn.close()
    
    def _build_hierarchical_relationships(self) -> None:
        """Build hierarchical relationships (chapter -> part -> section)"""
        logger.info("🔗 Building hierarchical relationships...")
        
        # Chapter -> Part relationships
        for node in self.nodes.values():
            if node.node_type == 'part':
                chapter_id = f"chapter_{node.metadata['chapter_id']}"
                if chapter_id in self.nodes:
                    self._add_edge(
                        source=chapter_id,
                        target=node.id,
                        relationship_type='contains',
                        weight=1.0,
                        confidence=1.0,
                        context="Hierarchical containment"
                    )
        
        # Part -> Section relationships
        for node in self.nodes.values():
            if node.node_type == 'section':
                part_id = f"part_{node.metadata['part_id']}"
                if part_id in self.nodes:
                    self._add_edge(
                        source=part_id,
                        target=node.id,
                        relationship_type='contains',
                        weight=1.0,
                        confidence=1.0,
                        context="Hierarchical containment"
                    )
    
    def _build_semantic_relationships(self) -> None:
        """Build semantic relationships using simple text similarity"""
        logger.info("🧠 Building semantic relationships...")
        
        # Simple keyword-based similarity
        keywords = [
            'safety', 'hazard', 'risk', 'injury', 'death', 'poison',
            'flammable', 'toxic', 'warning', 'label', 'package', 'test',
            'standard', 'compliance', 'violation', 'penalty', 'fine'
        ]
        
        # Build keyword vectors for each section
        section_keywords = {}
        for node in self.nodes.values():
            if node.node_type == 'section':
                text_lower = node.content.lower()
                keyword_counts = {kw: text_lower.count(kw) for kw in keywords}
                section_keywords[node.id] = keyword_counts
        
        # Find similar sections
        threshold = 0.3  # Minimum similarity threshold
        for i, (id1, keywords1) in enumerate(section_keywords.items()):
            for j, (id2, keywords2) in enumerate(list(section_keywords.items())[i+1:], i+1):
                similarity = self._calculate_similarity(keywords1, keywords2)
                
                if similarity >= threshold:
                    self._add_edge(
                        source=id1,
                        target=id2,
                        relationship_type='similar_to',
                        weight=similarity,
                        confidence=similarity,
                        context=f"Semantic similarity: {similarity:.3f}"
                    )
    
    def _calculate_similarity(self, keywords1: Dict[str, int], keywords2: Dict[str, int]) -> float:
        """Calculate simple cosine similarity between keyword vectors"""
        dot_product = sum(keywords1[k] * keywords2[k] for k in keywords1.keys())
        norm1 = sum(v**2 for v in keywords1.values())**0.5
        norm2 = sum(v**2 for v in keywords2.values())**0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _build_compliance_relationships(self) -> None:
        """Build compliance-related relationships"""
        logger.info("⚖️ Building compliance relationships...")
        
        compliance_keywords = [
            'shall', 'must', 'required', 'prohibited', 'mandatory',
            'compliance', 'violation', 'penalty', 'fine', 'enforcement'
        ]
        
        for node in self.nodes.values():
            if node.node_type == 'section':
                text_lower = node.content.lower()
                compliance_count = sum(text_lower.count(kw) for kw in compliance_keywords)
                
                if compliance_count > 0:
                    # Add metadata about compliance content
                    node.metadata['compliance_score'] = compliance_count
                    node.metadata['has_compliance'] = True
                    
                    # Find other sections with similar compliance content
                    for other_node in self.nodes.values():
                        if (other_node.node_type == 'section' and 
                            other_node.id != node.id and 
                            other_node.metadata.get('has_compliance', False)):
                            
                            other_compliance = other_node.metadata.get('compliance_score', 0)
                            if other_compliance > 0:
                                # Create compliance relationship
                                similarity = min(compliance_count, other_compliance) / max(compliance_count, other_compliance)
                                if similarity > 0.5:
                                    self._add_edge(
                                        source=node.id,
                                        target=other_node.id,
                                        relationship_type='compliance_related',
                                        weight=similarity,
                                        confidence=similarity,
                                        context="Similar compliance requirements"
                                    )
    
    def _add_edge(self, source: str, target: str, relationship_type: str, 
                  weight: float, confidence: float, context: str = "") -> None:
        """Add an edge to the graph"""
        edge_id = f"{source}_{relationship_type}_{target}"
        
        edge = RegulationEdge(
            id=edge_id,
            source=source,
            target=target,
            relationship_type=relationship_type,
            weight=weight,
            confidence=confidence,
            context=context
        )
        
        self.edges[edge_id] = edge
        
        # Update node degrees and neighbors
        if source in self.nodes:
            self.nodes[source].neighbors.append(target)
            self.nodes[source].degree += 1
        if target in self.nodes:
            self.nodes[target].neighbors.append(source)
            self.nodes[target].degree += 1
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search the knowledge graph"""
        results = []
        query_lower = query.lower()
        
        for node in self.nodes.values():
            if node.node_type == 'section':
                content_lower = node.content.lower()
                subject_lower = node.metadata.get('subject', '').lower()
                
                # Simple text matching
                score = 0
                if query_lower in content_lower:
                    score += content_lower.count(query_lower) * 0.1
                if query_lower in subject_lower:
                    score += 2.0
                
                if score > 0:
                    results.append({
                        'node': node,
                        'score': score,
                        'match_type': 'text_match'
                    })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def get_related_nodes(self, node_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get nodes related to a specific node"""
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        related = []
        
        for neighbor_id in node.neighbors:
            if neighbor_id in self.nodes:
                neighbor = self.nodes[neighbor_id]
                
                # Find connecting edge
                edge = None
                for e in self.edges.values():
                    if ((e.source == node_id and e.target == neighbor_id) or
                        (e.source == neighbor_id and e.target == node_id)):
                        edge = e
                        break
                
                related.append({
                    'node': neighbor,
                    'edge': edge,
                    'relationship_type': edge.relationship_type if edge else 'unknown',
                    'weight': edge.weight if edge else 0.0
                })
        
        # Sort by edge weight
        related.sort(key=lambda x: x['weight'], reverse=True)
        return related[:limit]
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': defaultdict(int),
            'relationship_types': defaultdict(int),
            'degree_stats': {},
            'compliance_stats': {}
        }
        
        # Node type distribution
        for node in self.nodes.values():
            stats['node_types'][node.node_type] += 1
        
        # Relationship type distribution
        for edge in self.edges.values():
            stats['relationship_types'][edge.relationship_type] += 1
        
        # Degree statistics
        degrees = [node.degree for node in self.nodes.values()]
        if degrees:
            stats['degree_stats'] = {
                'min': min(degrees),
                'max': max(degrees),
                'avg': sum(degrees) / len(degrees)
            }
        
        # Compliance statistics
        compliance_sections = [n for n in self.nodes.values() 
                             if n.node_type == 'section' and n.metadata.get('has_compliance', False)]
        stats['compliance_stats'] = {
            'total_compliance_sections': len(compliance_sections),
            'avg_compliance_score': sum(n.metadata.get('compliance_score', 0) for n in compliance_sections) / max(len(compliance_sections), 1)
        }
        
        return dict(stats)
    
    def export_to_json(self) -> str:
        """Export graph to JSON format"""
        export_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges)
            },
            'nodes': [
                {
                    'id': node.id,
                    'name': node.name,
                    'type': node.node_type,
                    'content_preview': node.content[:200] + "..." if len(node.content) > 200 else node.content,
                    'metadata': node.metadata,
                    'degree': node.degree
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'id': edge.id,
                    'source': edge.source,
                    'target': edge.target,
                    'relationship_type': edge.relationship_type,
                    'weight': edge.weight,
                    'confidence': edge.confidence,
                    'context': edge.context
                }
                for edge in self.edges.values()
            ]
        }
        
        return json.dumps(export_data, indent=2)
    
    def find_clusters(self, min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """Find clusters of related regulations using simple community detection"""
        # Simple clustering based on shared neighbors
        visited = set()
        clusters = []
        
        for node_id, node in self.nodes.items():
            if node_id in visited or node.node_type != 'section':
                continue
            
            # Start a new cluster
            cluster = [node_id]
            visited.add(node_id)
            
            # Find nodes with shared neighbors
            for other_id, other_node in self.nodes.items():
                if (other_id in visited or 
                    other_node.node_type != 'section' or 
                    other_id == node_id):
                    continue
                
                # Check for shared neighbors
                shared_neighbors = set(node.neighbors) & set(other_node.neighbors)
                if len(shared_neighbors) >= 2:  # At least 2 shared neighbors
                    cluster.append(other_id)
                    visited.add(other_id)
            
            if len(cluster) >= min_cluster_size:
                clusters.append({
                    'cluster_id': len(clusters),
                    'size': len(cluster),
                    'nodes': cluster,
                    'representative': node.name
                })
        
        return clusters

def main():
    """Test the simplified knowledge graph"""
    print("🚀 Simplified CPSC Regulation Knowledge Graph")
    print("=" * 50)
    
    # Build graph
    graph = SimpleKnowledgeGraph()
    graph.build_graph()
    
    # Get statistics
    stats = graph.get_graph_statistics()
    print(f"\n📊 Graph Statistics:")
    print(f"   Nodes: {stats['total_nodes']}")
    print(f"   Edges: {stats['total_edges']}")
    print(f"   Node types: {dict(stats['node_types'])}")
    print(f"   Relationship types: {dict(stats['relationship_types'])}")
    
    if stats['degree_stats']:
        print(f"   Degree stats: min={stats['degree_stats']['min']}, max={stats['degree_stats']['max']}, avg={stats['degree_stats']['avg']:.1f}")
    
    if stats['compliance_stats']['total_compliance_sections'] > 0:
        print(f"   Compliance sections: {stats['compliance_stats']['total_compliance_sections']}")
    
    # Test search
    print(f"\n🔍 Search Test:")
    search_results = graph.search("safety", limit=5)
    print(f"   Found {len(search_results)} results for 'safety'")
    for i, result in enumerate(search_results[:3], 1):
        node = result['node']
        print(f"   {i}. {node.name}: {node.metadata.get('subject', 'No subject')}")
    
    # Test related nodes
    if graph.nodes:
        first_section = next((n for n in graph.nodes.values() if n.node_type == 'section'), None)
        if first_section:
            print(f"\n🔗 Related Nodes Test:")
            related = graph.get_related_nodes(first_section.id, limit=3)
            print(f"   Found {len(related)} related nodes for {first_section.name}")
            for i, rel in enumerate(related[:3], 1):
                print(f"   {i}. {rel['node'].name} ({rel['relationship_type']}, weight={rel['weight']:.3f})")
    
    # Test clustering
    print(f"\n🎯 Clustering Test:")
    clusters = graph.find_clusters(min_cluster_size=2)
    print(f"   Found {len(clusters)} clusters")
    for i, cluster in enumerate(clusters[:3], 1):
        print(f"   Cluster {i}: {cluster['size']} nodes, representative: {cluster['representative']}")
    
    # Export test
    print(f"\n💾 Export Test:")
    json_export = graph.export_to_json()
    print(f"   Exported {len(json_export)} characters to JSON")
    
    print(f"\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()