#!/usr/bin/env python3
"""
Graph Builder for CPSC Regulation Knowledge Graph using Graphiti
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Graphiti imports
from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver

# Local imports
from entities import (
    RegulationSection, RegulationPart, RegulationChapter,
    Relationship, RelationshipType, GraphNode, GraphEdge, KnowledgeGraph
)
from data_loader import DataLoader
from config import get_config, validate_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegulationGraphBuilder:
    """Builds knowledge graph from CPSC regulation data using Graphiti"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        self.data_loader = DataLoader()
        self.graphiti = None
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize Graphiti
        self._initialize_graphiti()
    
    def _initialize_graphiti(self):
        """Initialize Graphiti with configuration"""
        try:
            # Set up OpenAI client
            from graphiti_core.llm_client.config import LLMConfig
            llm_config = LLMConfig(
                api_key=self.config["openai"]["api_key"],
                model=self.config["openai"]["model"]
            )
            openai_client = OpenAIClient(config=llm_config)
            
            # Set up embedder
            from graphiti_core.embedder.openai import OpenAIEmbedderConfig
            embedder_config = OpenAIEmbedderConfig(
                api_key=self.config["openai"]["api_key"],
                embedding_model=self.config["graphiti"]["embedder_model"]
            )
            embedder = OpenAIEmbedder(config=embedder_config)
            
            # Set up Neo4j driver
            neo4j_driver = Neo4jDriver(
                uri=self.config["database"]["neo4j_uri"],
                user=self.config["database"]["neo4j_user"],
                password=self.config["database"]["neo4j_password"],
                database=self.config["database"]["neo4j_database"]
            )
            
            # Initialize Graphiti
            self.graphiti = Graphiti(
                group_id=self.config["graphiti"]["group_id"],
                llm_client=openai_client,
                embedder=embedder,
                driver=neo4j_driver
            )
            
            logger.info("✅ Graphiti initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Graphiti: {e}")
            raise
    
    def build_graph(self) -> KnowledgeGraph:
        """Build the complete knowledge graph"""
        logger.info("🚀 Starting knowledge graph construction...")
        
        # Load data
        data = self.data_loader.load_from_sqlite()
        logger.info(f"📊 Loaded {len(data.sections)} sections, {len(data.parts)} parts, {len(data.chapters)} chapters")
        
        # Build hierarchical relationships
        self._build_hierarchical_relationships(data)
        
        # Build semantic relationships
        self._build_semantic_relationships(data)
        
        # Build compliance relationships
        self._build_compliance_relationships(data)
        
        # Add to Graphiti
        self._add_to_graphiti(data)
        
        logger.info(f"✅ Knowledge graph built with {len(self.knowledge_graph.nodes)} nodes and {len(self.knowledge_graph.edges)} edges")
        
        return self.knowledge_graph
    
    def _build_hierarchical_relationships(self, data):
        """Build hierarchical relationships (chapter -> part -> section)"""
        logger.info("🔗 Building hierarchical relationships...")
        
        # Create chapter nodes
        for chapter in data.chapters:
            node = GraphNode(
                entity=chapter,
                neighbors=[],
                degree=0
            )
            self.knowledge_graph.add_node(node)
        
        # Create part nodes and chapter-part relationships
        for part in data.parts:
            node = GraphNode(
                entity=part,
                neighbors=[],
                degree=0
            )
            self.knowledge_graph.add_node(node)
            
            # Add chapter-part relationship
            chapter_id = f"chapter_{part.chapter_id}"
            if chapter_id in self.knowledge_graph.nodes:
                self._add_relationship(
                    source_id=chapter_id,
                    target_id=part.id,
                    relationship_type=RelationshipType.CONTAINS,
                    weight=1.0,
                    context="Hierarchical containment"
                )
        
        # Create section nodes and part-section relationships
        for section in data.sections:
            node = GraphNode(
                entity=section,
                neighbors=[],
                degree=0
            )
            self.knowledge_graph.add_node(node)
            
            # Add part-section relationship
            part_id = f"part_{section.part_id}"
            if part_id in self.knowledge_graph.nodes:
                self._add_relationship(
                    source_id=part_id,
                    target_id=section.id,
                    relationship_type=RelationshipType.CONTAINS,
                    weight=1.0,
                    context="Hierarchical containment"
                )
    
    def _build_semantic_relationships(self, data):
        """Build semantic relationships between sections"""
        logger.info("🧠 Building semantic relationships...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Prepare text data for similarity analysis
        sections = data.sections
        texts = [f"{s.subject} {s.text}" for s in sections]
        
        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar sections
        threshold = self.config["analysis"]["min_similarity_score"]
        for i, section1 in enumerate(sections):
            for j, section2 in enumerate(sections[i+1:], i+1):
                similarity = similarity_matrix[i][j]
                
                if similarity >= threshold:
                    # Determine relationship type based on similarity
                    if similarity >= 0.9:
                        rel_type = RelationshipType.SIMILAR_TO
                        weight = similarity
                    elif similarity >= 0.7:
                        rel_type = RelationshipType.RELATED_TO
                        weight = similarity * 0.8
                    else:
                        continue
                    
                    self._add_relationship(
                        source_id=section1.id,
                        target_id=section2.id,
                        relationship_type=rel_type,
                        weight=weight,
                        context=f"Semantic similarity: {similarity:.3f}",
                        confidence=similarity
                    )
    
    def _build_compliance_relationships(self, data):
        """Build compliance-related relationships"""
        logger.info("⚖️ Building compliance relationships...")
        
        # Extract compliance requirements and safety standards
        for section in data.sections:
            # Extract compliance requirements (simple keyword-based)
            compliance_keywords = [
                "shall", "must", "required", "prohibited", "mandatory",
                "compliance", "violation", "penalty", "fine"
            ]
            
            text_lower = section.text.lower()
            compliance_requirements = []
            
            for keyword in compliance_keywords:
                if keyword in text_lower:
                    # Find sentences containing the keyword
                    sentences = section.text.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            compliance_requirements.append(sentence.strip())
            
            if compliance_requirements:
                section.compliance_requirements = compliance_requirements
            
            # Extract safety standards (look for standard references)
            import re
            standard_patterns = [
                r'ASTM\s+[A-Z0-9.-]+',
                r'ISO\s+\d+',
                r'ANSI\s+[A-Z0-9.-]+',
                r'UL\s+\d+',
                r'NFPA\s+\d+'
            ]
            
            safety_standards = []
            for pattern in standard_patterns:
                matches = re.findall(pattern, section.text, re.IGNORECASE)
                safety_standards.extend(matches)
            
            if safety_standards:
                section.safety_standards = list(set(safety_standards))
    
    def _add_relationship(self, source_id: str, target_id: str, 
                         relationship_type: RelationshipType, weight: float = 1.0,
                         context: str = None, confidence: float = 1.0):
        """Add a relationship to the knowledge graph"""
        rel_id = f"{source_id}_{relationship_type.value}_{target_id}"
        
        relationship = Relationship(
            id=rel_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            confidence=confidence,
            context=context
        )
        
        edge = GraphEdge(
            relationship=relationship,
            source_node=source_id,
            target_node=target_id,
            weight=weight,
            directed=True
        )
        
        self.knowledge_graph.add_edge(edge)
    
    def _add_to_graphiti(self, data):
        """Add entities and relationships to Graphiti"""
        logger.info("📤 Adding data to Graphiti...")
        
        try:
            # Create episodes for each section
            for section in data.sections:
                # Create entity node
                entity_node = EntityNode(
                    name=section.section_number,
                    entity_type="regulation_section",
                    attributes={
                        "subject": section.subject,
                        "text": section.text,
                        "citation": section.citation,
                        "word_count": section.word_count,
                        "sentence_count": section.sentence_count,
                        "compliance_requirements": section.compliance_requirements,
                        "safety_standards": section.safety_standards
                    }
                )
                
                # Create episodic node
                episodic_node = EpisodicNode(
                    name=f"Episode for {section.section_number}",
                    episode_type=EpisodeType.REGULATION,
                    content=section.text,
                    attributes={
                        "section_number": section.section_number,
                        "subject": section.subject,
                        "part_id": section.part_id,
                        "chapter_id": section.chapter_id
                    }
                )
                
                # Add to Graphiti
                self.graphiti.add_episode(
                    content=section.text,
                    entity_nodes=[entity_node],
                    episodic_nodes=[episodic_node]
                )
            
            logger.info("✅ Data added to Graphiti successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to add data to Graphiti: {e}")
            raise
    
    def search_graph(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search the knowledge graph"""
        if not self.graphiti:
            raise RuntimeError("Graphiti not initialized")
        
        try:
            results = self.graphiti.search(
                query=query,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_related_sections(self, section_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sections related to a specific section"""
        if section_id not in self.knowledge_graph.nodes:
            return []
        
        node = self.knowledge_graph.get_node(section_id)
        if not node:
            return []
        
        related = []
        for neighbor_id in node.neighbors:
            neighbor = self.knowledge_graph.get_node(neighbor_id)
            if neighbor:
                # Get relationship details
                edges = self.knowledge_graph.get_edges(section_id)
                relationship_info = None
                for edge in edges:
                    if edge.target_node == neighbor_id or edge.source_node == neighbor_id:
                        relationship_info = {
                            "type": edge.relationship.relationship_type.value,
                            "weight": edge.weight,
                            "confidence": edge.relationship.confidence,
                            "context": edge.relationship.context
                        }
                        break
                
                related.append({
                    "section": neighbor.entity,
                    "relationship": relationship_info
                })
        
        # Sort by relationship weight
        related.sort(key=lambda x: x["relationship"]["weight"] if x["relationship"] else 0, reverse=True)
        
        return related[:limit]
    
    def export_graph(self, format: str = "json") -> str:
        """Export the knowledge graph"""
        if format == "json":
            return self.knowledge_graph.model_dump_json(indent=2)
        elif format == "graphml":
            # Convert to GraphML format
            return self._convert_to_graphml()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _convert_to_graphml(self) -> str:
        """Convert knowledge graph to GraphML format"""
        # This is a simplified GraphML conversion
        # In a real implementation, you'd use a proper GraphML library
        graphml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        graphml += '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n'
        graphml += '  <graph id="cpsc_regulations" edgedefault="directed">\n'
        
        # Add nodes
        for node_id, node in self.knowledge_graph.nodes.items():
            graphml += f'    <node id="{node_id}">\n'
            graphml += f'      <data key="name">{node.entity.name}</data>\n'
            graphml += f'      <data key="type">{node.entity.entity_type}</data>\n'
            graphml += f'    </node>\n'
        
        # Add edges
        for edge_id, edge in self.knowledge_graph.edges.items():
            graphml += f'    <edge id="{edge_id}" source="{edge.source_node}" target="{edge.target_node}">\n'
            graphml += f'      <data key="relationship_type">{edge.relationship.relationship_type.value}</data>\n'
            graphml += f'      <data key="weight">{edge.weight}</data>\n'
            graphml += f'    </edge>\n'
        
        graphml += '  </graph>\n'
        graphml += '</graphml>\n'
        
        return graphml

def main():
    """Test the graph builder"""
    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed")
        return
    
    try:
        # Initialize builder
        builder = RegulationGraphBuilder()
        
        # Build graph
        graph = builder.build_graph()
        
        print(f"✅ Knowledge graph built successfully!")
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Edges: {len(graph.edges)}")
        
        # Test search
        search_results = builder.search_graph("safety requirements", limit=5)
        print(f"✅ Search test: Found {len(search_results)} results")
        
        # Test related sections
        if graph.nodes:
            first_section = list(graph.nodes.keys())[0]
            related = builder.get_related_sections(first_section, limit=3)
            print(f"✅ Related sections test: Found {len(related)} related sections")
        
        # Export graph
        json_export = builder.export_graph("json")
        print(f"✅ Graph exported to JSON ({len(json_export)} characters)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Graph building failed: {e}")

if __name__ == "__main__":
    main()