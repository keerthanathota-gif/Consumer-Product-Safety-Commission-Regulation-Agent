#!/usr/bin/env python3
"""
Graphiti Integration for Regulation Section Relationships
========================================================

This module integrates Graphiti to visualize relationships between regulation sections.
It creates a graph where nodes represent regulation sections and edges represent
various types of relationships (semantic similarity, cross-references, etc.).
"""

import sqlite3
import json
import re
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Try to import Graphiti
try:
    import graphiti
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    print("Warning: Graphiti not available. Using NetworkX as fallback.")


@dataclass
class SectionNode:
    """Represents a regulation section as a graph node."""
    section_id: str
    section_number: str
    subject: str
    text: str
    citation: str
    part_heading: str
    subchapter_name: str
    chapter_name: str


@dataclass
class SectionRelationship:
    """Represents a relationship between two regulation sections."""
    source_id: str
    target_id: str
    relationship_type: str
    strength: float
    description: str


class RegulationGraphBuilder:
    """Builds a graph representation of regulation sections and their relationships."""
    
    def __init__(self, db_path: str = "regulations.db"):
        self.db_path = db_path
        self.sections = {}
        self.relationships = []
        self.graph = nx.Graph()
        
    def load_sections_from_db(self) -> Dict[str, SectionNode]:
        """Load all regulation sections from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all sections with their hierarchy
        cursor.execute("""
            SELECT 
                s.section_id,
                s.section_number,
                s.subject,
                s.text,
                s.citation,
                p.heading as part_heading,
                sc.subchapter_name,
                c.chapter_name
            FROM sections s
            JOIN parts p ON s.part_id = p.part_id
            JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
            JOIN chapters c ON sc.chapter_id = c.chapter_id
            WHERE s.section_number IS NOT NULL AND s.section_number != ''
        """)
        
        sections = {}
        for row in cursor.fetchall():
            section_id, section_number, subject, text, citation, part_heading, subchapter_name, chapter_name = row
            
            # Clean and prepare text
            clean_text = self._clean_text(text)
            if len(clean_text) < 10:  # Skip very short sections
                continue
                
            section = SectionNode(
                section_id=str(section_id),
                section_number=section_number,
                subject=subject,
                text=clean_text,
                citation=citation,
                part_heading=part_heading,
                subchapter_name=subchapter_name,
                chapter_name=chapter_name
            )
            sections[section_id] = section
            
        conn.close()
        print(f"Loaded {len(sections)} regulation sections")
        return sections
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common regulation artifacts
        text = re.sub(r'§\s*\d+\.\d+', '', text)  # Remove section references
        text = re.sub(r'\([a-z]\)', '', text)     # Remove lettered subsections
        
        return text
    
    def detect_semantic_relationships(self, similarity_threshold: float = 0.3) -> List[SectionRelationship]:
        """Detect semantic relationships between sections using TF-IDF and cosine similarity."""
        if not self.sections:
            return []
        
        print("Detecting semantic relationships...")
        
        # Prepare texts for analysis
        section_texts = []
        section_ids = []
        
        for section_id, section in self.sections.items():
            # Combine subject and text for better analysis
            combined_text = f"{section.subject} {section.text}"
            section_texts.append(combined_text)
            section_ids.append(section_id)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(section_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            relationships = []
            
            # Find similar sections
            for i, source_id in enumerate(section_ids):
                for j, target_id in enumerate(section_ids):
                    if i >= j:  # Avoid duplicates and self-connections
                        continue
                    
                    similarity = similarity_matrix[i][j]
                    
                    if similarity >= similarity_threshold:
                        relationship = SectionRelationship(
                            source_id=source_id,
                            target_id=target_id,
                            relationship_type="semantic_similarity",
                            strength=similarity,
                            description=f"Semantic similarity: {similarity:.3f}"
                        )
                        relationships.append(relationship)
            
            print(f"Found {len(relationships)} semantic relationships")
            return relationships
            
        except Exception as e:
            print(f"Error in semantic analysis: {e}")
            return []
    
    def detect_cross_references(self) -> List[SectionRelationship]:
        """Detect cross-references between sections."""
        print("Detecting cross-references...")
        
        relationships = []
        section_numbers = {section.section_number: section_id for section_id, section in self.sections.items()}
        
        # Pattern to match section references
        section_ref_pattern = r'§\s*(\d+\.\d+)'
        
        for section_id, section in self.sections.items():
            # Find all section references in the text
            references = re.findall(section_ref_pattern, section.text)
            
            for ref in references:
                if ref in section_numbers and ref != section.section_number:
                    target_id = section_numbers[ref]
                    
                    relationship = SectionRelationship(
                        source_id=section_id,
                        target_id=target_id,
                        relationship_type="cross_reference",
                        strength=1.0,
                        description=f"References section {ref}"
                    )
                    relationships.append(relationship)
        
        print(f"Found {len(relationships)} cross-references")
        return relationships
    
    def detect_hierarchical_relationships(self) -> List[SectionRelationship]:
        """Detect hierarchical relationships (same part, subchapter, etc.)."""
        print("Detecting hierarchical relationships...")
        
        relationships = []
        
        # Group sections by part
        parts = {}
        for section_id, section in self.sections.items():
            part_key = f"{section.chapter_name}|{section.subchapter_name}|{section.part_heading}"
            if part_key not in parts:
                parts[part_key] = []
            parts[part_key].append(section_id)
        
        # Create relationships within each part
        for part_sections in parts.values():
            if len(part_sections) > 1:
                for i, source_id in enumerate(part_sections):
                    for target_id in part_sections[i+1:]:
                        relationship = SectionRelationship(
                            source_id=source_id,
                            target_id=target_id,
                            relationship_type="same_part",
                            strength=0.8,
                            description="Same regulatory part"
                        )
                        relationships.append(relationship)
        
        print(f"Found {len(relationships)} hierarchical relationships")
        return relationships
    
    def build_graph(self, similarity_threshold: float = 0.3) -> nx.Graph:
        """Build the complete regulation graph."""
        print("Building regulation relationship graph...")
        
        # Load sections
        self.sections = self.load_sections_from_db()
        
        # Detect all types of relationships
        semantic_rels = self.detect_semantic_relationships(similarity_threshold)
        cross_ref_rels = self.detect_cross_references()
        hierarchical_rels = self.detect_hierarchical_relationships()
        
        self.relationships = semantic_rels + cross_ref_rels + hierarchical_rels
        
        # Build NetworkX graph
        self.graph = nx.Graph()
        
        # Add nodes
        for section_id, section in self.sections.items():
            self.graph.add_node(
                section_id,
                section_number=section.section_number,
                subject=section.subject,
                part_heading=section.part_heading,
                subchapter_name=section.subchapter_name,
                chapter_name=section.chapter_name,
                text_length=len(section.text)
            )
        
        # Add edges
        for rel in self.relationships:
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                relationship_type=rel.relationship_type,
                strength=rel.strength,
                description=rel.description
            )
        
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def get_graph_statistics(self) -> Dict:
        """Get comprehensive graph statistics."""
        if not self.graph:
            return {}
        
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
            "average_clustering": nx.average_clustering(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
        }
        
        # Relationship type distribution
        rel_types = {}
        for rel in self.relationships:
            rel_type = rel.relationship_type
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        stats["relationship_types"] = rel_types
        
        return stats
    
    def create_plotly_visualization(self, max_nodes: int = 100) -> go.Figure:
        """Create an interactive Plotly visualization of the graph."""
        if not self.graph:
            return go.Figure()
        
        # Sample nodes if graph is too large
        if self.graph.number_of_nodes() > max_nodes:
            # Get most connected nodes
            degrees = dict(self.graph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            sampled_nodes = set([node for node, _ in top_nodes])
            
            # Create subgraph
            subgraph = self.graph.subgraph(sampled_nodes)
        else:
            subgraph = self.graph
        
        # Get positions using spring layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge attributes
            edge_attrs = subgraph[edge[0]][edge[1]]
            edge_info.append(f"Type: {edge_attrs.get('relationship_type', 'unknown')}<br>"
                           f"Strength: {edge_attrs.get('strength', 0):.3f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        
        # Color mapping for different relationship types
        color_map = {
            'semantic_similarity': '#ff7f0e',
            'cross_reference': '#2ca02c',
            'same_part': '#1f77b4'
        }
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Get node attributes
            attrs = subgraph.nodes[node]
            section_number = attrs.get('section_number', 'Unknown')
            subject = attrs.get('subject', 'No subject')[:50]
            
            node_text.append(section_number)
            node_info.append(f"Section: {section_number}<br>"
                           f"Subject: {subject}<br>"
                           f"Part: {attrs.get('part_heading', 'Unknown')[:30]}...")
            
            # Determine node color based on most common relationship type
            node_edges = list(subgraph.edges(node))
            if node_edges:
                rel_types = [subgraph[edge[0]][edge[1]].get('relationship_type', 'unknown') 
                           for edge in node_edges]
                most_common_type = max(set(rel_types), key=rel_types.count)
                node_colors.append(color_map.get(most_common_type, '#d3d3d3'))
            else:
                node_colors.append('#d3d3d3')
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text='Regulation Section Relationships', font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive graph showing relationships between regulation sections",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def export_graph_data(self, output_file: str = "regulation_graph.json"):
        """Export graph data to JSON format."""
        graph_data = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "total_sections": len(self.sections),
                "total_relationships": len(self.relationships),
                "graph_statistics": self.get_graph_statistics()
            }
        }
        
        # Export nodes
        for section_id, section in self.sections.items():
            graph_data["nodes"].append({
                "id": section_id,
                "section_number": section.section_number,
                "subject": section.subject,
                "part_heading": section.part_heading,
                "subchapter_name": section.subchapter_name,
                "chapter_name": section.chapter_name,
                "text_length": len(section.text)
            })
        
        # Export edges
        for rel in self.relationships:
            graph_data["edges"].append({
                "source": rel.source_id,
                "target": rel.target_id,
                "relationship_type": rel.relationship_type,
                "strength": rel.strength,
                "description": rel.description
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"Graph data exported to {output_file}")


def main():
    """Main function to demonstrate the regulation graph builder."""
    print("Regulation Section Relationship Graph Builder")
    print("=" * 50)
    
    # Initialize the graph builder
    builder = RegulationGraphBuilder()
    
    # Build the graph
    graph = builder.build_graph(similarity_threshold=0.25)
    
    # Get statistics
    stats = builder.get_graph_statistics()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create visualization
    print("\nCreating visualization...")
    fig = builder.create_plotly_visualization(max_nodes=50)
    
    # Save the figure
    fig.write_html("regulation_relationships.html")
    print("Interactive visualization saved to regulation_relationships.html")
    
    # Export graph data
    builder.export_graph_data("regulation_graph.json")
    
    print("\nGraph building completed!")


if __name__ == "__main__":
    main()