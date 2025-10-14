#!/usr/bin/env python3
"""
Interactive Graph Visualization Dashboard for CPSC Regulation Knowledge Graph
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import base64

# Local imports
from graph_builder import RegulationGraphBuilder
from data_loader import DataLoader
from entities import KnowledgeGraph, GraphNode, GraphEdge
from config import get_config, DASHBOARD_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG["title"],
    page_icon=DASHBOARD_CONFIG["page_icon"],
    layout=DASHBOARD_CONFIG["layout"],
    initial_sidebar_state=DASHBOARD_CONFIG["sidebar_state"]
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .panel-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-left: 4px solid #3498db;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .node-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .relationship-info {
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 3px;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
    .stDataFrame {
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_graph_data():
    """Load graph data with caching"""
    try:
        builder = RegulationGraphBuilder()
        graph = builder.build_graph()
        return graph, builder
    except Exception as e:
        st.error(f"Failed to load graph data: {e}")
        return None, None

@st.cache_data
def load_hierarchical_data():
    """Load hierarchical data with caching"""
    try:
        data_loader = DataLoader()
        return data_loader.get_hierarchical_structure()
    except Exception as e:
        st.error(f"Failed to load hierarchical data: {e}")
        return None

def create_network_visualization(graph: KnowledgeGraph, selected_nodes: List[str] = None):
    """Create interactive network visualization"""
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node_id, node in graph.nodes.items():
        if selected_nodes and node_id not in selected_nodes:
            continue
            
        G.add_node(
            node_id,
            name=node.entity.name,
            type=node.entity.entity_type,
            degree=node.degree,
            centrality=node.centrality_score or 0
        )
    
    # Add edges
    for edge_id, edge in graph.edges.items():
        if (selected_nodes and 
            edge.source_node not in selected_nodes and 
            edge.target_node not in selected_nodes):
            continue
            
        G.add_edge(
            edge.source_node,
            edge.target_node,
            relationship_type=edge.relationship.relationship_type.value,
            weight=edge.weight,
            confidence=edge.relationship.confidence
        )
    
    # Calculate layout
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, k=1, iterations=50)
    else:
        return None
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[G.nodes[node]['name'] for node in G.nodes()],
        textposition="middle center",
        showlegend=False,
        marker=dict(
            size=[max(10, min(50, G.nodes[node]['degree'] * 3)) for node in G.nodes()],
            color=[G.nodes[node]['centrality'] for node in G.nodes()],
            colorscale='Viridis',
            colorbar=dict(title="Centrality"),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace],
                   layout=go.Layout(
                       title='CPSC Regulation Knowledge Graph',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Interactive network visualization of regulation relationships",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor='left', yanchor='bottom',
                           font=dict(color='#888', size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white'
                   ))
    
    return fig

def create_hierarchical_visualization(hierarchical_data: Dict[str, Any]):
    """Create hierarchical visualization of regulation structure"""
    if not hierarchical_data:
        return None
    
    # Create sunburst chart
    labels = []
    parents = []
    values = []
    colors = []
    
    # Add chapters
    for chapter in hierarchical_data['chapters']:
        labels.append(chapter['name'])
        parents.append('')
        values.append(chapter.get('section_count', 0))
        colors.append('lightblue')
        
        # Add parts
        for subchapter in chapter['subchapters']:
            for part in subchapter['parts']:
                labels.append(part['name'])
                parents.append(chapter['name'])
                values.append(len(part['sections']))
                colors.append('lightgreen')
                
                # Add sections
                for section in part['sections']:
                    labels.append(section['number'])
                    parents.append(part['name'])
                    values.append(1)
                    colors.append('lightcoral')
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
        marker=dict(colors=colors)
    ))
    
    fig.update_layout(
        title="Regulation Hierarchy",
        font_size=12,
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    return fig

def create_relationship_analysis(graph: KnowledgeGraph):
    """Create relationship analysis visualizations"""
    if not graph.edges:
        return None, None
    
    # Analyze relationship types
    rel_types = {}
    rel_weights = []
    
    for edge in graph.edges.values():
        rel_type = edge.relationship.relationship_type.value
        if rel_type not in rel_types:
            rel_types[rel_type] = 0
        rel_types[rel_type] += 1
        rel_weights.append(edge.weight)
    
    # Relationship type distribution
    rel_df = pd.DataFrame(list(rel_types.items()), columns=['Type', 'Count'])
    rel_fig = px.pie(rel_df, values='Count', names='Type', 
                     title='Relationship Type Distribution')
    
    # Weight distribution
    weight_fig = px.histogram(x=rel_weights, nbins=20,
                             title='Relationship Weight Distribution',
                             labels={'x': 'Weight', 'y': 'Count'})
    
    return rel_fig, weight_fig

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<div class="main-header">🕸️ CPSC Regulation Knowledge Graph Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Load data
    with st.spinner("Loading graph data..."):
        graph, builder = load_graph_data()
        hierarchical_data = load_hierarchical_data()
    
    if graph is None:
        st.error("Failed to load graph data. Please check the configuration.")
        return
    
    # Sidebar filters
    st.sidebar.subheader("Filters")
    
    # Node type filter
    node_types = list(set(node.entity.entity_type for node in graph.nodes.values()))
    selected_types = st.sidebar.multiselect(
        "Entity Types",
        options=node_types,
        default=node_types
    )
    
    # Relationship type filter
    rel_types = list(set(edge.relationship.relationship_type.value for edge in graph.edges.values()))
    selected_rel_types = st.sidebar.multiselect(
        "Relationship Types",
        options=rel_types,
        default=rel_types
    )
    
    # Weight threshold
    weight_threshold = st.sidebar.slider(
        "Minimum Relationship Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )
    
    # Filter nodes and edges
    filtered_nodes = {
        node_id: node for node_id, node in graph.nodes.items()
        if node.entity.entity_type in selected_types
    }
    
    filtered_edges = {
        edge_id: edge for edge_id, edge in graph.edges.items()
        if (edge.relationship.relationship_type.value in selected_rel_types and
            edge.weight >= weight_threshold)
    }
    
    # Create filtered graph
    filtered_graph = KnowledgeGraph()
    filtered_graph.nodes = filtered_nodes
    filtered_graph.edges = filtered_edges
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🕸️ Network View", "🏗️ Hierarchy", "🔍 Analysis"])
    
    with tab1:
        st.markdown('<div class="panel-header">Graph Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Nodes", len(graph.nodes))
        with col2:
            st.metric("Total Edges", len(graph.edges))
        with col3:
            st.metric("Filtered Nodes", len(filtered_nodes))
        with col4:
            st.metric("Filtered Edges", len(filtered_edges))
        
        # Entity type distribution
        entity_counts = {}
        for node in graph.nodes.values():
            entity_type = node.entity.entity_type
            if entity_type not in entity_counts:
                entity_counts[entity_type] = 0
            entity_counts[entity_type] += 1
        
        entity_df = pd.DataFrame(list(entity_counts.items()), columns=['Type', 'Count'])
        fig_entities = px.bar(entity_df, x='Type', y='Count', 
                             title='Entity Type Distribution')
        st.plotly_chart(fig_entities, use_container_width=True)
        
        # Relationship analysis
        rel_fig, weight_fig = create_relationship_analysis(graph)
        if rel_fig and weight_fig:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(rel_fig, use_container_width=True)
            with col2:
                st.plotly_chart(weight_fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="panel-header">Interactive Network Visualization</div>', unsafe_allow_html=True)
        
        # Network visualization
        network_fig = create_network_visualization(filtered_graph)
        if network_fig:
            st.plotly_chart(network_fig, use_container_width=True)
        else:
            st.info("No nodes to display with current filters")
        
        # Node details
        if filtered_nodes:
            st.subheader("Node Details")
            selected_node = st.selectbox(
                "Select a node to view details:",
                options=list(filtered_nodes.keys()),
                format_func=lambda x: f"{filtered_nodes[x].entity.name} ({filtered_nodes[x].entity.entity_type})"
            )
            
            if selected_node:
                node = filtered_nodes[selected_node]
                st.markdown(f'<div class="node-info">', unsafe_allow_html=True)
                st.write(f"**Name:** {node.entity.name}")
                st.write(f"**Type:** {node.entity.entity_type}")
                st.write(f"**Degree:** {node.degree}")
                st.write(f"**Centrality:** {node.centrality_score or 'N/A'}")
                
                if hasattr(node.entity, 'description') and node.entity.description:
                    st.write(f"**Description:** {node.entity.description}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show relationships
                st.subheader("Relationships")
                relationships = filtered_graph.get_edges(selected_node)
                for edge in relationships:
                    other_node_id = edge.target_node if edge.source_node == selected_node else edge.source_node
                    other_node = filtered_graph.get_node(other_node_id)
                    if other_node:
                        st.markdown(f'<div class="relationship-info">', unsafe_allow_html=True)
                        st.write(f"**{edge.relationship.relationship_type.value}** → {other_node.entity.name}")
                        st.write(f"Weight: {edge.weight:.3f}, Confidence: {edge.relationship.confidence:.3f}")
                        if edge.relationship.context:
                            st.write(f"Context: {edge.relationship.context}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="panel-header">Regulation Hierarchy</div>', unsafe_allow_html=True)
        
        # Hierarchical visualization
        hierarchy_fig = create_hierarchical_visualization(hierarchical_data)
        if hierarchy_fig:
            st.plotly_chart(hierarchy_fig, use_container_width=True)
        
        # Hierarchy details
        if hierarchical_data:
            st.subheader("Hierarchy Details")
            
            # Chapter selection
            chapter_options = [f"{ch['name']} ({ch.get('section_count', 0)} sections)" 
                             for ch in hierarchical_data['chapters']]
            selected_chapter_idx = st.selectbox("Select Chapter:", range(len(chapter_options)), 
                                              format_func=lambda x: chapter_options[x])
            
            if selected_chapter_idx is not None:
                chapter = hierarchical_data['chapters'][selected_chapter_idx]
                st.write(f"**Chapter:** {chapter['name']}")
                st.write(f"**Total Sections:** {chapter.get('section_count', 0)}")
                
                # Show subchapters and parts
                for subchapter in chapter['subchapters']:
                    st.write(f"**Subchapter:** {subchapter['id']}")
                    for part in subchapter['parts']:
                        st.write(f"  - **Part:** {part['name']} ({len(part['sections'])} sections)")
                        for section in part['sections'][:5]:  # Show first 5 sections
                            st.write(f"    - {section['number']}: {section['subject']}")
                        if len(part['sections']) > 5:
                            st.write(f"    ... and {len(part['sections']) - 5} more sections")
    
    with tab4:
        st.markdown('<div class="panel-header">Advanced Analysis</div>', unsafe_allow_html=True)
        
        # Search functionality
        st.subheader("Graph Search")
        search_query = st.text_input("Search for regulations:", placeholder="Enter search terms...")
        
        if search_query and builder:
            with st.spinner("Searching..."):
                search_results = builder.search_graph(search_query, limit=10)
                
                if search_results:
                    st.success(f"Found {len(search_results)} results")
                    for i, result in enumerate(search_results):
                        with st.expander(f"Result {i+1}"):
                            st.json(result)
                else:
                    st.info("No results found")
        
        # Graph statistics
        st.subheader("Graph Statistics")
        
        if filtered_nodes:
            # Calculate basic statistics
            degrees = [node.degree for node in filtered_nodes.values()]
            weights = [edge.weight for edge in filtered_edges.values()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Degree", f"{np.mean(degrees):.2f}")
                st.metric("Max Degree", max(degrees))
                st.metric("Min Degree", min(degrees))
            
            with col2:
                st.metric("Average Weight", f"{np.mean(weights):.2f}")
                st.metric("Max Weight", f"{max(weights):.3f}")
                st.metric("Min Weight", f"{min(weights):.3f}")
        
        # Export functionality
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Graph as JSON"):
                json_data = filtered_graph.model_dump_json(indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="cpsc_regulation_graph.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export as GraphML"):
                graphml_data = builder.export_graph("graphml")
                st.download_button(
                    label="Download GraphML",
                    data=graphml_data,
                    file_name="cpsc_regulation_graph.graphml",
                    mime="application/xml"
                )

if __name__ == "__main__":
    main()