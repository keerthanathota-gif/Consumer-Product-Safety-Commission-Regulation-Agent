#!/usr/bin/env python3
"""
Enhanced Dashboard with Graphiti Integration for Regulation Relationships
======================================================================

This dashboard integrates the regulation relationship graph visualization
with the existing CPSC redundancy analysis dashboard.
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List
import sqlite3
import networkx as nx
from graphiti_integration import RegulationGraphBuilder

# Page configuration
st.set_page_config(
    page_title="CPSC Regulation Relationship Dashboard",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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
    .graph-container {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        background-color: #fafafa;
    }
    .stDataFrame {
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data
def load_regulation_graph():
    """Load or build the regulation relationship graph."""
    try:
        # Try to load existing graph data
        with open("regulation_graph.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Build graph if not exists
        st.info("Building regulation relationship graph... This may take a moment.")
        builder = RegulationGraphBuilder()
        graph = builder.build_graph(similarity_threshold=0.25)
        builder.export_graph_data("regulation_graph.json")
        
        # Load the exported data
        with open("regulation_graph.json", 'r', encoding='utf-8') as f:
            return json.load(f)

@st.cache_data
def get_graph_statistics(graph_data: Dict):
    """Calculate comprehensive graph statistics."""
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    # Basic statistics
    total_nodes = len(nodes)
    total_edges = len(edges)
    
    # Relationship type distribution
    rel_types = {}
    for edge in edges:
        rel_type = edge.get('relationship_type', 'unknown')
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    # Node degree distribution
    node_degrees = {}
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        node_degrees[source] = node_degrees.get(source, 0) + 1
        node_degrees[target] = node_degrees.get(target, 0) + 1
    
    avg_degree = sum(node_degrees.values()) / max(total_nodes, 1)
    max_degree = max(node_degrees.values()) if node_degrees else 0
    
    # Part distribution
    parts = {}
    for node in nodes:
        part = node.get('part_heading', 'Unknown')
        parts[part] = parts.get(part, 0) + 1
    
    return {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'relationship_types': rel_types,
        'average_degree': avg_degree,
        'max_degree': max_degree,
        'part_distribution': parts
    }

# ============================================================================
# GRAPH VISUALIZATION FUNCTIONS
# ============================================================================

def create_network_visualization(graph_data: Dict, max_nodes: int = 100, 
                                relationship_filter: List[str] = None):
    """Create an interactive network visualization."""
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    if not nodes or not edges:
        return go.Figure()
    
    # Filter edges by relationship type
    if relationship_filter:
        filtered_edges = [e for e in edges if e.get('relationship_type') in relationship_filter]
    else:
        filtered_edges = edges
    
    # Get nodes that are connected by filtered edges
    connected_nodes = set()
    for edge in filtered_edges:
        connected_nodes.add(edge.get('source'))
        connected_nodes.add(edge.get('target'))
    
    # Sample nodes if too many
    if len(connected_nodes) > max_nodes:
        # Get most connected nodes
        node_degrees = {}
        for edge in filtered_edges:
            source = edge.get('source')
            target = edge.get('target')
            node_degrees[source] = node_degrees.get(source, 0) + 1
            node_degrees[target] = node_degrees.get(target, 0) + 1
        
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        sampled_nodes = set([node for node, _ in top_nodes])
    else:
        sampled_nodes = connected_nodes
    
    # Filter edges to only include sampled nodes
    filtered_edges = [e for e in filtered_edges 
                     if e.get('source') in sampled_nodes and e.get('target') in sampled_nodes]
    
    # Create NetworkX graph for layout
    G = nx.Graph()
    for node in nodes:
        if node.get('id') in sampled_nodes:
            G.add_node(node.get('id'), **node)
    
    for edge in filtered_edges:
        G.add_edge(edge.get('source'), edge.get('target'), **edge)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in filtered_edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_info.append(f"Type: {edge.get('relationship_type', 'unknown')}<br>"
                           f"Strength: {edge.get('strength', 0):.3f}")
    
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
    
    for node in nodes:
        node_id = node.get('id')
        if node_id in sampled_nodes and node_id in pos:
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            section_number = node.get('section_number', 'Unknown')
            subject = node.get('subject', 'No subject')[:50]
            
            node_text.append(section_number)
            node_info.append(f"Section: {section_number}<br>"
                           f"Subject: {subject}<br>"
                           f"Part: {node.get('part_heading', 'Unknown')[:30]}...")
            
            # Determine node color based on most common relationship type
            node_edges = [e for e in filtered_edges 
                         if e.get('source') == node_id or e.get('target') == node_id]
            if node_edges:
                rel_types = [e.get('relationship_type', 'unknown') for e in node_edges]
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
                           text=f"Showing {len(sampled_nodes)} sections with {len(filtered_edges)} relationships",
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

def create_relationship_analysis_charts(graph_data: Dict):
    """Create analysis charts for relationship data."""
    edges = graph_data.get('edges', [])
    nodes = graph_data.get('nodes', [])
    
    if not edges:
        return None, None, None
    
    # Relationship type distribution
    rel_types = {}
    for edge in edges:
        rel_type = edge.get('relationship_type', 'unknown')
        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
    rel_df = pd.DataFrame(list(rel_types.items()), columns=['Type', 'Count'])
    rel_fig = px.pie(rel_df, values='Count', names='Type', 
                     title='Relationship Type Distribution')
    
    # Strength distribution
    strengths = [edge.get('strength', 0) for edge in edges]
    strength_fig = px.histogram(x=strengths, nbins=20, 
                               title='Relationship Strength Distribution',
                               labels={'x': 'Strength', 'y': 'Count'})
    
    # Part distribution
    parts = {}
    for node in nodes:
        part = node.get('part_heading', 'Unknown')
        parts[part] = parts.get(part, 0) + 1
    
    # Get top 10 parts
    top_parts = sorted(parts.items(), key=lambda x: x[1], reverse=True)[:10]
    part_df = pd.DataFrame(top_parts, columns=['Part', 'Sections'])
    part_fig = px.bar(part_df, x='Sections', y='Part', orientation='h',
                     title='Top 10 Parts by Section Count')
    
    return rel_fig, strength_fig, part_fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.markdown('<div class="main-header">🕸️ CPSC Regulation Relationship Dashboard</div>', unsafe_allow_html=True)

# Load graph data
graph_data = load_regulation_graph()
stats = get_graph_statistics(graph_data)

# Sidebar controls
st.sidebar.title("🎯 Graph Controls")
st.sidebar.markdown("---")

# Graph filters
max_nodes = st.sidebar.slider(
    "Maximum Nodes to Display",
    min_value=10,
    max_value=200,
    value=50,
    step=10
)

relationship_types = list(stats['relationship_types'].keys())
selected_relationships = st.sidebar.multiselect(
    "Relationship Types to Show",
    options=relationship_types,
    default=relationship_types
)

similarity_threshold = st.sidebar.slider(
    "Minimum Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

# Filter edges by similarity threshold
filtered_edges = [e for e in graph_data.get('edges', []) 
                 if e.get('strength', 0) >= similarity_threshold]
filtered_graph_data = {
    'nodes': graph_data.get('nodes', []),
    'edges': filtered_edges
}

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Sections", stats['total_nodes'])

with col2:
    st.metric("Total Relationships", stats['total_edges'])

with col3:
    st.metric("Average Connections", f"{stats['average_degree']:.1f}")

with col4:
    st.metric("Max Connections", stats['max_degree'])

# Graph visualization
st.markdown('<div class="panel-header">Network Visualization</div>', unsafe_allow_html=True)

with st.spinner("Generating network visualization..."):
    fig = create_network_visualization(
        filtered_graph_data, 
        max_nodes=max_nodes,
        relationship_filter=selected_relationships
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Analysis charts
st.markdown('<div class="panel-header">Relationship Analysis</div>', unsafe_allow_html=True)

rel_fig, strength_fig, part_fig = create_relationship_analysis_charts(filtered_graph_data)

if rel_fig and strength_fig and part_fig:
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(rel_fig, use_container_width=True)
    
    with col2:
        st.plotly_chart(strength_fig, use_container_width=True)
    
    st.plotly_chart(part_fig, use_container_width=True)

# Detailed relationship table
st.markdown('<div class="panel-header">Detailed Relationship Data</div>', unsafe_allow_html=True)

if filtered_edges:
    # Create a detailed table
    df_data = []
    for i, edge in enumerate(filtered_edges[:100]):  # Limit to first 100 for performance
        source_node = next((n for n in graph_data['nodes'] if n['id'] == edge['source']), {})
        target_node = next((n for n in graph_data['nodes'] if n['id'] == edge['target']), {})
        
        df_data.append({
            'ID': i + 1,
            'Type': edge.get('relationship_type', 'unknown'),
            'Strength': f"{edge.get('strength', 0):.3f}",
            'Source Section': source_node.get('section_number', 'Unknown'),
            'Target Section': target_node.get('section_number', 'Unknown'),
            'Source Subject': source_node.get('subject', 'No subject')[:50] + "...",
            'Target Subject': target_node.get('subject', 'No subject')[:50] + "..."
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, height=400)
else:
    st.warning("No relationships found with the current filters.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>"
    "🕸️ CPSC Regulation Relationship Dashboard v1.0<br>"
    "Powered by NetworkX and Plotly | Interactive Graph Visualization"
    "</div>",
    unsafe_allow_html=True
)