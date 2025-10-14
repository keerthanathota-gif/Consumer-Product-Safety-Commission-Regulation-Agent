#!/usr/bin/env python3
"""
Integrated CPSC Dashboard with Graphiti Relationship Visualization
================================================================

This comprehensive dashboard combines:
1. Original redundancy analysis
2. Regulation relationship graph visualization
3. Interactive controls and filtering
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
    page_title="CPSC Integrated Analysis Dashboard",
    page_icon="📊",
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
    .tab-content {
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================

@st.cache_data
def load_redundancy_report():
    """Load the redundancy analysis report."""
    try:
        with open("enterprise_redundancy_report.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_regulation_graph():
    """Load or build the regulation relationship graph."""
    try:
        with open("regulation_graph.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.info("Building regulation relationship graph... This may take a moment.")
        builder = RegulationGraphBuilder()
        graph = builder.build_graph(similarity_threshold=0.25)
        builder.export_graph_data("regulation_graph.json")
        
        with open("regulation_graph.json", 'r', encoding='utf-8') as f:
            return json.load(f)

@st.cache_data
def get_graph_statistics(graph_data: Dict):
    """Calculate comprehensive graph statistics."""
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
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
    
    return {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'relationship_types': rel_types,
        'average_degree': avg_degree,
        'max_degree': max_degree
    }

# ============================================================================
# GRAPH VISUALIZATION FUNCTIONS
# ============================================================================

def create_network_visualization(graph_data: Dict, max_nodes: int = 100, 
                                relationship_filter: List[str] = None,
                                similarity_threshold: float = 0.3):
    """Create an interactive network visualization."""
    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    
    if not nodes or not edges:
        return go.Figure()
    
    # Filter edges by relationship type and similarity
    filtered_edges = [e for e in edges 
                     if e.get('strength', 0) >= similarity_threshold]
    
    if relationship_filter:
        filtered_edges = [e for e in filtered_edges 
                         if e.get('relationship_type') in relationship_filter]
    
    # Get connected nodes
    connected_nodes = set()
    for edge in filtered_edges:
        connected_nodes.add(edge.get('source'))
        connected_nodes.add(edge.get('target'))
    
    # Sample nodes if too many
    if len(connected_nodes) > max_nodes:
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
    
    if G.number_of_nodes() == 0:
        return go.Figure()
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Prepare edge traces
    edge_x = []
    edge_y = []
    
    for edge in filtered_edges:
        source = edge.get('source')
        target = edge.get('target')
        if source in pos and target in pos:
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
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

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.markdown('<div class="main-header">📊 CPSC Integrated Analysis Dashboard</div>', unsafe_allow_html=True)

# Load data
redundancy_report = load_redundancy_report()
graph_data = load_regulation_graph()
graph_stats = get_graph_statistics(graph_data)

# Sidebar navigation
st.sidebar.title("🎯 Dashboard Navigation")
st.sidebar.markdown("---")

# Main navigation
main_tab = st.sidebar.radio(
    "Select Analysis Type",
    ["Redundancy Analysis", "Relationship Graph", "Combined View"]
)

# Sidebar controls based on selected tab
if main_tab == "Redundancy Analysis":
    st.sidebar.subheader("Redundancy Filters")
    
    if redundancy_report:
        severity_filter = st.sidebar.multiselect(
            "Impact Severity",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        
        similarity_threshold = st.sidebar.slider(
            "Minimum Similarity Score",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05
        )
    else:
        st.sidebar.warning("Redundancy report not found. Run the analyzer first.")

elif main_tab == "Relationship Graph":
    st.sidebar.subheader("Graph Controls")
    
    max_nodes = st.sidebar.slider(
        "Maximum Nodes to Display",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    relationship_types = list(graph_stats['relationship_types'].keys())
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

elif main_tab == "Combined View":
    st.sidebar.subheader("Combined Analysis")
    
    # Graph controls
    max_nodes = st.sidebar.slider(
        "Maximum Graph Nodes",
        min_value=10,
        max_value=100,
        value=30,
        step=10
    )
    
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05
    )
    
    # Redundancy controls
    if redundancy_report:
        severity_filter = st.sidebar.multiselect(
            "Redundancy Severity",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"]
        )

# Main content based on selected tab
if main_tab == "Redundancy Analysis":
    if redundancy_report:
        st.markdown('<div class="panel-header">Redundancy Analysis Dashboard</div>', unsafe_allow_html=True)
        
        # Filter redundancy pairs
        filtered_pairs = [
            p for p in redundancy_report['redundancy_pairs']
            if p['impact_severity'] in severity_filter
            and p['similarity_score'] >= similarity_threshold
        ]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pairs", len(filtered_pairs))
        
        with col2:
            high_count = len([p for p in filtered_pairs if p['impact_severity'] == 'High'])
            st.metric("High Severity", high_count)
        
        with col3:
            medium_count = len([p for p in filtered_pairs if p['impact_severity'] == 'Medium'])
            st.metric("Medium Severity", medium_count)
        
        with col4:
            low_count = len([p for p in filtered_pairs if p['impact_severity'] == 'Low'])
            st.metric("Low Severity", low_count)
        
        # Redundancy pairs table
        if filtered_pairs:
            df_data = []
            for idx, pair in enumerate(filtered_pairs[:50]):  # Limit for performance
                df_data.append({
                    'ID': idx + 1,
                    'Similarity': f"{pair['similarity_score']:.4f}",
                    'Severity': pair['impact_severity'],
                    'Type': pair['overlap_type'],
                    'Section 1': pair['context_info']['section1_number'],
                    'Section 2': pair['context_info']['section2_number'],
                    'Subject 1': pair['context_info']['section1_subject'][:50] + "...",
                    'Subject 2': pair['context_info']['section2_subject'][:50] + "..."
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.warning("No redundancy pairs found with current filters.")
    
    else:
        st.error("Redundancy report not found. Please run the enterprise redundancy analyzer first.")

elif main_tab == "Relationship Graph":
    st.markdown('<div class="panel-header">Regulation Relationship Graph</div>', unsafe_allow_html=True)
    
    # Graph statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sections", graph_stats['total_nodes'])
    
    with col2:
        st.metric("Total Relationships", graph_stats['total_edges'])
    
    with col3:
        st.metric("Average Connections", f"{graph_stats['average_degree']:.1f}")
    
    with col4:
        st.metric("Max Connections", graph_stats['max_degree'])
    
    # Network visualization
    with st.spinner("Generating network visualization..."):
        fig = create_network_visualization(
            graph_data,
            max_nodes=max_nodes,
            relationship_filter=selected_relationships,
            similarity_threshold=similarity_threshold
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Relationship analysis charts
    st.markdown("### Relationship Analysis")
    
    # Relationship type distribution
    rel_types = graph_stats['relationship_types']
    rel_df = pd.DataFrame(list(rel_types.items()), columns=['Type', 'Count'])
    rel_fig = px.pie(rel_df, values='Count', names='Type', 
                     title='Relationship Type Distribution')
    st.plotly_chart(rel_fig, use_container_width=True)

elif main_tab == "Combined View":
    st.markdown('<div class="panel-header">Combined Analysis View</div>', unsafe_allow_html=True)
    
    # Combined metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Sections", graph_stats['total_nodes'])
    
    with col2:
        st.metric("Relationships", graph_stats['total_edges'])
    
    with col3:
        if redundancy_report:
            total_pairs = len(redundancy_report['redundancy_pairs'])
            st.metric("Redundancy Pairs", total_pairs)
        else:
            st.metric("Redundancy Pairs", "N/A")
    
    with col4:
        st.metric("Avg Connections", f"{graph_stats['average_degree']:.1f}")
    
    with col5:
        st.metric("Graph Density", f"{graph_stats['total_edges'] / (graph_stats['total_nodes'] * (graph_stats['total_nodes'] - 1) / 2):.3f}")
    
    # Side-by-side visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Network Graph")
        with st.spinner("Generating network visualization..."):
            fig = create_network_visualization(
                graph_data,
                max_nodes=max_nodes,
                similarity_threshold=similarity_threshold
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Redundancy Analysis")
        if redundancy_report:
            # Filter redundancy pairs
            filtered_pairs = [
                p for p in redundancy_report['redundancy_pairs']
                if p['impact_severity'] in severity_filter
                and p['similarity_score'] >= similarity_threshold
            ]
            
            # Severity distribution
            severity_counts = {}
            for pair in filtered_pairs:
                severity = pair['impact_severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity_counts:
                severity_df = pd.DataFrame(list(severity_counts.items()), columns=['Severity', 'Count'])
                severity_fig = px.bar(severity_df, x='Severity', y='Count', 
                                    title='Redundancy Severity Distribution',
                                    color='Severity',
                                    color_discrete_map={'High': '#d9534f', 'Medium': '#f0ad4e', 'Low': '#5bc0de'})
                st.plotly_chart(severity_fig, use_container_width=True)
            else:
                st.info("No redundancy pairs found with current filters.")
        else:
            st.warning("Redundancy report not available.")
    
    # Detailed analysis section
    st.markdown("### Detailed Analysis")
    
    # Relationship strength vs redundancy similarity comparison
    if redundancy_report and graph_data:
        st.markdown("#### Relationship Strength vs Redundancy Similarity")
        
        # Get relationship strengths
        rel_strengths = [e.get('strength', 0) for e in graph_data.get('edges', [])]
        redun_similarities = [p.get('similarity_score', 0) for p in redundancy_report.get('redundancy_pairs', [])]
        
        if rel_strengths and redun_similarities:
            # Create comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=rel_strengths,
                name='Relationship Strengths',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig.add_trace(go.Histogram(
                x=redun_similarities,
                name='Redundancy Similarities',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig.update_layout(
                title='Distribution Comparison: Relationship Strengths vs Redundancy Similarities',
                xaxis_title='Score',
                yaxis_title='Count',
                barmode='overlay'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>"
    "📊 CPSC Integrated Analysis Dashboard v1.0<br>"
    "Combining Redundancy Analysis with Relationship Graph Visualization"
    "</div>",
    unsafe_allow_html=True
)