#!/usr/bin/env python3
"""
CPSC Regulation Knowledge Graph Analysis Dashboard
Interactive analysis and visualization of the regulation data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import networkx as nx
from simple_knowledge_graph import SimpleKnowledgeGraph
import sqlite3
from collections import Counter, defaultdict
import re

# Page configuration
st.set_page_config(
    page_title="CPSC Regulation Knowledge Graph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data():
    """Load the knowledge graph data"""
    if 'graph' not in st.session_state:
        with st.spinner("Building knowledge graph..."):
            graph = SimpleKnowledgeGraph()
            graph.build_graph()
            st.session_state.graph = graph
    return st.session_state.graph

def get_database_stats():
    """Get detailed database statistics"""
    conn = sqlite3.connect("/workspace/regulations.db")
    conn.row_factory = sqlite3.Row
    
    stats = {}
    
    # Basic counts
    tables = ['chapters', 'subchapters', 'parts', 'sections']
    for table in tables:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        stats[f"{table}_count"] = cursor.fetchone()[0]
    
    # Text analysis
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            MIN(LENGTH(text)) as min_length,
            MAX(LENGTH(text)) as max_length,
            AVG(LENGTH(text)) as avg_length,
            COUNT(CASE WHEN text IS NOT NULL AND text != '' THEN 1 END) as non_empty_sections
        FROM sections
    """)
    text_stats = cursor.fetchone()
    stats.update(dict(text_stats))
    
    # Word count analysis
    cursor.execute("""
        SELECT 
            section_number,
            subject,
            (LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1) as word_count
        FROM sections 
        WHERE text IS NOT NULL AND text != ''
        ORDER BY word_count DESC 
        LIMIT 10
    """)
    stats['longest_sections'] = [dict(row) for row in cursor.fetchall()]
    
    # Subject analysis
    cursor.execute("""
        SELECT subject, COUNT(*) as count
        FROM sections 
        WHERE subject IS NOT NULL
        GROUP BY subject
        ORDER BY count DESC
        LIMIT 20
    """)
    stats['common_subjects'] = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return stats

def create_hierarchical_visualization(graph):
    """Create hierarchical visualization of the regulation structure"""
    # Create a tree structure
    tree_data = []
    
    for node in graph.nodes.values():
        if node.node_type == 'chapter':
            tree_data.append({
                'id': node.id,
                'parent': '',
                'name': node.name,
                'type': 'chapter',
                'size': node.degree
            })
        elif node.node_type == 'part':
            chapter_id = f"chapter_{node.metadata['chapter_id']}"
            tree_data.append({
                'id': node.id,
                'parent': chapter_id,
                'name': node.name[:50] + "..." if len(node.name) > 50 else node.name,
                'type': 'part',
                'size': node.degree
            })
        elif node.node_type == 'section':
            part_id = f"part_{node.metadata['part_id']}"
            tree_data.append({
                'id': node.id,
                'parent': part_id,
                'name': node.name,
                'type': 'section',
                'size': min(node.degree, 50)  # Cap size for visualization
            })
    
    return tree_data

def create_network_visualization(graph, max_nodes=100):
    """Create network visualization of the knowledge graph"""
    # Get top nodes by degree
    nodes_by_degree = sorted(graph.nodes.values(), key=lambda x: x.degree, reverse=True)
    top_nodes = nodes_by_degree[:max_nodes]
    top_node_ids = {node.id for node in top_nodes}
    
    # Create network data
    nodes_data = []
    edges_data = []
    
    for node in top_nodes:
        nodes_data.append({
            'id': node.id,
            'name': node.name,
            'type': node.node_type,
            'degree': node.degree,
            'size': min(node.degree, 50)
        })
    
    # Add edges between top nodes
    for edge in graph.edges.values():
        if edge.source in top_node_ids and edge.target in top_node_ids:
            edges_data.append({
                'source': edge.source,
                'target': edge.target,
                'weight': edge.weight,
                'type': edge.relationship_type
            })
    
    return nodes_data, edges_data

def analyze_compliance_patterns(graph):
    """Analyze compliance patterns in the regulations"""
    compliance_sections = []
    
    for node in graph.nodes.values():
        if node.node_type == 'section' and node.metadata.get('has_compliance', False):
            compliance_sections.append({
                'section': node.name,
                'subject': node.metadata.get('subject', ''),
                'compliance_score': node.metadata.get('compliance_score', 0),
                'word_count': node.metadata.get('word_count', 0),
                'part_id': node.metadata.get('part_id', 0)
            })
    
    return pd.DataFrame(compliance_sections)

def extract_keywords(text, top_n=20):
    """Extract most common keywords from text"""
    # Simple keyword extraction
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Filter out common stop words
    stop_words = {
        'that', 'this', 'with', 'have', 'from', 'they', 'been', 'were', 'said', 'each',
        'which', 'their', 'time', 'will', 'about', 'there', 'could', 'other', 'more',
        'very', 'what', 'know', 'just', 'first', 'also', 'after', 'back', 'well',
        'work', 'last', 'right', 'through', 'before', 'years', 'much', 'good', 'man',
        'little', 'own', 'see', 'him', 'two', 'how', 'its', 'who', 'oil', 'sit'
    }
    
    filtered_words = [w for w in words if w not in stop_words]
    return Counter(filtered_words).most_common(top_n)

def main():
    """Main dashboard application"""
    st.title("🕸️ CPSC Regulation Knowledge Graph Analysis")
    st.markdown("---")
    
    # Load data
    graph = load_data()
    db_stats = get_database_stats()
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    
    # Overview metrics
    st.sidebar.metric("Total Sections", db_stats['sections_count'])
    st.sidebar.metric("Total Parts", db_stats['parts_count'])
    st.sidebar.metric("Total Chapters", db_stats['chapters_count'])
    st.sidebar.metric("Avg Text Length", f"{db_stats['avg_length']:.0f} chars")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🔍 Search & Explore", "🕸️ Network Analysis", 
        "⚖️ Compliance Analysis", "📊 Data Insights"
    ])
    
    with tab1:
        st.header("📈 Knowledge Graph Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graph statistics
            stats = graph.get_graph_statistics()
            
            st.subheader("Graph Structure")
            st.metric("Total Nodes", stats['total_nodes'])
            st.metric("Total Edges", stats['total_edges'])
            st.metric("Avg Node Degree", f"{stats['degree_stats']['avg']:.1f}")
            
            # Node type distribution
            node_types = stats['node_types']
            fig_pie = px.pie(
                values=list(node_types.values()),
                names=list(node_types.keys()),
                title="Node Type Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Relationship types
            rel_types = stats['relationship_types']
            fig_bar = px.bar(
                x=list(rel_types.keys()),
                y=list(rel_types.values()),
                title="Relationship Types Distribution"
            )
            fig_bar.update_xaxis(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Compliance statistics
            if stats['compliance_stats']['total_compliance_sections'] > 0:
                st.subheader("Compliance Analysis")
                st.metric(
                    "Compliance Sections", 
                    stats['compliance_stats']['total_compliance_sections']
                )
                st.metric(
                    "Avg Compliance Score", 
                    f"{stats['compliance_stats']['avg_compliance_score']:.1f}"
                )
    
    with tab2:
        st.header("🔍 Search & Explore Regulations")
        
        # Search functionality
        search_query = st.text_input("Search regulations:", placeholder="Enter keywords like 'safety', 'hazard', 'testing'...")
        
        if search_query:
            search_results = graph.search(search_query, limit=20)
            
            if search_results:
                st.subheader(f"Search Results for '{search_query}' ({len(search_results)} found)")
                
                for i, result in enumerate(search_results, 1):
                    node = result['node']
                    with st.expander(f"{i}. {node.name}: {node.metadata.get('subject', 'No subject')}"):
                        st.write(f"**Section:** {node.name}")
                        st.write(f"**Subject:** {node.metadata.get('subject', 'No subject')}")
                        st.write(f"**Word Count:** {node.metadata.get('word_count', 0)}")
                        st.write(f"**Compliance Score:** {node.metadata.get('compliance_score', 0)}")
                        st.write(f"**Connections:** {node.degree}")
                        
                        # Show content preview
                        content = node.content
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.write(f"**Content Preview:** {content}")
            else:
                st.warning("No results found for your search query.")
        
        # Section details
        st.subheader("Section Details")
        section_options = [f"{node.name}: {node.metadata.get('subject', 'No subject')}" 
                          for node in graph.nodes.values() if node.node_type == 'section']
        
        selected_section = st.selectbox("Select a section to explore:", section_options)
        
        if selected_section:
            section_name = selected_section.split(':')[0]
            section_node = next((node for node in graph.nodes.values() 
                               if node.name == section_name), None)
            
            if section_node:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Section:** {section_node.name}")
                    st.write(f"**Subject:** {section_node.metadata.get('subject', 'No subject')}")
                    st.write(f"**Part ID:** {section_node.metadata.get('part_id', 'N/A')}")
                    st.write(f"**Word Count:** {section_node.metadata.get('word_count', 0)}")
                    st.write(f"**Character Count:** {section_node.metadata.get('char_count', 0)}")
                    st.write(f"**Connections:** {section_node.degree}")
                
                with col2:
                    # Show related sections
                    related = graph.get_related_nodes(section_node.id, limit=10)
                    if related:
                        st.write("**Related Sections:**")
                        for rel in related[:5]:
                            st.write(f"• {rel['node'].name} ({rel['relationship_type']}, {rel['weight']:.3f})")
    
    with tab3:
        st.header("🕸️ Network Analysis")
        
        # Network visualization
        st.subheader("Regulation Network")
        
        max_nodes = st.slider("Maximum nodes to display:", 50, 200, 100)
        
        nodes_data, edges_data = create_network_visualization(graph, max_nodes)
        
        if nodes_data and edges_data:
            # Create network plot
            fig = go.Figure()
            
            # Add edges
            for edge in edges_data:
                source_node = next(n for n in nodes_data if n['id'] == edge['source'])
                target_node = next(n for n in nodes_data if n['id'] == edge['target'])
                
                fig.add_trace(go.Scatter(
                    x=[source_node['id'], target_node['id']],
                    y=[source_node['degree'], target_node['degree']],
                    mode='lines',
                    line=dict(width=edge['weight'] * 2, color='lightgray'),
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            # Add nodes
            for node in nodes_data:
                fig.add_trace(go.Scatter(
                    x=[node['id']],
                    y=[node['degree']],
                    mode='markers',
                    marker=dict(
                        size=node['size'],
                        color=node['degree'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Node Degree")
                    ),
                    text=node['name'],
                    hovertemplate=f"<b>{node['name']}</b><br>Type: {node['type']}<br>Degree: {node['degree']}<extra></extra>",
                    name=node['type']
                ))
            
            fig.update_layout(
                title="Regulation Network (Top Nodes by Degree)",
                xaxis_title="Node ID",
                yaxis_title="Node Degree",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Clustering analysis
        st.subheader("Regulation Clusters")
        clusters = graph.find_clusters(min_cluster_size=3)
        
        if clusters:
            st.write(f"Found {len(clusters)} clusters of related regulations:")
            
            for i, cluster in enumerate(clusters, 1):
                with st.expander(f"Cluster {i}: {cluster['size']} regulations (Representative: {cluster['representative']})"):
                    st.write(f"**Size:** {cluster['size']} regulations")
                    st.write(f"**Representative:** {cluster['representative']}")
                    
                    # Show some nodes in the cluster
                    cluster_nodes = [graph.nodes[node_id] for node_id in cluster['nodes'][:10]]
                    for node in cluster_nodes:
                        st.write(f"• {node.name}: {node.metadata.get('subject', 'No subject')}")
                    
                    if len(cluster['nodes']) > 10:
                        st.write(f"... and {len(cluster['nodes']) - 10} more regulations")
        else:
            st.info("No significant clusters found with the current parameters.")
    
    with tab4:
        st.header("⚖️ Compliance Analysis")
        
        # Compliance patterns
        compliance_df = analyze_compliance_patterns(graph)
        
        if not compliance_df.empty:
            st.subheader("Compliance Section Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Compliance score distribution
                fig_hist = px.histogram(
                    compliance_df, 
                    x='compliance_score',
                    title="Distribution of Compliance Scores",
                    nbins=20
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Top compliance sections
                top_compliance = compliance_df.nlargest(10, 'compliance_score')
                fig_bar = px.bar(
                    top_compliance,
                    x='compliance_score',
                    y='section',
                    orientation='h',
                    title="Top 10 Compliance Sections"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Compliance by part
            part_compliance = compliance_df.groupby('part_id')['compliance_score'].agg(['mean', 'count']).reset_index()
            part_compliance = part_compliance[part_compliance['count'] >= 3]  # Parts with at least 3 compliance sections
            
            if not part_compliance.empty:
                fig_scatter = px.scatter(
                    part_compliance,
                    x='count',
                    y='mean',
                    size='count',
                    title="Compliance Score by Part (Parts with 3+ compliance sections)",
                    labels={'count': 'Number of Compliance Sections', 'mean': 'Average Compliance Score'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Detailed compliance table
            st.subheader("Compliance Sections Details")
            st.dataframe(
                compliance_df[['section', 'subject', 'compliance_score', 'word_count']].sort_values('compliance_score', ascending=False),
                use_container_width=True
            )
        else:
            st.info("No compliance sections found in the current analysis.")
    
    with tab5:
        st.header("📊 Data Insights")
        
        # Text analysis
        st.subheader("Text Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Longest sections
            st.write("**Longest Sections by Word Count:**")
            for section in db_stats['longest_sections'][:5]:
                st.write(f"• {section['section_number']}: {section['subject']} ({section['word_count']} words)")
        
        with col2:
            # Common subjects
            st.write("**Most Common Section Subjects:**")
            for subject in db_stats['common_subjects'][:5]:
                st.write(f"• {subject['subject']}: {subject['count']} sections")
        
        # Keyword analysis
        st.subheader("Keyword Analysis")
        
        # Combine all section text
        all_text = " ".join([node.content for node in graph.nodes.values() if node.node_type == 'section'])
        
        if all_text:
            keywords = extract_keywords(all_text, top_n=30)
            
            # Create keyword frequency chart
            keyword_df = pd.DataFrame(keywords, columns=['keyword', 'frequency'])
            
            fig_keywords = px.bar(
                keyword_df.head(20),
                x='frequency',
                y='keyword',
                orientation='h',
                title="Most Common Keywords in Regulations"
            )
            st.plotly_chart(fig_keywords, use_container_width=True)
        
        # Export options
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Graph as JSON"):
                json_data = graph.export_to_json()
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="cpsc_knowledge_graph.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export Compliance Analysis"):
                csv_data = compliance_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="compliance_analysis.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()