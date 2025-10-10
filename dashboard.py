#!/usr/bin/env python3
"""
Professional Interactive Dashboard for CPSC Redundancy Analysis
================================================================

Implements the complete professional dashboard specification:
- Panel 1: Executive Overview
- Panel 2: Interactive Redundancy Listing
- Panel 3: Detailed Analysis View

Launch with: streamlit run dashboard.py
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List
import re

# Page configuration
st.set_page_config(
    page_title="CPSC Redundancy Analysis Dashboard",
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
    .severity-high {
        background-color: #d9534f;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .severity-medium {
        background-color: #f0ad4e;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .severity-low {
        background-color: #5bc0de;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .vague-word {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        font-weight: bold;
        color: #856404;
    }
    .recommendation-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stDataFrame {
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_report(report_path: str = "enterprise_redundancy_report.json") -> Dict:
    """Load the analysis report with caching."""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Report not found at {report_path}")
        st.info("Please run `python enterprise_redundancy_analyzer.py` first to generate the report.")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON format in report file")
        st.stop()


def highlight_vague_words(text: str, vague_words: List[str]) -> str:
    """Highlight vague words in text with HTML spans."""
    highlighted = text
    for word in vague_words:
        # Case-insensitive replacement with HTML highlighting
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<span class="vague-word">{word}</span>',
            highlighted
        )
    return highlighted


# ============================================================================
# LOAD DATA
# ============================================================================

report = load_report()


# ============================================================================
# SIDEBAR - FILTERS AND NAVIGATION
# ============================================================================

st.sidebar.title("🎯 Dashboard Controls")
st.sidebar.markdown("---")

# Filter options
st.sidebar.subheader("Filters")

severity_filter = st.sidebar.multiselect(
    "Impact Severity",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

overlap_filter = st.sidebar.multiselect(
    "Overlap Type",
    options=["Semantic", "Lexical", "Structural"],
    default=["Semantic", "Lexical", "Structural"]
)

similarity_threshold = st.sidebar.slider(
    "Minimum Similarity Score",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05
)

confidence_threshold = st.sidebar.slider(
    "Minimum Confidence Score",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")
panel_selection = st.sidebar.radio(
    "Jump to Panel",
    ["Executive Overview", "Redundancy Listing", "Detailed Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Report Date:** {report['metadata']['analysis_date'][:10]}")
st.sidebar.info(f"**Model:** {report['metadata']['analyzer_config']['model']}")
st.sidebar.info(f"**Total Sections:** {report['metadata']['corpus_statistics']['total_sections']:,}")


# ============================================================================
# MAIN DASHBOARD HEADER
# ============================================================================

st.markdown('<div class="main-header">📊 CPSC Redundancy Analysis Dashboard</div>', unsafe_allow_html=True)


# ============================================================================
# PANEL 1: EXECUTIVE OVERVIEW
# ============================================================================

st.markdown('<div class="panel-header">Panel 1: Executive Overview</div>', unsafe_allow_html=True)

# Filter redundancy pairs
filtered_pairs = [
    p for p in report['redundancy_pairs']
    if p['impact_severity'] in severity_filter
    and p['overlap_type'] in overlap_filter
    and p['similarity_score'] >= similarity_threshold
    and p['confidence'] >= confidence_threshold
]

# Calculate metrics
total_pairs = len(filtered_pairs)
high_count = len([p for p in filtered_pairs if p['impact_severity'] == 'High'])
medium_count = len([p for p in filtered_pairs if p['impact_severity'] == 'Medium'])
low_count = len([p for p in filtered_pairs if p['impact_severity'] == 'Low'])

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📋 Total Redundancy Pairs",
        value=total_pairs,
        delta=f"{(total_pairs/len(report['redundancy_pairs'])*100):.1f}% of all pairs" if report['redundancy_pairs'] else "0%"
    )

with col2:
    st.metric(
        label="🔴 High Severity",
        value=high_count,
        delta=f"{(high_count/max(total_pairs,1)*100):.1f}% of filtered"
    )

with col3:
    st.metric(
        label="🟡 Medium Severity",
        value=medium_count,
        delta=f"{(medium_count/max(total_pairs,1)*100):.1f}% of filtered"
    )

with col4:
    st.metric(
        label="🔵 Low Severity",
        value=low_count,
        delta=f"{(low_count/max(total_pairs,1)*100):.1f}% of filtered"
    )

# Quality Score Section
st.markdown("### 📈 Document Quality Score")
col1, col2, col3, col4 = st.columns(4)

quality = report['quality_metrics']

with col1:
    readability = quality['avg_readability']
    color = "🟢" if readability > 60 else "🟡" if readability > 30 else "🔴"
    st.metric(
        label=f"{color} Readability Score",
        value=f"{readability:.1f}/100"
    )

with col2:
    vague_density = quality['avg_vague_density']
    color = "🟢" if vague_density < 3 else "🟡" if vague_density < 7 else "🔴"
    st.metric(
        label=f"{color} Vague Language Density",
        value=f"{vague_density:.2f}%"
    )

with col3:
    sent_length = quality['avg_sentence_length']
    color = "🟢" if sent_length < 20 else "🟡" if sent_length < 30 else "🔴"
    st.metric(
        label=f"{color} Avg Sentence Length",
        value=f"{sent_length:.1f} words"
    )

with col4:
    consistency = quality['avg_consistency']
    color = "🟢" if consistency > 0.7 else "🟡" if consistency > 0.5 else "🔴"
    st.metric(
        label=f"{color} Structural Consistency",
        value=f"{consistency:.3f}"
    )

# Key Recommendations Summary
st.markdown("### 🎯 Key Recommendations Summary")

exec_summary = report['executive_summary']
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**{exec_summary['finding_1']['title']}**")
    st.info(exec_summary['finding_1']['description'])
    st.success(exec_summary['finding_1']['action'])

with col2:
    st.markdown(f"**{exec_summary['finding_2']['title']}**")
    st.info(exec_summary['finding_2']['description'])
    st.success(exec_summary['finding_2']['action'])

with col3:
    st.markdown(f"**{exec_summary['finding_3']['title']}**")
    st.info(exec_summary['finding_3']['description'])
    st.success(exec_summary['finding_3']['action'])

# Distribution Charts
st.markdown("### 📊 Distribution Analysis")
col1, col2 = st.columns(2)

with col1:
    # Severity Distribution Pie Chart
    severity_data = {
        'Severity': ['High', 'Medium', 'Low'],
        'Count': [high_count, medium_count, low_count],
        'Color': ['#d9534f', '#f0ad4e', '#5bc0de']
    }
    fig_severity = px.pie(
        severity_data,
        values='Count',
        names='Severity',
        title='Impact Severity Distribution',
        color='Severity',
        color_discrete_map={'High': '#d9534f', 'Medium': '#f0ad4e', 'Low': '#5bc0de'}
    )
    fig_severity.update_traces(textposition='inside', textinfo='percent+label+value')
    st.plotly_chart(fig_severity, use_container_width=True)

with col2:
    # Overlap Type Distribution
    overlap_counts = pd.DataFrame([p['overlap_type'] for p in filtered_pairs], columns=['Type']).value_counts().reset_index()
    overlap_counts.columns = ['Overlap Type', 'Count']
    fig_overlap = px.bar(
        overlap_counts,
        x='Overlap Type',
        y='Count',
        title='Overlap Type Distribution',
        color='Overlap Type',
        color_discrete_map={'Semantic': '#5cb85c', 'Lexical': '#5bc0de', 'Structural': '#f0ad4e'}
    )
    st.plotly_chart(fig_overlap, use_container_width=True)


st.markdown("---")


# ============================================================================
# PANEL 2: INTERACTIVE REDUNDANCY LISTING
# ============================================================================

st.markdown('<div class="panel-header">Panel 2: Interactive Redundancy Listing</div>', unsafe_allow_html=True)

if total_pairs == 0:
    st.warning("No redundancy pairs match the current filters. Adjust the filter settings in the sidebar.")
else:
    # Prepare DataFrame
    df_data = []
    for idx, pair in enumerate(filtered_pairs):
        df_data.append({
            'ID': idx,
            'Similarity': f"{pair['similarity_score']:.4f}",
            'Severity': pair['impact_severity'],
            'Type': pair['overlap_type'],
            'Confidence': f"{pair['confidence']:.4f}",
            'Section 1': pair['context_info']['section1_number'],
            'Section 2': pair['context_info']['section2_number'],
            'Subject 1': pair['context_info']['section1_subject'][:50] + "..." if len(pair['context_info']['section1_subject']) > 50 else pair['context_info']['section1_subject'],
            'Subject 2': pair['context_info']['section2_subject'][:50] + "..." if len(pair['context_info']['section2_subject']) > 50 else pair['context_info']['section2_subject'],
            'Vague Words': len(pair['vague_words'])
        })

    df = pd.DataFrame(df_data)

    st.markdown(f"**Showing {len(df)} redundancy pairs** (sorted by similarity score)")

    # Add column configuration for better display
    st.dataframe(
        df,
        column_config={
            "ID": st.column_config.NumberColumn("ID", help="Row identifier"),
            "Similarity": st.column_config.TextColumn("Similarity Score", help="Semantic similarity (0-1)"),
            "Severity": st.column_config.TextColumn("Impact", help="High/Medium/Low"),
            "Type": st.column_config.TextColumn("Overlap Type", help="Semantic/Lexical/Structural"),
            "Confidence": st.column_config.TextColumn("Confidence", help="Detection confidence (0-1)"),
            "Vague Words": st.column_config.NumberColumn("Vague Words", help="Count of vague words detected")
        },
        use_container_width=True,
        height=400
    )

    # Sort options
    st.markdown("### 🔍 Sort & Search")
    col1, col2 = st.columns(2)

    with col1:
        sort_by = st.selectbox(
            "Sort by",
            options=['Similarity', 'Confidence', 'Severity', 'Type', 'Vague Words']
        )

    with col2:
        sort_order = st.radio("Order", options=['Descending', 'Ascending'], horizontal=True)

    # Apply sorting
    ascending = (sort_order == 'Ascending')
    if sort_by in ['Similarity', 'Confidence']:
        df_sorted = df.sort_values(by=sort_by, ascending=ascending, key=lambda x: x.astype(float))
    else:
        df_sorted = df.sort_values(by=sort_by, ascending=ascending)

    st.dataframe(df_sorted, use_container_width=True, height=300)


st.markdown("---")


# ============================================================================
# PANEL 3: DETAILED ANALYSIS VIEW
# ============================================================================

st.markdown('<div class="panel-header">Panel 3: Detailed Analysis View</div>', unsafe_allow_html=True)

if total_pairs == 0:
    st.warning("No redundancy pairs to display. Adjust filters.")
else:
    # Pair selection
    st.markdown("### 🔎 Select Redundancy Pair for Detailed Analysis")

    # Create selection options
    pair_options = [
        f"Pair {idx} | Similarity: {p['similarity_score']:.4f} | {p['impact_severity']} | {p['context_info']['section1_number']} ↔ {p['context_info']['section2_number']}"
        for idx, p in enumerate(filtered_pairs)
    ]

    selected_pair_str = st.selectbox("Select pair:", pair_options)
    selected_idx = int(selected_pair_str.split("|")[0].replace("Pair", "").strip())
    selected_pair = filtered_pairs[selected_idx]

    # Display full pair details
    st.markdown("---")

    # Header with key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Similarity Score", f"{selected_pair['similarity_score']:.4f}")
    with col2:
        severity = selected_pair['impact_severity']
        st.markdown(f"**Impact Severity**<br><span class='severity-{severity.lower()}'>{severity}</span>", unsafe_allow_html=True)
    with col3:
        st.metric("Confidence", f"{selected_pair['confidence']:.4f}")
    with col4:
        st.metric("Overlap Type", selected_pair['overlap_type'])

    st.markdown("---")

    # Full Text Comparison
    st.markdown("### 📝 Full Text Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Primary Section: {selected_pair['context_info']['section1_number']}**")
        st.markdown(f"*Subject: {selected_pair['context_info']['section1_subject']}*")

        # Highlight vague words in text
        primary_highlighted = highlight_vague_words(
            selected_pair['primary_text'],
            selected_pair['vague_words']
        )
        st.markdown(
            f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid #007bff;">{primary_highlighted}</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(f"**Redundant Section: {selected_pair['context_info']['section2_number']}**")
        st.markdown(f"*Subject: {selected_pair['context_info']['section2_subject']}*")

        # Highlight vague words in text
        redundant_highlighted = highlight_vague_words(
            selected_pair['redundant_text'],
            selected_pair['vague_words']
        )
        st.markdown(
            f'<div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; border-left: 4px solid #dc3545;">{redundant_highlighted}</div>',
            unsafe_allow_html=True
        )

    # Semantic Similarity Visualization
    st.markdown("### 📊 Semantic Similarity Visualization")

    # Create gauge chart for similarity
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=selected_pair['similarity_score'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Similarity Score (%)"},
        delta={'reference': 70, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 70], 'color': "#5bc0de"},
                {'range': [70, 85], 'color': "#f0ad4e"},
                {'range': [85, 100], 'color': "#d9534f"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95
            }
        }
    ))

    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Overlap Analysis with Color Coding
    st.markdown("### 🎨 Overlap Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Detected Vague Words:**")
        if selected_pair['vague_words']:
            vague_html = " ".join([f'<span class="vague-word">{word}</span>' for word in selected_pair['vague_words']])
            st.markdown(vague_html, unsafe_allow_html=True)
        else:
            st.success("No vague words detected")

    with col2:
        st.markdown("**Similarity Breakdown:**")
        st.markdown(f"- **Lexical Similarity:** {selected_pair['context_info'].get('lexical_similarity', 'N/A')}")
        st.markdown(f"- **Overlap Type:** {selected_pair['overlap_type']}")
        st.markdown(f"- **Confidence Level:** {selected_pair['confidence']:.2%}")

    # AI-Generated Rewrite Suggestions
    st.markdown("### 💡 AI-Generated Rewrite Recommendation")

    st.markdown(
        f'<div class="recommendation-box"><strong>Recommendation:</strong><br>{selected_pair["rewrite_recommendation"]}</div>',
        unsafe_allow_html=True
    )

    # Impact Assessment
    st.markdown("### ⚖️ Impact Assessment")

    impact_text = {
        'High': "🔴 **Critical Impact** - This redundancy significantly affects document clarity and length. Immediate consolidation is recommended to improve regulatory effectiveness.",
        'Medium': "🟡 **Moderate Impact** - This redundancy contributes to document complexity. Review and deduplication will improve consistency.",
        'Low': "🔵 **Low Impact** - This represents related content that may benefit from cross-referencing rather than consolidation."
    }

    st.markdown(impact_text.get(selected_pair['impact_severity'], "Impact assessment unavailable"))


st.markdown("---")


# ============================================================================
# FOOTER - ACTION PLAN
# ============================================================================

st.markdown("### 📋 Priority-Based Action Plan")

if 'action_plan' in report:
    for action in report['action_plan']:
        priority_color = {
            'CRITICAL': '🔴',
            'HIGH': '🟡',
            'MEDIUM': '🔵'
        }

        with st.expander(f"{priority_color.get(action['priority'], '⚪')} {action['priority']}: {action['action']}"):
            st.markdown(f"**Affected Pairs:** {action['affected_pairs']}")
            st.markdown(f"**Estimated Effort:** {action['estimated_effort']}")
            st.markdown(f"**Expected Impact:** {action['expected_impact']}")
            st.markdown(f"**Recommended Deadline:** {action['recommended_deadline']}")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>"
    "📊 CPSC Enterprise Redundancy Analysis Dashboard v1.0<br>"
    f"Powered by Sentence-BERT (all-MiniLM-L6-v2) | Report Date: {report['metadata']['analysis_date'][:10]}<br>"
    f"Processing Time: {report['metadata']['performance']['processing_time_seconds']:.2f}s"
    "</div>",
    unsafe_allow_html=True
)
