# Graphiti Integration for Regulation Section Relationships

This project integrates Graphiti (and NetworkX as fallback) to visualize relationships between regulation sections in the CPSC database. The integration provides interactive graph visualizations that help identify semantic similarities, cross-references, and hierarchical relationships between regulation sections.

## 🚀 Quick Start

### 1. Launch the Integrated Dashboard
```bash
python3 launch_dashboard.py integrated
```

### 2. Launch Graph-Only Dashboard
```bash
python3 launch_dashboard.py graph
```

### 3. Launch Original Redundancy Dashboard
```bash
python3 launch_dashboard.py original
```

## 📊 Features

### Graph Visualization
- **Interactive Network Graph**: Visual representation of regulation sections as nodes with relationships as edges
- **Multiple Relationship Types**:
  - Semantic Similarity: Sections with similar content (TF-IDF + cosine similarity)
  - Cross-References: Sections that reference other sections
  - Hierarchical: Sections within the same regulatory part
- **Dynamic Filtering**: Filter by relationship type, similarity threshold, and number of nodes
- **Color-Coded Nodes**: Different colors for different relationship types

### Analysis Capabilities
- **Graph Statistics**: Node count, edge count, density, clustering coefficient
- **Relationship Analysis**: Distribution of relationship types and strengths
- **Part Distribution**: Analysis of sections across different regulatory parts
- **Interactive Exploration**: Hover for details, zoom, pan, and select nodes

### Integration Features
- **Combined Dashboard**: Merges redundancy analysis with relationship visualization
- **Real-time Filtering**: Adjust similarity thresholds and relationship types
- **Export Capabilities**: Save graph data and visualizations
- **Performance Optimization**: Handles large graphs with intelligent sampling

## 🏗️ Architecture

### Core Components

1. **`graphiti_integration.py`**: Core graph building and analysis engine
   - `RegulationGraphBuilder`: Main class for building regulation graphs
   - `SectionNode`: Data structure for regulation sections
   - `SectionRelationship`: Data structure for relationships between sections

2. **`integrated_dashboard.py`**: Comprehensive dashboard combining all features
   - Redundancy analysis integration
   - Graph visualization
   - Combined analysis views

3. **`graph_dashboard.py`**: Graph-focused dashboard
   - Network visualization
   - Relationship analysis
   - Interactive controls

4. **`launch_dashboard.py`**: Easy launcher script
   - Package dependency checking
   - Multiple dashboard options
   - Error handling

### Data Flow

```
Regulations Database (SQLite)
    ↓
RegulationGraphBuilder.load_sections_from_db()
    ↓
Detect Relationships:
    - Semantic Similarity (TF-IDF + Cosine Similarity)
    - Cross-References (Regex Pattern Matching)
    - Hierarchical (Same Part Grouping)
    ↓
NetworkX Graph Construction
    ↓
Plotly Interactive Visualization
    ↓
Streamlit Dashboard
```

## 🔧 Configuration

### Similarity Thresholds
- **Semantic Similarity**: Default 0.3 (adjustable 0.0-1.0)
- **Cross-References**: Fixed at 1.0 (exact matches)
- **Hierarchical**: Fixed at 0.8 (same part)

### Graph Parameters
- **Max Nodes**: Default 50 (adjustable 10-200)
- **Layout Algorithm**: Spring layout with k=1, iterations=50
- **Node Size**: Fixed at 20px
- **Edge Width**: 0.5px

### Color Scheme
- **Semantic Similarity**: Orange (#ff7f0e)
- **Cross-References**: Green (#2ca02c)
- **Same Part**: Blue (#1f77b4)
- **Unconnected**: Light Gray (#d3d3d3)

## 📈 Performance

### Graph Statistics (Current Dataset)
- **Total Sections**: 564
- **Total Relationships**: 6,603
- **Graph Density**: 0.042
- **Connected Components**: 1
- **Average Clustering**: 0.632
- **Average Degree**: 23.4

### Relationship Distribution
- **Semantic Similarity**: 4,033 relationships
- **Same Part**: 3,665 relationships
- **Cross-References**: 0 relationships (limited in current dataset)

## 🛠️ Technical Details

### Dependencies
```
numpy>=1.21.0
scikit-learn>=1.0.0
lxml>=4.9.0
networkx>=2.6.0
matplotlib>=3.5.0
plotly>=5.0.0
pandas>=1.3.0
graphiti>=0.1.0
streamlit
```

### Database Schema
The integration works with the existing regulation database schema:
- `chapters`: Top-level regulatory chapters
- `subchapters`: Regulatory subchapters
- `parts`: Regulatory parts within subchapters
- `sections`: Individual regulation sections

### Graph Data Export
Graph data is exported in JSON format with the following structure:
```json
{
  "nodes": [
    {
      "id": "section_id",
      "section_number": "§ 1000.1",
      "subject": "Section subject",
      "part_heading": "Part heading",
      "subchapter_name": "Subchapter name",
      "chapter_name": "Chapter name",
      "text_length": 1234
    }
  ],
  "edges": [
    {
      "source": "source_section_id",
      "target": "target_section_id",
      "relationship_type": "semantic_similarity",
      "strength": 0.75,
      "description": "Semantic similarity: 0.750"
    }
  ],
  "metadata": {
    "total_sections": 564,
    "total_relationships": 6603,
    "graph_statistics": {...}
  }
}
```

## 🎯 Usage Examples

### 1. Explore High-Similarity Relationships
```python
# In the dashboard, set similarity threshold to 0.7
# Filter to show only semantic similarity relationships
# This will highlight the most similar regulation sections
```

### 2. Analyze Part Structure
```python
# Filter to show only "same_part" relationships
# This visualizes the hierarchical structure within regulatory parts
```

### 3. Find Cross-References
```python
# Filter to show only "cross_reference" relationships
# This identifies sections that explicitly reference other sections
```

### 4. Combined Analysis
```python
# Use the integrated dashboard to compare:
# - Redundancy analysis results
# - Relationship graph patterns
# - Similarity score distributions
```

## 🔍 Troubleshooting

### Common Issues

1. **Graph Not Loading**
   - Ensure `regulations.db` exists
   - Check database schema matches expected format
   - Verify all required packages are installed

2. **Performance Issues**
   - Reduce max_nodes parameter
   - Increase similarity threshold
   - Use relationship type filtering

3. **Visualization Problems**
   - Check browser compatibility with Plotly
   - Ensure sufficient memory for large graphs
   - Try different layout algorithms

### Debug Mode
```python
# Enable debug output in graphiti_integration.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Future Enhancements

### Planned Features
- **Advanced Layout Algorithms**: Force-directed, hierarchical layouts
- **Community Detection**: Identify clusters of related sections
- **Temporal Analysis**: Track relationship changes over time
- **Export Options**: PNG, SVG, PDF export
- **Real-time Updates**: Live graph updates as data changes

### Potential Integrations
- **Machine Learning**: Enhanced similarity detection
- **Natural Language Processing**: Better text analysis
- **Database Integration**: Direct database querying
- **API Endpoints**: RESTful API for graph data

## 📚 References

- [Graphiti Documentation](https://github.com/getzep/graphiti)
- [NetworkX Documentation](https://networkx.org/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🤝 Contributing

To contribute to this integration:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This integration is part of the CPSC regulation analysis project and follows the same licensing terms.