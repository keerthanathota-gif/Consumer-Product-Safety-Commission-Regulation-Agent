# 🕸️ CPSC Regulation Knowledge Graph

A comprehensive knowledge graph system for analyzing Consumer Product Safety Commission (CPSC) regulations with interactive visualization and compliance analysis capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- CPSC regulation database (`regulations.db`)

### Installation

1. **Clone or download this repository**
```bash
git clone <repository-url>
cd cpsc_knowledge_graph
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure database is available**
Make sure `regulations.db` is in the parent directory or update the path in the code.

### Usage

#### Option 1: Quick Test
```bash
python3 simple_knowledge_graph.py
```

#### Option 2: Interactive Dashboard
```bash
chmod +x launch_dashboard.sh
./launch_dashboard.sh
```
Then open http://localhost:8501 in your browser.

#### Option 3: Programmatic Access
```python
from simple_knowledge_graph import SimpleKnowledgeGraph

# Build the knowledge graph
graph = SimpleKnowledgeGraph()
graph.build_graph()

# Search regulations
results = graph.search("safety requirements", limit=10)

# Get related regulations
related = graph.get_related_nodes("section_1", limit=5)

# Export data
json_data = graph.export_to_json()
```

## 📊 Features

### Knowledge Graph Analysis
- **713 nodes**: 1 chapter, 135 parts, 577 sections
- **59,719 edges**: Hierarchical, semantic, and compliance relationships
- **408 compliance sections** identified and analyzed
- **70.7% compliance coverage** across regulations

### Interactive Dashboard
- **Search & Explore**: Find regulations by keywords
- **Network Analysis**: Visualize regulation relationships
- **Compliance Analysis**: Analyze compliance patterns
- **Data Insights**: Statistical analysis and reporting
- **Export Capabilities**: JSON and CSV export

### Key Insights
- **Highly Interconnected**: Average of 167.5 relationships per regulation
- **Compliance-Focused**: Strong compliance language patterns
- **Semantic Clustering**: Related regulations cluster together
- **Well-Organized**: Clear hierarchical structure

## 📁 Project Structure

```
cpsc_knowledge_graph/
├── simple_knowledge_graph.py    # Core knowledge graph implementation
├── analysis_dashboard.py        # Interactive Streamlit dashboard
├── launch_dashboard.sh         # Dashboard launcher script
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── README.md                   # This file
├── QUICK_START.md             # Detailed quick start guide
└── ANALYSIS_REPORT.md         # Comprehensive analysis report
```

## 🔧 Configuration

### Database Path
Update the database path in `simple_knowledge_graph.py` if your database is in a different location:

```python
graph = SimpleKnowledgeGraph(db_path="/path/to/your/regulations.db")
```

### Dashboard Port
Change the dashboard port in `launch_dashboard.sh`:

```bash
streamlit run analysis_dashboard.py --server.port 8502 --server.address 0.0.0.0
```

## 📈 Analysis Capabilities

### 1. Regulatory Structure Analysis
- Hierarchical relationship mapping
- Network connectivity analysis
- Node degree distribution
- Clustering analysis

### 2. Compliance Analysis
- Compliance requirement detection
- Compliance score calculation
- Compliance pattern analysis
- Compliance clustering

### 3. Semantic Analysis
- Text similarity analysis
- Keyword extraction and analysis
- Content-based relationship discovery
- Semantic clustering

### 4. Interactive Exploration
- Real-time search functionality
- Interactive network visualization
- Section-by-section exploration
- Relationship analysis

## 🎯 Use Cases

### Regulatory Analysis
- Understand regulatory structure and relationships
- Identify compliance requirements and patterns
- Find related regulations and cross-references
- Analyze regulatory complexity and density

### Compliance Management
- Track compliance requirements across regulations
- Identify compliance gaps and overlaps
- Monitor compliance language patterns
- Generate compliance reports

### Knowledge Discovery
- Discover hidden relationships between regulations
- Identify regulatory clusters and patterns
- Analyze regulatory evolution and changes
- Generate regulatory insights

## 📊 Performance

- **Graph Construction**: < 30 seconds for 577 sections
- **Search Response**: < 1 second for most queries
- **Memory Usage**: Efficient for large datasets
- **Scalability**: Handles 1000+ regulations easily

## 🔍 API Reference

### SimpleKnowledgeGraph Class

#### Methods
- `build_graph()`: Build the knowledge graph from database
- `search(query, limit=10)`: Search regulations by keywords
- `get_related_nodes(node_id, limit=10)`: Get related regulations
- `get_graph_statistics()`: Get comprehensive graph statistics
- `export_to_json()`: Export graph to JSON format
- `find_clusters(min_cluster_size=3)`: Find regulation clusters

#### Properties
- `nodes`: Dictionary of graph nodes
- `edges`: Dictionary of graph edges
- `db_path`: Path to the regulation database

## 🚀 Advanced Usage

### Custom Analysis
```python
from simple_knowledge_graph import SimpleKnowledgeGraph

graph = SimpleKnowledgeGraph()
graph.build_graph()

# Custom analysis
compliance_sections = [node for node in graph.nodes.values() 
                      if node.node_type == 'section' and 
                      node.metadata.get('has_compliance', False)]

print(f"Found {len(compliance_sections)} compliance sections")
```

### Export and Integration
```python
# Export to JSON
json_data = graph.export_to_json()
with open('cpsc_graph.json', 'w') as f:
    f.write(json_data)

# Export compliance analysis
import pandas as pd
compliance_data = []
for node in graph.nodes.values():
    if node.node_type == 'section' and node.metadata.get('has_compliance', False):
        compliance_data.append({
            'section': node.name,
            'subject': node.metadata.get('subject', ''),
            'compliance_score': node.metadata.get('compliance_score', 0)
        })

df = pd.DataFrame(compliance_data)
df.to_csv('compliance_analysis.csv', index=False)
```

## 🐛 Troubleshooting

### Common Issues

1. **Database not found**
   - Ensure `regulations.db` is in the correct location
   - Update the database path in the code

2. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.8+ is installed

3. **Dashboard won't start**
   - Check if port 8501 is available
   - Try a different port: `streamlit run analysis_dashboard.py --server.port 8502`

4. **Memory issues**
   - Reduce the number of nodes in network visualization
   - Use smaller batch sizes for large datasets

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For questions or issues, please refer to the documentation or create an issue in the repository.

---

**Built with ❤️ for CPSC Regulation Analysis**