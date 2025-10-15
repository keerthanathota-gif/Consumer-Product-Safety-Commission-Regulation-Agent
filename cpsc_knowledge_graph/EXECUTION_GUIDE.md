# 🚀 Execution Guide - CPSC Knowledge Graph

## 📋 Quick Start (5 minutes)

### Step 1: Navigate to the Project
```bash
cd /workspace/cpsc_knowledge_graph
```

### Step 2: Test Installation
```bash
python3 test_installation.py
```

### Step 3: Run the Knowledge Graph
```bash
# Option A: Quick test
python3 simple_knowledge_graph.py

# Option B: Interactive dashboard
./launch_dashboard.sh
```

## 🔧 Detailed Execution Instructions

### Prerequisites Check
Before running, ensure you have:
- ✅ Python 3.8 or higher
- ✅ All dependencies installed
- ✅ Database file available
- ✅ Proper permissions

### Method 1: Command Line Analysis

#### Basic Analysis
```bash
# Navigate to project directory
cd /workspace/cpsc_knowledge_graph

# Run the knowledge graph analysis
python3 simple_knowledge_graph.py
```

**Expected Output:**
```
🚀 Simplified CPSC Regulation Knowledge Graph
==================================================

📊 Graph Statistics:
   Nodes: 713
   Edges: 59719
   Node types: {'chapter': 1, 'part': 135, 'section': 577}
   Relationship types: {'contains': 712, 'similar_to': 35813, 'compliance_related': 23194}
   Degree stats: min=1, max=479, avg=167.5
   Compliance sections: 408

🔍 Search Test:
   Found 3 results for 'safety'
   1. § 1236.2: Requirements for infant sleep products.
   2. § 1500.90: Requirements for exclusions from lead limits...
   3. § 1250.2: Requirements for toy safety.

✅ All tests completed successfully!
```

#### Custom Analysis
```python
# Create a custom analysis script
python3 -c "
from simple_knowledge_graph import SimpleKnowledgeGraph

# Build graph
graph = SimpleKnowledgeGraph()
graph.build_graph()

# Custom analysis
print('📊 Custom Analysis Results:')
print(f'Total regulations: {len([n for n in graph.nodes.values() if n.node_type == \"section\"])}')
print(f'Compliance sections: {len([n for n in graph.nodes.values() if n.metadata.get(\"has_compliance\", False)])}')

# Search for specific terms
safety_results = graph.search('safety', limit=5)
print(f'\\nSafety-related regulations:')
for i, result in enumerate(safety_results, 1):
    print(f'{i}. {result[\"node\"].name}: {result[\"node\"].metadata.get(\"subject\", \"No subject\")}')
"
```

### Method 2: Interactive Dashboard

#### Launch Dashboard
```bash
# Make sure script is executable
chmod +x launch_dashboard.sh

# Launch the dashboard
./launch_dashboard.sh
```

#### Access Dashboard
1. Open your web browser
2. Navigate to: `http://localhost:8501`
3. Wait for the dashboard to load (may take 30-60 seconds)

#### Dashboard Features
- **Overview Tab**: Graph statistics and metrics
- **Search & Explore Tab**: Search regulations by keywords
- **Network Analysis Tab**: Interactive network visualization
- **Compliance Analysis Tab**: Compliance patterns and analysis
- **Data Insights Tab**: Statistical analysis and reporting

#### Stop Dashboard
- Press `Ctrl+C` in the terminal
- Or close the terminal window

### Method 3: Programmatic Usage

#### Python Script Example
```python
# Create analysis_script.py
from simple_knowledge_graph import SimpleKnowledgeGraph
import json

def analyze_regulations():
    # Initialize and build graph
    graph = SimpleKnowledgeGraph()
    graph.build_graph()
    
    # Get statistics
    stats = graph.get_graph_statistics()
    print(f"Graph has {stats['total_nodes']} nodes and {stats['total_edges']} edges")
    
    # Search for specific terms
    search_terms = ['safety', 'hazard', 'compliance', 'testing']
    
    for term in search_terms:
        results = graph.search(term, limit=5)
        print(f"\n{term.upper()} regulations:")
        for i, result in enumerate(results, 1):
            node = result['node']
            print(f"  {i}. {node.name}: {node.metadata.get('subject', 'No subject')}")
    
    # Export data
    json_data = graph.export_to_json()
    with open('regulation_analysis.json', 'w') as f:
        f.write(json_data)
    print(f"\nData exported to regulation_analysis.json")

if __name__ == "__main__":
    analyze_regulations()
```

#### Run the Script
```bash
python3 analysis_script.py
```

## 📊 Understanding the Output

### Graph Statistics
- **Nodes**: Total entities (chapters, parts, sections)
- **Edges**: Total relationships between entities
- **Node Types**: Distribution of different entity types
- **Relationship Types**: Types of connections (hierarchical, semantic, compliance)
- **Degree Stats**: Network connectivity metrics

### Search Results
- **Score**: Relevance score (0-1, higher is better)
- **Match Type**: How the match was found (text_match, semantic, etc.)
- **Node**: The regulation section that matched

### Compliance Analysis
- **Compliance Score**: Number of compliance-related keywords found
- **Compliance Sections**: Sections containing compliance language
- **Compliance Clusters**: Groups of related compliance requirements

## 🔍 Advanced Usage

### Custom Search Queries
```python
# Search for specific compliance terms
compliance_results = graph.search("shall must required prohibited", limit=10)

# Search for safety-related content
safety_results = graph.search("safety hazard risk injury", limit=10)

# Search for testing procedures
testing_results = graph.search("test testing procedure method", limit=10)
```

### Export Data
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
            'compliance_score': node.metadata.get('compliance_score', 0),
            'word_count': node.metadata.get('word_count', 0)
        })

df = pd.DataFrame(compliance_data)
df.to_csv('compliance_analysis.csv', index=False)
```

### Network Analysis
```python
# Find highly connected regulations
highly_connected = [node for node in graph.nodes.values() 
                   if node.node_type == 'section' and node.degree > 100]

print("Highly connected regulations:")
for node in highly_connected:
    print(f"{node.name}: {node.degree} connections")

# Find regulation clusters
clusters = graph.find_clusters(min_cluster_size=5)
print(f"Found {len(clusters)} clusters")
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Database not found"
```bash
# Check if database exists
ls -la ../regulations.db

# If not found, update the path in the code
# Edit simple_knowledge_graph.py, line with db_path
```

#### 2. "Permission denied" for launch_dashboard.sh
```bash
# Fix permissions
chmod +x launch_dashboard.sh
```

#### 3. "Module not found" errors
```bash
# Install missing dependencies
pip install -r requirements.txt
```

#### 4. Dashboard won't start
```bash
# Try a different port
streamlit run analysis_dashboard.py --server.port 8502

# Check if port is in use
lsof -i :8501
```

#### 5. Memory issues
```bash
# Close other applications
# Use smaller datasets
# Increase system memory
```

### Performance Optimization

#### For Large Datasets
```python
# Reduce similarity threshold
threshold = 0.5  # Instead of 0.3

# Limit network visualization
max_nodes = 50  # Instead of 100

# Use smaller batch sizes
batch_size = 50  # Instead of 100
```

#### For Better Performance
```bash
# Use SSD storage
# Increase RAM
# Close unnecessary applications
# Use Python 3.9+ for better performance
```

## 📈 Expected Performance

### Graph Construction
- **Small dataset** (< 100 sections): < 5 seconds
- **Medium dataset** (100-500 sections): 5-30 seconds
- **Large dataset** (500+ sections): 30-60 seconds

### Search Performance
- **Simple queries**: < 1 second
- **Complex queries**: 1-5 seconds
- **Large result sets**: 5-10 seconds

### Dashboard Performance
- **Initial load**: 30-60 seconds
- **Navigation**: < 1 second
- **Visualization**: 1-5 seconds

## 🎯 Use Cases

### 1. Regulatory Research
- Find all regulations related to a specific topic
- Identify compliance requirements
- Analyze regulatory structure

### 2. Compliance Analysis
- Track compliance language across regulations
- Identify compliance gaps
- Generate compliance reports

### 3. Knowledge Discovery
- Discover hidden relationships
- Find regulatory patterns
- Analyze regulatory evolution

### 4. Training and Education
- Interactive regulation exploration
- Compliance training materials
- Regulatory navigation tools

## 📞 Getting Help

### Self-Help
1. **Check the logs** for error messages
2. **Read the documentation** in README.md
3. **Run the test script** to identify issues
4. **Check system requirements**

### Common Solutions
- **Install dependencies**: `pip install -r requirements.txt`
- **Check database path**: Ensure regulations.db is accessible
- **Verify Python version**: Must be 3.8 or higher
- **Check permissions**: Make scripts executable

### Debug Mode
```bash
# Run with verbose output
python3 -u simple_knowledge_graph.py 2>&1 | tee debug.log

# Check the log file
cat debug.log
```

## ✅ Success Checklist

- [ ] Installation test passed
- [ ] Database accessible
- [ ] Dependencies installed
- [ ] Scripts executable
- [ ] Graph builds successfully
- [ ] Search works
- [ ] Dashboard accessible
- [ ] Export functions work

Once all items are checked, you're ready to use the CPSC Knowledge Graph! 🎉

---

**Need help?** Check the troubleshooting section or run `python3 test_installation.py` to diagnose issues.