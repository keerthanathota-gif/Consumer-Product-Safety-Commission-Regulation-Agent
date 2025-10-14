# Graphiti Integration for CPSC Regulation Agent

## Overview

This integration adds graph-based knowledge visualization capabilities to the Consumer Product Safety Commission Regulation Agent using [Graphiti](https://github.com/getzep/graphiti), a framework for building real-time knowledge graphs for AI agents.

## Features

### 🕸️ Knowledge Graph Construction
- **Regulatory Entity Mapping**: Maps CPSC regulations to graph entities
- **Relationship Discovery**: Identifies semantic and structural relationships between regulations
- **Temporal Awareness**: Tracks regulation evolution and dependencies over time
- **Hierarchical Structure**: Preserves chapter → subchapter → part → section hierarchy

### 📊 Graph Visualization Dashboard
- **Interactive Network Visualization**: Explore regulation relationships visually
- **Semantic Clustering**: Group related regulations by topic and similarity
- **Relationship Analysis**: Identify regulatory dependencies and overlaps
- **Search and Filter**: Find specific regulations and their connections

### 🔍 Advanced Analytics
- **Redundancy Detection**: Enhanced redundancy analysis using graph traversal
- **Impact Analysis**: Understand how changes to one regulation affect others
- **Compliance Mapping**: Visualize compliance requirements and dependencies
- **Knowledge Discovery**: Discover hidden patterns in regulatory structure

## Architecture

```
graphiti_integration/
├── requirements.txt              # Dependencies
├── README.md                    # This file
├── config.py                    # Configuration settings
├── entities.py                  # Pydantic models for regulatory entities
├── graph_builder.py             # Core graph construction logic
├── data_loader.py               # Data loading from SQLite/JSON
├── relationship_analyzer.py     # Relationship discovery algorithms
├── graph_dashboard.py           # Streamlit dashboard for visualization
├── integration_api.py           # API for integration with existing system
└── examples/
    ├── basic_usage.py           # Basic usage examples
    └── advanced_queries.py      # Advanced graph queries
```

## Quick Start

### 1. Install Dependencies

```bash
cd graphiti_integration
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
# Create .env file with your configuration
cp .env.example .env
# Edit .env with your settings
```

### 3. Build the Knowledge Graph

```bash
python graph_builder.py
```

### 4. Launch Visualization Dashboard

```bash
streamlit run graph_dashboard.py
```

## Integration with Existing System

The Graphiti integration is designed to work alongside the existing CPSC Regulation Agent without modifying the original codebase:

- **Data Source**: Reads from existing `regulations.db` SQLite database
- **Analysis Enhancement**: Enhances existing redundancy analysis with graph insights
- **Dashboard Integration**: Provides additional visualization layer
- **API Access**: Offers programmatic access to graph data

## Key Components

### Entity Models (`entities.py`)
- `RegulationSection`: Core regulatory section entity
- `RegulationPart`: Part-level entity with hierarchical relationships
- `RegulationChapter`: Chapter-level entity for high-level organization
- `ComplianceRequirement`: Extracted compliance requirements
- `SafetyStandard`: Safety standards and their relationships

### Graph Builder (`graph_builder.py`)
- Loads data from existing SQLite database
- Creates entities and relationships
- Builds temporal knowledge graph
- Handles incremental updates

### Relationship Analyzer (`relationship_analyzer.py`)
- Semantic similarity analysis
- Hierarchical relationship mapping
- Cross-reference detection
- Dependency analysis

### Visualization Dashboard (`graph_dashboard.py`)
- Interactive network visualization
- Relationship exploration
- Search and filtering
- Export capabilities

## Use Cases

### 1. Regulatory Compliance Analysis
- Map compliance requirements across regulations
- Identify conflicting or overlapping requirements
- Track regulatory changes and their impacts

### 2. Knowledge Discovery
- Discover hidden relationships between regulations
- Identify regulatory patterns and trends
- Find gaps in regulatory coverage

### 3. Enhanced Redundancy Analysis
- Use graph traversal to find indirect redundancies
- Identify regulatory clusters with similar content
- Analyze regulatory evolution over time

### 4. Interactive Exploration
- Visualize regulatory structure
- Explore relationships interactively
- Search and filter regulations by various criteria

## Configuration

The integration uses a configuration file (`config.py`) to manage:

- Database connections
- Graph database settings
- Visualization parameters
- Analysis thresholds
- API endpoints

## API Integration

The `integration_api.py` provides a clean interface for integrating graph capabilities with the existing system:

```python
from integration_api import GraphitiIntegration

# Initialize integration
graphiti = GraphitiIntegration()

# Get related regulations
related = graphiti.get_related_regulations("§ 1000.1")

# Find regulatory clusters
clusters = graphiti.find_regulatory_clusters()

# Analyze relationships
relationships = graphiti.analyze_relationships("§ 1000.1", "§ 1000.2")
```

## Performance Considerations

- **Incremental Updates**: Only processes new or changed regulations
- **Caching**: Implements intelligent caching for frequently accessed data
- **Parallel Processing**: Uses parallel processing for large-scale analysis
- **Memory Management**: Efficient memory usage for large graphs

## Future Enhancements

- **Real-time Updates**: Live updates as regulations change
- **Machine Learning**: ML-powered relationship discovery
- **Natural Language Queries**: Query regulations using natural language
- **Export Capabilities**: Export graphs in various formats
- **API Integration**: REST API for external access

## Contributing

This integration is designed to be extensible. Key extension points:

1. **Custom Entity Types**: Add new entity types in `entities.py`
2. **Relationship Algorithms**: Implement new relationship discovery in `relationship_analyzer.py`
3. **Visualization Components**: Add new visualization types in `graph_dashboard.py`
4. **Analysis Methods**: Extend analysis capabilities in `integration_api.py`

## License

This integration follows the same license as the main CPSC Regulation Agent project.

## Support

For issues specific to this integration, please refer to the main project documentation or create an issue in the project repository.