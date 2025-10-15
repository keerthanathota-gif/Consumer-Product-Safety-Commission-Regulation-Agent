#!/bin/bash

echo "🕸️ CPSC Knowledge Graph - Launcher"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -f "simple_knowledge_graph.py" ]; then
    echo "❌ Error: Please run this script from the cpsc_knowledge_graph directory"
    echo "   cd /workspace/cpsc_knowledge_graph"
    exit 1
fi

# Function to show menu
show_menu() {
    echo "Choose an option:"
    echo ""
    echo "1. 🧪 Test Installation"
    echo "2. 📊 Run Knowledge Graph Analysis"
    echo "3. 🌐 Launch Interactive Dashboard"
    echo "4. 🔍 Quick Search Test"
    echo "5. 📈 Export Data"
    echo "6. ❓ Show Help"
    echo "7. 🚪 Exit"
    echo ""
    echo -n "Enter your choice (1-7): "
}

# Function to test installation
test_installation() {
    echo "🧪 Testing installation..."
    python3 test_installation.py
}

# Function to run analysis
run_analysis() {
    echo "📊 Running knowledge graph analysis..."
    python3 simple_knowledge_graph.py
}

# Function to launch dashboard
launch_dashboard() {
    echo "🌐 Launching interactive dashboard..."
    echo "The dashboard will be available at: http://localhost:8501"
    echo "Press Ctrl+C to stop the dashboard"
    echo ""
    ./launch_dashboard.sh
}

# Function to run quick search
quick_search() {
    echo "🔍 Quick search test..."
    echo -n "Enter search term: "
    read search_term
    
    if [ -z "$search_term" ]; then
        search_term="safety"
    fi
    
    echo "Searching for: $search_term"
    python3 -c "
from simple_knowledge_graph import SimpleKnowledgeGraph
graph = SimpleKnowledgeGraph()
graph.build_graph()
results = graph.search('$search_term', limit=5)
print(f'Found {len(results)} results:')
for i, result in enumerate(results, 1):
    node = result['node']
    print(f'{i}. {node.name}: {node.metadata.get(\"subject\", \"No subject\")}')
"
}

# Function to export data
export_data() {
    echo "📈 Exporting data..."
    python3 -c "
from simple_knowledge_graph import SimpleKnowledgeGraph
import json

graph = SimpleKnowledgeGraph()
graph.build_graph()

# Export to JSON
json_data = graph.export_to_json()
with open('cpsc_knowledge_graph.json', 'w') as f:
    f.write(json_data)

print('✅ Data exported to cpsc_knowledge_graph.json')
print(f'File size: {len(json_data)} characters')
"
}

# Function to show help
show_help() {
    echo "❓ Help - CPSC Knowledge Graph"
    echo "=============================="
    echo ""
    echo "This system analyzes CPSC regulation data using a knowledge graph approach."
    echo ""
    echo "Available options:"
    echo "1. Test Installation - Verify all components are working"
    echo "2. Run Analysis - Build and analyze the knowledge graph"
    echo "3. Launch Dashboard - Start interactive web interface"
    echo "4. Quick Search - Search for specific terms"
    echo "5. Export Data - Export graph data to JSON"
    echo "6. Show Help - Display this help message"
    echo "7. Exit - Close the launcher"
    echo ""
    echo "For detailed instructions, see:"
    echo "- README.md"
    echo "- EXECUTION_GUIDE.md"
    echo "- QUICK_START.md"
    echo ""
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            test_installation
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        2)
            run_analysis
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        3)
            launch_dashboard
            ;;
        4)
            quick_search
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        5)
            export_data
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        6)
            show_help
            echo ""
            echo "Press Enter to continue..."
            read
            ;;
        7)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-7."
            echo ""
            ;;
    esac
done