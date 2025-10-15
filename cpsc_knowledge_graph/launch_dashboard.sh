#!/bin/bash

echo "🚀 Launching CPSC Regulation Knowledge Graph Dashboard"
echo "=================================================="
echo ""
echo "The dashboard will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Change to the correct directory
cd /workspace/graphiti_integration

# Launch Streamlit dashboard
streamlit run analysis_dashboard.py --server.port 8501 --server.address 0.0.0.0