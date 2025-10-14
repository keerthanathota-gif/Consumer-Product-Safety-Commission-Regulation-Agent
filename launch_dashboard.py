#!/usr/bin/env python3
"""
Dashboard Launcher Script
========================

This script provides an easy way to launch different dashboard views.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'networkx', 'matplotlib', 'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Packages installed successfully!")

def launch_dashboard(dashboard_type="integrated"):
    """Launch the specified dashboard."""
    dashboard_files = {
        "integrated": "integrated_dashboard.py",
        "graph": "graph_dashboard.py", 
        "original": "dashboard.py"
    }
    
    if dashboard_type not in dashboard_files:
        print(f"Unknown dashboard type: {dashboard_type}")
        print(f"Available options: {', '.join(dashboard_files.keys())}")
        return
    
    dashboard_file = dashboard_files[dashboard_type]
    
    if not Path(dashboard_file).exists():
        print(f"Dashboard file not found: {dashboard_file}")
        return
    
    print(f"Launching {dashboard_type} dashboard...")
    print(f"Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run(["streamlit", "run", dashboard_file], check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error launching dashboard: {e}")

def main():
    """Main launcher function."""
    print("CPSC Regulation Analysis Dashboard Launcher")
    print("=" * 50)
    
    # Check requirements
    check_requirements()
    
    # Get dashboard type from command line or prompt
    if len(sys.argv) > 1:
        dashboard_type = sys.argv[1]
    else:
        print("\nAvailable dashboards:")
        print("1. integrated - Combined redundancy analysis and relationship graph")
        print("2. graph - Regulation relationship graph only")
        print("3. original - Original redundancy analysis only")
        
        choice = input("\nSelect dashboard (1-3) or type name: ").strip()
        
        dashboard_map = {
            "1": "integrated",
            "2": "graph", 
            "3": "original",
            "integrated": "integrated",
            "graph": "graph",
            "original": "original"
        }
        
        dashboard_type = dashboard_map.get(choice, "integrated")
    
    # Launch dashboard
    launch_dashboard(dashboard_type)

if __name__ == "__main__":
    main()