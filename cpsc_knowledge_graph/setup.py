#!/usr/bin/env python3
"""
Setup script for CPSC Knowledge Graph
"""

from setuptools import setup, find_packages

setup(
    name="cpsc-knowledge-graph",
    version="1.0.0",
    description="Knowledge Graph for CPSC Regulations Analysis",
    author="AI Assistant",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "streamlit>=1.28.0",
        "plotly>=5.14.0",
        "networkx>=3.0",
        "numpy>=1.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "cpsc-graph=simple_knowledge_graph:main",
        ],
    },
)