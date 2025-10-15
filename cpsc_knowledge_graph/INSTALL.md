# 📦 Installation Guide - CPSC Knowledge Graph

## 🚀 Quick Installation

### Step 1: Prerequisites
- **Python 3.8 or higher** (check with `python3 --version`)
- **pip** package manager
- **CPSC regulation database** (`regulations.db`)

### Step 2: Download/Clone
```bash
# If using git
git clone <repository-url>
cd cpsc_knowledge_graph

# Or download and extract the folder
# Then navigate to the cpsc_knowledge_graph directory
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually
pip install pandas scikit-learn pydantic python-dotenv streamlit plotly networkx
```

### Step 4: Verify Installation
```bash
# Test the installation
python3 simple_knowledge_graph.py
```

## 🔧 Detailed Installation

### Option 1: Using pip (Recommended)
```bash
# Navigate to the project directory
cd cpsc_knowledge_graph

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x launch_dashboard.sh
```

### Option 2: Using conda
```bash
# Create a new conda environment
conda create -n cpsc-graph python=3.9

# Activate the environment
conda activate cpsc-graph

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using virtual environment
```bash
# Create virtual environment
python3 -m venv cpsc-graph-env

# Activate virtual environment
# On Linux/Mac:
source cpsc-graph-env/bin/activate
# On Windows:
# cpsc-graph-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🗄️ Database Setup

### Option 1: Use Existing Database
If you have `regulations.db` in the parent directory:
```bash
# No additional setup needed
# The system will automatically find the database
```

### Option 2: Specify Database Path
If your database is in a different location:
```python
# Edit simple_knowledge_graph.py
graph = SimpleKnowledgeGraph(db_path="/path/to/your/regulations.db")
```

### Option 3: Create Test Database
If you don't have a database, you can create a simple test one:
```python
import sqlite3

# Create a simple test database
conn = sqlite3.connect('test_regulations.db')
cursor = conn.cursor()

# Create tables
cursor.execute('''
    CREATE TABLE chapters (
        chapter_id INTEGER PRIMARY KEY,
        chapter_name TEXT
    )
''')

cursor.execute('''
    CREATE TABLE subchapters (
        subchapter_id INTEGER PRIMARY KEY,
        chapter_id INTEGER,
        subchapter_name TEXT
    )
''')

cursor.execute('''
    CREATE TABLE parts (
        part_id INTEGER PRIMARY KEY,
        subchapter_id INTEGER,
        heading TEXT
    )
''')

cursor.execute('''
    CREATE TABLE sections (
        section_id INTEGER PRIMARY KEY,
        part_id INTEGER,
        section_number TEXT,
        subject TEXT,
        text TEXT
    )
''')

# Insert test data
cursor.execute("INSERT INTO chapters VALUES (1, 'Test Chapter')")
cursor.execute("INSERT INTO subchapters VALUES (1, 1, 'Test Subchapter')")
cursor.execute("INSERT INTO parts VALUES (1, 1, 'Test Part')")
cursor.execute("INSERT INTO sections VALUES (1, 1, '§ 1.1', 'Test Section', 'This is a test regulation.')")

conn.commit()
conn.close()
```

## 🧪 Testing Installation

### Test 1: Basic Functionality
```bash
python3 simple_knowledge_graph.py
```
Expected output:
```
🚀 Simplified CPSC Regulation Knowledge Graph
==================================================
📊 Graph Statistics:
   Nodes: 713
   Edges: 59719
   ...
✅ All tests completed successfully!
```

### Test 2: Dashboard
```bash
./launch_dashboard.sh
```
Expected output:
```
🚀 Launching CPSC Regulation Knowledge Graph Dashboard
==================================================

The dashboard will be available at: http://localhost:8501
Press Ctrl+C to stop the dashboard
```

### Test 3: Python Import
```python
python3 -c "from simple_knowledge_graph import SimpleKnowledgeGraph; print('✅ Import successful')"
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "ModuleNotFoundError: No module named 'pandas'"
```bash
# Solution: Install missing dependencies
pip install pandas scikit-learn pydantic python-dotenv streamlit plotly networkx
```

#### 2. "Permission denied" when running launch_dashboard.sh
```bash
# Solution: Make script executable
chmod +x launch_dashboard.sh
```

#### 3. "Database not found" error
```bash
# Solution: Check database path
ls -la ../regulations.db
# Or update the path in the code
```

#### 4. "Port 8501 already in use"
```bash
# Solution: Use a different port
streamlit run analysis_dashboard.py --server.port 8502
```

#### 5. "Python version not supported"
```bash
# Solution: Check Python version
python3 --version
# Should be 3.8 or higher
# If not, install a newer version
```

### Performance Issues

#### Memory Issues
- Reduce the number of nodes in network visualization
- Use smaller batch sizes for large datasets
- Close other applications to free up memory

#### Slow Performance
- Ensure you have sufficient RAM (8GB+ recommended)
- Use SSD storage for better I/O performance
- Close unnecessary applications

## 🔧 Configuration Options

### Database Configuration
```python
# In simple_knowledge_graph.py
graph = SimpleKnowledgeGraph(
    db_path="/path/to/your/database.db"  # Custom database path
)
```

### Dashboard Configuration
```bash
# In launch_dashboard.sh
streamlit run analysis_dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
```

### Analysis Configuration
```python
# In simple_knowledge_graph.py
# Adjust similarity thresholds
threshold = 0.3  # Minimum similarity threshold
min_cluster_size = 3  # Minimum cluster size
```

## 📊 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 1GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.9+
- **RAM**: 8GB+
- **Storage**: 2GB+ free space
- **OS**: Linux or macOS

## 🚀 Next Steps

After successful installation:

1. **Read the Quick Start Guide**: `QUICK_START.md`
2. **Explore the Analysis Report**: `ANALYSIS_REPORT.md`
3. **Run the dashboard**: `./launch_dashboard.sh`
4. **Start analyzing your data!**

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** for common solutions
2. **Read the error messages** carefully
3. **Check the logs** for detailed error information
4. **Verify your Python version** and dependencies
5. **Test with the sample data** first

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Database available (`regulations.db`)
- [ ] Scripts executable (`chmod +x launch_dashboard.sh`)
- [ ] Basic test passed (`python3 simple_knowledge_graph.py`)
- [ ] Dashboard accessible (http://localhost:8501)

Once all items are checked, you're ready to use the CPSC Knowledge Graph! 🎉