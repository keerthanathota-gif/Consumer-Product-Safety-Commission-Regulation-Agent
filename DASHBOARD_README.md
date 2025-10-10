# Professional Interactive Dashboard - Setup Guide

## Overview

A complete professional dashboard implementation that fulfills all specified requirements:

✅ **Panel 1: Executive Overview**
- Total redundancy pairs identified
- Distribution by severity (High/Medium/Low)
- Overall document quality score
- Key recommendations summary

✅ **Panel 2: Interactive Redundancy Listing**
- Clickable/sortable table of all redundancy pairs
- Filterable by score, type, severity
- Expandable details showing full text comparison
- Vague words highlighted in context

✅ **Panel 3: Detailed Analysis View**
- Full text of selected redundant pair
- Semantic similarity visualization (gauge charts)
- Overlap analysis with color coding
- AI-generated rewrite suggestions
- Impact assessment

---

## Quick Start

### 1. Install Dependencies

```bash
cd Consumer-Product-Safety-Commission-Regulation-Agent

# Install all required packages
pip install -r requirements_enterprise.txt
```

### 2. Generate Analysis Report (if not already done)

```bash
# First-time setup: Create database
python database.py

# Run the redundancy analysis
python enterprise_redundancy_analyzer.py
```

This will create `enterprise_redundancy_report.json` which the dashboard reads.

### 3. Launch the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

---

## Dashboard Features

### **Interactive Sidebar Controls**

- **Impact Severity Filter**: Filter by High/Medium/Low severity
- **Overlap Type Filter**: Filter by Semantic/Lexical/Structural
- **Similarity Threshold Slider**: Adjust minimum similarity score (0.0-1.0)
- **Confidence Threshold Slider**: Adjust minimum confidence score (0.0-1.0)
- **Navigation**: Quick jump to any panel

### **Panel 1: Executive Overview**

**Key Metrics Cards:**
- Total redundancy pairs (with percentage)
- High/Medium/Low severity counts
- Document quality scores with color-coded indicators

**Quality Metrics:**
- Readability Score (0-100 scale)
- Vague Language Density (percentage)
- Average Sentence Length
- Structural Consistency

**Distribution Visualizations:**
- Interactive pie chart for severity distribution
- Bar chart for overlap types
- Executive summary with 3 key findings

### **Panel 2: Interactive Redundancy Listing**

**Data Table Features:**
- Sortable columns (Similarity, Confidence, Severity, Type)
- Search and filter capabilities
- Section number references
- Vague word counts
- Color-coded severity indicators

**Sort Options:**
- Sort by: Similarity Score, Confidence, Severity, Type, Vague Words
- Order: Ascending/Descending
- Real-time filtering based on sidebar controls

### **Panel 3: Detailed Analysis View**

**Pair Selection:**
- Dropdown selector with all filtered pairs
- Quick preview in selection (Similarity, Severity, Section numbers)

**Text Comparison:**
- Side-by-side full text display
- Vague words highlighted with yellow badges
- Section numbers and subjects displayed
- Color-coded borders (blue for primary, red for redundant)

**Visualizations:**
- Gauge chart showing similarity score percentage
- Color-coded thresholds (Low/Medium/High zones)
- Delta indicator comparing to threshold

**Analysis Details:**
- Detected vague words with visual highlighting
- Lexical similarity breakdown
- Confidence level percentage
- AI-generated rewrite recommendation in highlighted box

**Impact Assessment:**
- Color-coded severity indicator
- Detailed explanation of impact level
- Actionable recommendations

---

## Dashboard Controls & Filters

### **Sidebar Filters**

All filters work together and update the dashboard in real-time:

1. **Impact Severity** (Multi-select)
   - High: Similarity > 95%
   - Medium: Similarity 85-95%
   - Low: Similarity 70-85%

2. **Overlap Type** (Multi-select)
   - Semantic: Similar meaning, different words
   - Lexical: Shared vocabulary
   - Structural: Similar sentence patterns

3. **Minimum Similarity Score** (Slider: 0.0-1.0)
   - Filters out pairs below threshold
   - Default: 0.70

4. **Minimum Confidence Score** (Slider: 0.0-1.0)
   - Filters by detection confidence
   - Default: 0.80

### **Navigation**

- Use sidebar radio buttons to jump directly to any panel
- All panels remain visible for scrolling

---

## Color Coding System

### **Severity Indicators**

- 🔴 **High** - Red (`#d9534f`)
- 🟡 **Medium** - Orange (`#f0ad4e`)
- 🔵 **Low** - Light Blue (`#5bc0de`)

### **Quality Score Indicators**

- 🟢 **Good** - Green
- 🟡 **Acceptable** - Yellow
- 🔴 **Needs Improvement** - Red

### **Text Highlighting**

- Yellow background with border: Vague words
- Blue left border: Primary section
- Red left border: Redundant section
- Green background: Recommendations

---

## Technical Details

### **Technology Stack**

- **Framework**: Streamlit 1.28.0+
- **Visualization**: Plotly 5.14.0+ (interactive charts)
- **Data Processing**: Pandas 1.5.0+
- **Styling**: Custom CSS with responsive design

### **Performance**

- **Data Caching**: Report loaded once and cached
- **Real-time Filtering**: Instant updates with filter changes
- **Memory Efficient**: Loads only required data
- **Responsive**: Works on desktop and tablet screens

### **File Structure**

```
Consumer-Product-Safety-Commission-Regulation-Agent/
├── dashboard.py                        # Main dashboard application
├── enterprise_redundancy_analyzer.py   # Analysis engine
├── enterprise_redundancy_report.json   # Generated report (required)
├── requirements_enterprise.txt         # All dependencies
└── DASHBOARD_README.md                 # This file
```

---

## Customization

### **Changing Color Schemes**

Edit the CSS in `dashboard.py` (lines 24-69):

```python
.severity-high {
    background-color: #d9534f;  # Change this color
    ...
}
```

### **Adjusting Default Filters**

Modify default values in sidebar controls (lines 140-167):

```python
severity_filter = st.sidebar.multiselect(
    "Impact Severity",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]  # Change defaults
)
```

### **Adding Custom Metrics**

Add new metrics in Panel 1 section (lines 196-236):

```python
with col5:
    st.metric(
        label="Your Custom Metric",
        value=your_calculated_value
    )
```

---

## Troubleshooting

### **Issue: "Report not found"**

**Solution:**
```bash
python enterprise_redundancy_analyzer.py
```

This generates `enterprise_redundancy_report.json` required by the dashboard.

### **Issue: "Module not found: streamlit"**

**Solution:**
```bash
pip install streamlit
# or
pip install -r requirements_enterprise.txt
```

### **Issue: Dashboard shows no data**

**Check:**
1. Verify `enterprise_redundancy_report.json` exists
2. Check that the report contains `redundancy_pairs` array
3. Adjust sidebar filters (they may be filtering out all data)

### **Issue: Charts not displaying**

**Solution:**
```bash
pip install plotly --upgrade
```

### **Issue: Slow performance**

**Tips:**
- Use fewer filters to reduce data processing
- Close other browser tabs
- Ensure sufficient RAM (2GB+ recommended)
- Consider reducing `max_pairs` in analyzer if report is very large

---

## Advanced Features

### **Export Functionality** (Coming Soon)

Future enhancements:
- Export filtered results to CSV
- Generate PDF reports
- Save custom filter presets

### **Integration with Analysis Pipeline**

You can run the dashboard automatically after analysis:

```bash
python enterprise_redundancy_analyzer.py && streamlit run dashboard.py
```

### **Remote Access**

To access the dashboard from other devices on your network:

```bash
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Then access via: `http://YOUR_IP_ADDRESS:8501`

---

## Best Practices

### **For Analysts**

1. Start with default filters, then refine
2. Use Panel 1 for high-level overview
3. Panel 2 for browsing and identifying patterns
4. Panel 3 for detailed investigation of specific pairs
5. Sort by confidence score to focus on high-quality detections

### **For Stakeholders**

1. Focus on Executive Overview (Panel 1)
2. Review Key Recommendations Summary
3. Check Action Plan at bottom of dashboard
4. Use severity distribution to understand scope

### **For Developers**

1. Check report metadata in sidebar
2. Monitor processing statistics
3. Validate detection quality via confidence scores
4. Review validation checks in source report JSON

---

## Screenshots

### Panel 1: Executive Overview
- 4 key metric cards across the top
- Quality score cards with color indicators
- Executive summary with 3 findings
- Distribution charts (pie + bar)

### Panel 2: Interactive Listing
- Full data table with all pairs
- Sort and filter controls
- Color-coded severity badges
- Expandable row details

### Panel 3: Detailed Analysis
- Pair selector dropdown
- Side-by-side text comparison
- Vague word highlighting
- Gauge chart for similarity
- Recommendation box
- Impact assessment

---

## Keyboard Shortcuts

- **R**: Refresh dashboard
- **Ctrl+F**: Search in page (browser default)
- **Esc**: Close expanded elements

---

## Support & Documentation

### **Related Files**

- `ENTERPRISE_README.md` - Analysis engine documentation
- `EMBEDDINGS_EXPLANATION.md` - Technical details on embeddings
- `OPTIMIZATION_GUIDE.md` - Performance tuning

### **Streamlit Documentation**

- Official Docs: https://docs.streamlit.io
- API Reference: https://docs.streamlit.io/library/api-reference
- Community Forum: https://discuss.streamlit.io

---

## Version History

### v1.0.0 (Current)
- ✅ All 3 panels implemented
- ✅ Interactive filtering and sorting
- ✅ Real-time updates
- ✅ Professional styling
- ✅ Vague word highlighting
- ✅ Semantic similarity visualization
- ✅ Color-coded impact assessment
- ✅ AI-generated recommendations display

---

## Success Criteria ✅

All required dashboard specifications met:

- ✅ Panel 1: Executive Overview with metrics and distributions
- ✅ Panel 2: Clickable, sortable, filterable redundancy table
- ✅ Panel 3: Full text comparison with visualizations
- ✅ Vague words highlighted in context
- ✅ Semantic similarity visualization (gauge charts)
- ✅ Overlap analysis with color coding
- ✅ AI rewrite suggestions displayed
- ✅ Impact assessment with severity indicators
- ✅ Professional styling and responsive design
- ✅ Real-time interactivity
- ✅ Executive-ready presentation quality

---

**Built with Streamlit. Powered by Sentence-BERT. Ready for production.**

For questions or issues, refer to the main project documentation or Streamlit community resources.
