# CPSC Regulations Redundancy Analysis

## Overview

This tool analyzes Title 16 CFR (Consumer Product Safety Commission) regulations to identify:
1. **Redundant sections** - Near-duplicate text (>95% similarity)
2. **Overlapping sections** - Partially similar content (80-95% similarity)
3. **Parity sections** - Semantically similar but differently worded (70-80% similarity)

## Features

- **Embedding-based analysis** using TF-IDF vectors (upgradeable to transformer models)
- **Chunk-level overlap detection** to find specific overlapping text
- **Per-section counts** showing how many similarity issues each section has
- **Actionable recommendations** for consolidating or cleaning regulations
- **Structured JSON output** with section IDs and text snippets

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python embedding_redundancy_analyzer.py
```

This will:
1. Load all sections from `regulations.db`
2. Create embeddings for each section
3. Compare all pairs of sections
4. Generate a detailed JSON report: `redundancy_analysis_results.json`

### Output Structure

The JSON output contains:

```json
{
  "metadata": {
    "total_sections_analyzed": 1234,
    "embedding_dimensions": 1000,
    "thresholds": {
      "redundant": 0.95,
      "overlap": 0.80,
      "parity": 0.70
    }
  },
  "summary": {
    "redundant_pairs": 45,
    "overlapping_pairs": 123,
    "parity_pairs": 234,
    "sections_with_issues": 345
  },
  "redundant_sections": [
    {
      "section1_id": 123,
      "section1_label": "§ 1000.1",
      "section1_heading": "The Commission",
      "section2_id": 456,
      "section2_label": "§ 1001.2",
      "section2_heading": "Commission Authority",
      "embedding_similarity": 0.96,
      "text_similarity": 0.97,
      "max_similarity": 0.97,
      "text1_snippet": "First 300 chars...",
      "text2_snippet": "First 300 chars..."
    }
  ],
  "overlapping_sections": [
    {
      "section1_id": 789,
      "section1_label": "§ 1500.1",
      "section2_id": 890,
      "section2_label": "§ 1501.1",
      "embedding_similarity": 0.85,
      "text_similarity": 0.87,
      "max_similarity": 0.87,
      "overlapping_chunks": [
        {
          "text": "Overlapping text snippet...",
          "length": 50,
          "similarity": 0.92,
          "position1": 10,
          "position2": 25
        }
      ],
      "num_overlaps": 3
    }
  ],
  "parity_sections": [
    {
      "section1_id": 111,
      "section1_label": "§ 1200.1",
      "section2_id": 222,
      "section2_label": "§ 1300.1",
      "embedding_similarity": 0.75,
      "text_similarity": 0.73,
      "max_similarity": 0.75,
      "text1_snippet": "First 200 chars...",
      "text2_snippet": "First 200 chars..."
    }
  ],
  "section_counts": [
    {
      "section_id": 123,
      "section_label": "§ 1000.1",
      "heading": "The Commission",
      "part": "PART 1000—COMMISSION ORGANIZATION",
      "redundancy_count": 5,
      "overlap_count": 12,
      "parity_count": 8,
      "total_similarity_issues": 25
    }
  ],
  "recommendations": [
    {
      "type": "redundancy",
      "priority": "HIGH",
      "affected_sections": 90,
      "recommendation": "Remove or consolidate near-duplicate sections",
      "details": [...]
    }
  ]
}
```

## Understanding the Results

### 1. Redundant Sections (>95% similar)
These are near-duplicates that should be consolidated or removed:
- **Action**: Keep one version or merge into a comprehensive section
- **Priority**: HIGH

### 2. Overlapping Sections (80-95% similar)
These share significant common content:
- **Action**: Extract common text into shared definitions or base sections
- **Priority**: MEDIUM

### 3. Parity Sections (70-80% similar)
These have similar meaning but different wording:
- **Action**: Add cross-references for better navigation
- **Priority**: LOW

### 4. Section Counts
Shows which sections have the most similarity issues:
- High counts may indicate core definitions that should be centralized
- Useful for prioritizing cleanup efforts

## Customization

### Adjust Similarity Thresholds

Edit the thresholds in `embedding_redundancy_analyzer.py`:

```python
self.REDUNDANT_THRESHOLD = 0.95  # Near-duplicate
self.OVERLAP_THRESHOLD = 0.80    # Partial overlap
self.PARITY_THRESHOLD = 0.70     # Semantic parity
```

### Use Better Embeddings

For production use, upgrade to transformer-based embeddings:

```python
from sentence_transformers import SentenceTransformer

# Replace create_embeddings() method
def create_embeddings(self):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [self._get_full_text(section) for section in self.sections]
    self.embeddings = model.encode(texts)
    return model
```

Install dependency:
```bash
pip install sentence-transformers
```

### Use OpenAI Embeddings

```python
from openai import OpenAI

def create_embeddings(self):
    client = OpenAI(api_key='your-api-key')
    texts = [self._get_full_text(section) for section in self.sections]

    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    self.embeddings = np.array(embeddings)
```

## Interpreting Recommendations

The tool provides 4 types of recommendations:

1. **Redundancy**: Sections that are near-duplicates
   - Review pairs and decide which to keep
   - Consider consolidating into single authoritative section

2. **Overlap**: Sections with shared content
   - Extract common text into referenced definitions
   - Reduces maintenance burden

3. **Parity**: Semantically similar sections
   - Add "See also" cross-references
   - Improves user navigation

4. **Multiple Issues**: Sections appearing in many pairs
   - Often core concepts used throughout regulations
   - Consider creating central definition sections

## Performance

- Analysis time depends on number of sections: O(n²)
- For large datasets (>5000 sections), consider:
  - Using approximate nearest neighbors (FAISS, Annoy)
  - Processing in batches
  - Sampling strategy for initial analysis

## Example Workflow

1. **Run analysis**:
   ```bash
   python embedding_redundancy_analyzer.py
   ```

2. **Review output**: `redundancy_analysis_results.json`

3. **Prioritize**: Start with HIGH priority recommendations

4. **Take action**:
   - Remove redundant sections
   - Create shared definition sections
   - Add cross-references

5. **Re-run**: Verify improvements

## Integration with Your Database

If your database has a different schema, modify the `load_sections()` method:

```python
def load_sections(self):
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Adjust query to match your schema
    cursor.execute("""
        SELECT
            section_id,
            year,
            title,
            part,
            section_label,
            heading,
            text,
            citations
        FROM your_table_name
        WHERE text IS NOT NULL AND text != ''
    """)

    for row in cursor.fetchall():
        self.sections.append({
            'section_id': row[0],
            'year': row[1],
            'title': row[2],
            'part': row[3],
            'section_label': row[4],
            'heading': row[5],
            'text': row[6],
            'citations': row[7]
        })

    conn.close()
```

## Troubleshooting

### Out of Memory
- Process in batches
- Reduce embedding dimensions
- Use sparse embeddings

### Slow Performance
- Use approximate nearest neighbors
- Sample sections for initial analysis
- Parallelize comparisons

### Poor Results
- Check embedding quality
- Adjust similarity thresholds
- Use domain-specific embeddings

## License

This tool is designed for regulatory analysis and compliance purposes.
