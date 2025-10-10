# Embeddings and Retrieval System - Detailed Explanation

## Overview

This document explains how the embeddings system works, what data is embedded, and how to use it for semantic search of CPSC regulations.

---

## What is a "Section"?

A **section** in the CPSC regulations database represents the finest level of regulatory detail. Each section contains:

### Section Structure

```
{
    "section_id": 1,                    # Unique database ID
    "section_number": "§ 1000.1",       # Official section number (CFR notation)
    "subject": "The Commission.",       # Brief topic/title
    "text": "Full regulatory text...",  # Complete regulation content
    "citation": "",                     # Legal citations (if any)
    "part_heading": "PART 1000—...",    # Parent part
    "subchapter_name": "SUBCHAPTER A—...", # Parent subchapter
    "chapter_name": "CHAPTER II—..."    # Parent chapter
}
```

### Example Real Section

**Section Number:** § 1000.1
**Subject:** The Commission.
**Text:**
> "(a) The Consumer Product Safety Commission is an independent regulatory agency formed on May 14, 1973, under the provisions of the Consumer Product Safety Act (Pub. L. 92-573, 86 Stat. 1207, as amended (15 U.S.C. 2051)). The purposes of the Commission under the CPSA are: (1) To protect the public against unreasonable risks of injury associated with consumer products; (2) To assist consumers in evaluating the comparative safety of consumer products..."

**Hierarchy:**
- Chapter: CHAPTER II—CONSUMER PRODUCT SAFETY COMMISSION
- Subchapter: SUBCHAPTER A—GENERAL
- Part: PART 1000—COMMISSION ORGANIZATION AND FUNCTIONS

---

## How Embeddings Work

### What are Embeddings?

Embeddings are **numerical representations** of text that capture semantic meaning. Instead of comparing text word-by-word, embeddings allow us to find regulations based on **meaning and context**.

### The Embedding Process

#### 1. Document Preparation
Each section is converted into a single "document" by combining:
```python
document = section_number + " " + subject + " " + text
# Example: "§ 1000.1 The Commission. The Consumer Product Safety Commission is..."
```

#### 2. TF-IDF Vectorization

**TF-IDF** = Term Frequency-Inverse Document Frequency

This creates a numerical vector for each document where:

- **TF (Term Frequency)**: How often a word appears in THIS document
  - Example: "safety" appears 10 times in a section → higher TF

- **IDF (Inverse Document Frequency)**: How rare/common a word is across ALL documents
  - Common words like "the", "is", "and" → low IDF (not distinctive)
  - Specific terms like "flammable", "hazard" → high IDF (more meaningful)

**TF-IDF Score = TF × IDF**

Words that appear frequently in a specific document but rarely across all documents get the highest scores.

#### 3. Vector Representation

Each section becomes a vector with 5,000 dimensions (features):

```
Section Vector = [0.0, 0.0, 0.23, 0.0, ..., 0.45, 0.0, ...]
                  ↑     ↑     ↑                ↑
                  word1 word2 word3          word5000
```

Most values are 0 (sparse matrix) because each document uses only a small subset of vocabulary.

#### 4. Similarity Calculation

To find relevant sections, we:

1. **Convert query to embedding** using the same vectorizer
2. **Calculate cosine similarity** between query vector and all section vectors
3. **Rank sections** by similarity score

**Cosine Similarity** measures the angle between two vectors:
- Score = 1.0: Identical meaning
- Score = 0.5: Moderately similar
- Score = 0.0: Completely different

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

---

## Example Output Walkthrough

Let's search for: **"toy safety requirements for children"**

### Step 1: Query Processing

Query is converted to embedding vector:
```python
query = "toy safety requirements for children"
query_embedding = vectorizer.transform([query])
# Result: Vector [0.0, 0.12, 0.0, 0.34, ..., 0.0, 0.28]
```

### Step 2: Similarity Calculation

System compares query against all 2,000+ sections:
```
Section 1:  similarity = 0.0234
Section 2:  similarity = 0.1523
Section 3:  similarity = 0.6789  ← High match!
Section 4:  similarity = 0.0456
...
```

### Step 3: Results Returned

```python
[
    {
        'section_id': 142,
        'section_number': '§ 1500.18',
        'subject': 'Banned toys and other banned articles intended for use by children.',
        'text': 'The following types of toys are banned under section 2(q)(1) of the Federal Hazardous Substances Act...',
        'similarity_score': 0.6789,  # Very relevant!
        'chapter_name': 'CHAPTER II—CONSUMER PRODUCT SAFETY COMMISSION',
        'subchapter_name': 'SUBCHAPTER C—FEDERAL HAZARDOUS SUBSTANCES ACT REGULATIONS',
        'part_heading': 'PART 1500—HAZARDOUS SUBSTANCES AND ARTICLES'
    },
    {
        'section_id': 156,
        'section_number': '§ 1501.2',
        'subject': 'Definitions.',
        'text': 'For purposes of this part: (a) Children means persons who are less than 3 years old...',
        'similarity_score': 0.5432,  # Moderately relevant
        'chapter_name': 'CHAPTER II—CONSUMER PRODUCT SAFETY COMMISSION',
        'subchapter_name': 'SUBCHAPTER C—FEDERAL HAZARDOUS SUBSTANCES ACT REGULATIONS',
        'part_heading': 'PART 1501—METHOD FOR IDENTIFYING TOYS AND OTHER ARTICLES'
    },
    {
        'section_id': 178,
        'section_number': '§ 1502.3',
        'subject': 'Requirements for determining sharp points in toys.',
        'text': 'Test procedures for sharp points are described below...',
        'similarity_score': 0.4821,  # Somewhat relevant
        'chapter_name': 'CHAPTER II—CONSUMER PRODUCT SAFETY COMMISSION',
        'subchapter_name': 'SUBCHAPTER C—FEDERAL HAZARDOUS SUBSTANCES ACT REGULATIONS',
        'part_heading': 'PART 1502—PROCEDURES FOR TESTING TOYS'
    }
]
```

### Understanding the Results

**Result 1: Best Match (0.6789 similarity)**
- Contains keywords: "toys", "banned", "children"
- Discusses safety requirements
- High semantic alignment with query intent

**Result 2: Good Match (0.5432 similarity)**
- Defines "children" (age < 3 years)
- Related to toy identification
- Contextually relevant to child safety

**Result 3: Moderate Match (0.4821 similarity)**
- About toy testing procedures
- Mentions "sharp points" safety aspect
- Somewhat related to safety requirements

---

## What Gets Embedded?

### Included in Embeddings:
1. **Section number** (§ 1000.1) - helps with specific section searches
2. **Subject** (The Commission) - captures the topic
3. **Full text** - the complete regulation content

### Why This Matters:
- You can search by **section number**: "§ 1500.18"
- You can search by **topic**: "flammable fabrics"
- You can search by **specific concepts**: "choking hazards for toddlers"
- You can search by **use case**: "what are requirements for children's nightwear?"

### Not Embedded (but available in results):
- Citation information
- Part headings
- Subchapter names
- Chapter names

These are included in search results for context but not used in similarity matching.

---

## Key Features Explained

### 1. **Semantic Search**
Traditional keyword search: "toy" only finds sections with exact word "toy"
Semantic search: finds sections about "playthings", "children's products", "youth articles"

### 2. **Context Preservation**
Each result includes full hierarchy:
```
Chapter → Subchapter → Part → Section
```

### 3. **Similarity Scoring**
- **0.7-1.0**: Highly relevant, direct match
- **0.4-0.7**: Good match, contextually related
- **0.2-0.4**: Weak match, tangentially related
- **0.0-0.2**: Not relevant

### 4. **Flexible Querying**
- Natural language: "What are the rules about fire safety in children's clothing?"
- Keywords: "flammable fabric children"
- Section reference: "§ 1500"
- Concept: "choking hazard small parts"

---

## Technical Details

### Embedding Matrix Shape
```
Shape: (2000+ sections, 5000 features)
```
- Each row = one section
- Each column = one vocabulary term (word or bigram)
- Sparse matrix (mostly zeros)

### Parameters

**max_features = 5000**
- Vocabulary limited to 5,000 most important terms
- Reduces noise, improves performance

**ngram_range = (1, 2)**
- Uses unigrams (single words): "safety", "toy"
- Uses bigrams (word pairs): "safety requirements", "toy standards"
- Captures phrases for better context

**stop_words = 'english'**
- Removes common words: "the", "is", "and", "or"
- Focuses on meaningful content words

**min_df = 2**
- Ignore words appearing in < 2 documents
- Removes typos and rare terms

**max_df = 0.8**
- Ignore words appearing in > 80% of documents
- Removes overly common terms like "commission", "section"

---

## Usage Examples

### Example 1: Basic Search
```python
from embeddings_retrieval import RegulationEmbeddings

# Initialize
system = RegulationEmbeddings()
system.load_embeddings()

# Search
results = system.retrieve("lead paint in children's toys", top_k=5)
```

### Example 2: Filtered Search
```python
# Only return highly relevant results
results = system.search(
    query="bicycle safety standards",
    top_k=10,
    min_similarity=0.3  # Only scores > 0.3
)
```

### Example 3: Find Similar Sections
```python
# Find sections similar to section #142
similar = system.find_similar_sections(section_id=142, top_k=5)
```

### Example 4: Get Specific Section
```python
section = system.get_section_by_id(142)
print(section['subject'])
print(section['text'])
```

---

## Performance Notes

- **Initial embedding creation**: ~30-60 seconds (one-time)
- **Loading pre-computed embeddings**: ~1-2 seconds
- **Single search query**: ~0.1-0.5 seconds
- **Storage**: ~2-5 MB for embeddings file

---

## Advantages Over Simple Keyword Search

| Feature | Keyword Search | Semantic Search (Embeddings) |
|---------|---------------|------------------------------|
| Find synonyms | No | Yes |
| Understand context | No | Yes |
| Handle misspellings | No | Partial |
| Find related concepts | No | Yes |
| Ranking quality | Basic | Advanced |
| Query flexibility | Low | High |

---

## Next Steps

You can enhance this system by:

1. **Using sentence-transformers** for better embeddings (BERT-based)
2. **Adding filters** by chapter, subchapter, or part
3. **Implementing hybrid search** (keyword + semantic)
4. **Creating a web interface** for easier access
5. **Adding question-answering** capabilities with LLMs

---

## Summary

The embeddings system converts regulatory sections into numerical vectors that capture semantic meaning. This enables powerful searches like:

- "What are the safety rules for toys with small parts?"
- "Find regulations about flammable materials in children's products"
- "Show me requirements for product recalls"

All without needing to know exact keywords or section numbers!
