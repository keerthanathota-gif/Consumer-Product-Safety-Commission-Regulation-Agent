#!/usr/bin/env python3
"""
Embeddings and Retrieval System for CPSC Regulations
Creates embeddings for section text and provides semantic search functionality.
"""

import sqlite3
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class RegulationEmbeddings:
    """
    Creates and manages embeddings for regulation sections.
    Provides semantic search and retrieval functionality.
    """

    def __init__(self, db_path: str = "regulations.db",
                 embeddings_path: str = "section_embeddings.pkl"):
        """
        Initialize the embeddings system.

        Args:
            db_path: Path to the SQLite database
            embeddings_path: Path to save/load embeddings
        """
        self.db_path = Path(db_path)
        self.embeddings_path = Path(embeddings_path)
        self.vectorizer = None
        self.embeddings_matrix = None
        self.sections_data = []

    def connect_db(self) -> sqlite3.Connection:
        """Create database connection."""
        return sqlite3.connect(str(self.db_path))

    def fetch_all_sections(self) -> List[Dict]:
        """
        Fetch all sections from the database with hierarchical context.

        Returns:
            List of dictionaries containing section data with full context
        """
        conn = self.connect_db()
        cursor = conn.cursor()

        query = """
            SELECT
                s.section_id,
                s.section_number,
                s.subject,
                s.text,
                s.citation,
                p.heading as part_heading,
                sc.subchapter_name,
                c.chapter_name
            FROM sections s
            JOIN parts p ON s.part_id = p.part_id
            JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
            JOIN chapters c ON sc.chapter_id = c.chapter_id
            WHERE s.text IS NOT NULL AND s.text != ''
            ORDER BY s.section_id
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        sections = []
        for row in rows:
            section = {
                'section_id': row[0],
                'section_number': row[1] if row[1] else '',
                'subject': row[2] if row[2] else '',
                'text': row[3],
                'citation': row[4] if row[4] else '',
                'part_heading': row[5],
                'subchapter_name': row[6],
                'chapter_name': row[7]
            }
            sections.append(section)

        print(f"Fetched {len(sections)} sections from database")
        return sections

    def create_embeddings(self, max_features: int = 5000,
                         ngram_range: Tuple[int, int] = (1, 2)):
        """
        Create TF-IDF embeddings for all sections.

        The embeddings capture the semantic meaning of each section by:
        1. Combining section number, subject, and text into a single document
        2. Using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
        3. Creating dense vector representations that can be compared

        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to use (1,2) means unigrams and bigrams
        """
        print("\n=== Creating Embeddings ===")

        # Fetch all sections
        self.sections_data = self.fetch_all_sections()

        if not self.sections_data:
            print("No sections found in database!")
            return

        # Prepare documents for embedding
        # Each document combines: section_number + subject + text
        # This provides complete context for semantic matching
        documents = []
        for section in self.sections_data:
            # Combine all relevant text fields
            doc_parts = []

            if section['section_number']:
                doc_parts.append(section['section_number'])
            if section['subject']:
                doc_parts.append(section['subject'])
            if section['text']:
                doc_parts.append(section['text'])

            # Create a single document string
            document = ' '.join(doc_parts)
            documents.append(document)

        print(f"Preparing to embed {len(documents)} documents...")

        # Create TF-IDF vectorizer
        # TF-IDF measures how important a word is to a document
        # - TF (Term Frequency): How often a word appears in the document
        # - IDF (Inverse Document Frequency): How rare/common the word is across all documents
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,  # Limit vocabulary size
            ngram_range=ngram_range,    # Use 1-grams and 2-grams (single words and word pairs)
            stop_words='english',        # Remove common English words (the, is, at, etc.)
            lowercase=True,              # Convert all text to lowercase
            min_df=2,                    # Ignore words that appear in less than 2 documents
            max_df=0.8                   # Ignore words that appear in more than 80% of documents
        )

        # Fit and transform documents to create embeddings matrix
        # Result: sparse matrix of shape (n_documents, n_features)
        print("Computing TF-IDF embeddings...")
        self.embeddings_matrix = self.vectorizer.fit_transform(documents)

        print(f"Embeddings shape: {self.embeddings_matrix.shape}")
        print(f"  - {self.embeddings_matrix.shape[0]} documents")
        print(f"  - {self.embeddings_matrix.shape[1]} features (vocabulary size)")
        print(f"  - Matrix density: {(self.embeddings_matrix.nnz / (self.embeddings_matrix.shape[0] * self.embeddings_matrix.shape[1])) * 100:.2f}%")

        # Save embeddings
        self.save_embeddings()

        print("\nEmbeddings created successfully!")

    def save_embeddings(self):
        """Save embeddings and metadata to disk."""
        data = {
            'vectorizer': self.vectorizer,
            'embeddings_matrix': self.embeddings_matrix,
            'sections_data': self.sections_data
        }

        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"Embeddings saved to: {self.embeddings_path}")
        print(f"File size: {self.embeddings_path.stat().st_size / (1024*1024):.2f} MB")

    def load_embeddings(self) -> bool:
        """
        Load pre-computed embeddings from disk.

        Returns:
            True if embeddings loaded successfully, False otherwise
        """
        if not self.embeddings_path.exists():
            print(f"Embeddings file not found: {self.embeddings_path}")
            return False

        print(f"Loading embeddings from: {self.embeddings_path}")

        with open(self.embeddings_path, 'rb') as f:
            data = pickle.load(f)

        self.vectorizer = data['vectorizer']
        self.embeddings_matrix = data['embeddings_matrix']
        self.sections_data = data['sections_data']

        print(f"Loaded embeddings for {len(self.sections_data)} sections")
        return True

    def search(self, query: str, top_k: int = 5,
               min_similarity: float = 0.0) -> List[Dict]:
        """
        Semantic search for relevant sections.

        How it works:
        1. Convert query text to embedding using the same vectorizer
        2. Calculate cosine similarity between query and all section embeddings
        3. Rank sections by similarity score
        4. Return top-k most relevant sections

        Args:
            query: Search query text
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of dictionaries containing section data and similarity scores
        """
        if self.embeddings_matrix is None or self.vectorizer is None:
            print("Embeddings not loaded. Call load_embeddings() or create_embeddings() first.")
            return []

        # Transform query to embedding vector
        query_embedding = self.vectorizer.transform([query])

        # Calculate cosine similarity between query and all sections
        # Cosine similarity measures the angle between two vectors
        # Score range: 0 (completely different) to 1 (identical)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]

        # Get indices sorted by similarity (highest first)
        ranked_indices = np.argsort(similarities)[::-1]

        # Prepare results
        results = []
        for idx in ranked_indices[:top_k]:
            similarity_score = similarities[idx]

            # Skip results below minimum similarity threshold
            if similarity_score < min_similarity:
                continue

            section = self.sections_data[idx].copy()
            section['similarity_score'] = float(similarity_score)
            results.append(section)

        return results

    def retrieve(self, query: str, top_k: int = 5,
                min_similarity: float = 0.1, verbose: bool = True) -> List[Dict]:
        """
        High-level retrieval function with formatted output.

        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            verbose: Print formatted results

        Returns:
            List of relevant sections with similarity scores
        """
        results = self.search(query, top_k, min_similarity)

        if verbose:
            self.print_results(query, results)

        return results

    def print_results(self, query: str, results: List[Dict]):
        """Print search results in a formatted way."""
        print(f"\n{'='*80}")
        print(f"SEARCH QUERY: {query}")
        print(f"{'='*80}")
        print(f"Found {len(results)} relevant sections\n")

        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Section: {result['section_number']}")
            print(f"Subject: {result['subject']}")
            print(f"Chapter: {result['chapter_name']}")
            print(f"Subchapter: {result['subchapter_name']}")
            print(f"Part: {result['part_heading']}")
            print(f"\nText Preview: {result['text'][:300]}...")
            print(f"-" * 80)

    def get_section_by_id(self, section_id: int) -> Optional[Dict]:
        """
        Retrieve a specific section by its ID.

        Args:
            section_id: The section ID to retrieve

        Returns:
            Section dictionary or None if not found
        """
        for section in self.sections_data:
            if section['section_id'] == section_id:
                return section
        return None

    def find_similar_sections(self, section_id: int, top_k: int = 5) -> List[Dict]:
        """
        Find sections similar to a given section.

        Args:
            section_id: Reference section ID
            top_k: Number of similar sections to return

        Returns:
            List of similar sections with similarity scores
        """
        # Find the index of the given section
        section_idx = None
        for idx, section in enumerate(self.sections_data):
            if section['section_id'] == section_id:
                section_idx = idx
                break

        if section_idx is None:
            print(f"Section ID {section_id} not found")
            return []

        # Get embedding for this section
        section_embedding = self.embeddings_matrix[section_idx]

        # Calculate similarities
        similarities = cosine_similarity(section_embedding, self.embeddings_matrix)[0]

        # Get top-k (excluding the section itself)
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices:
            if idx == section_idx:  # Skip the section itself
                continue

            if len(results) >= top_k:
                break

            section = self.sections_data[idx].copy()
            section['similarity_score'] = float(similarities[idx])
            results.append(section)

        return results

    def get_statistics(self) -> Dict:
        """Get statistics about the embeddings."""
        if not self.sections_data:
            return {}

        text_lengths = [len(s['text']) for s in self.sections_data]

        return {
            'total_sections': len(self.sections_data),
            'embedding_dimensions': self.embeddings_matrix.shape[1] if self.embeddings_matrix is not None else 0,
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0,
            'avg_text_length': np.mean(text_lengths),
            'min_text_length': np.min(text_lengths),
            'max_text_length': np.max(text_lengths),
            'total_chapters': len(set(s['chapter_name'] for s in self.sections_data)),
            'total_subchapters': len(set(s['subchapter_name'] for s in self.sections_data)),
            'total_parts': len(set(s['part_heading'] for s in self.sections_data))
        }


def main():
    """Main function to demonstrate embeddings creation and retrieval."""
    import sys

    # Get the script directory
    script_dir = Path(__file__).parent.resolve()
    db_path = script_dir / "regulations.db"

    # Check if database exists
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Please run database.py first to create the database.")
        sys.exit(1)

    # Initialize embeddings system
    embeddings_system = RegulationEmbeddings(
        db_path=str(db_path),
        embeddings_path=str(script_dir / "section_embeddings.pkl")
    )

    # Create embeddings
    print("Creating embeddings for all sections...")
    embeddings_system.create_embeddings()

    # Display statistics
    print("\n=== Statistics ===")
    stats = embeddings_system.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Example searches
    print("\n\n=== Example Search 1: Toy Safety ===")
    results = embeddings_system.retrieve("toy safety requirements for children", top_k=3)

    print("\n\n=== Example Search 2: Flammable Materials ===")
    results = embeddings_system.retrieve("flammable fabrics and clothing", top_k=3)

    print("\n\n=== Example Search 3: Product Recalls ===")
    results = embeddings_system.retrieve("product recall procedures", top_k=3)

    print("\n\nEmbeddings system ready!")
    print("You can now use the RegulationEmbeddings class to search regulations.")


if __name__ == "__main__":
    main()
