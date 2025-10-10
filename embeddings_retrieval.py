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
import sys
sys.stdout.reconfigure(encoding='utf-8')


class RegulationEmbeddings:
    """
    Creates and manages embeddings for regulation sections.
    Provides semantic search and retrieval functionality.
    """

    def __init__(self, db_path: str = "regulations.db",
                 embeddings_path: str = "section_embeddings.pkl"):
        self.db_path = Path(db_path)
        self.embeddings_path = Path(embeddings_path)
        self.vectorizer = None
        self.embeddings_matrix = None
        self.sections_data = []

    def connect_db(self) -> sqlite3.Connection:
        """Create database connection."""
        return sqlite3.connect(str(self.db_path))

    def fetch_all_sections(self) -> List[Dict]:
        """Fetch all sections from the database with hierarchical context."""
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

        sections = [
            {
                'section_id': row[0],
                'section_number': row[1] or '',
                'subject': row[2] or '',
                'text': row[3],
                'citation': row[4] or '',
                'part_heading': row[5],
                'subchapter_name': row[6],
                'chapter_name': row[7]
            }
            for row in rows
        ]

        print(f"Fetched {len(sections)} sections from database.")
        return sections

    def create_embeddings(self, max_features: int = 5000,
                         ngram_range: Tuple[int, int] = (1, 2)):
        """Create TF-IDF embeddings for all sections."""
        print("\n=== Creating Embeddings ===")

        self.sections_data = self.fetch_all_sections()
        if not self.sections_data:
            print("No sections found in database!")
            return

        # Prepare documents
        documents = [
            ' '.join(filter(None, [
                s['section_number'], s['subject'], s['text']
            ]))
            for s in self.sections_data
        ]

        print(f"Preparing to embed {len(documents)} documents...")

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )

        print("Computing TF-IDF embeddings...")
        self.embeddings_matrix = self.vectorizer.fit_transform(documents)

        print(f"Embeddings created successfully:")
        print(f"  • Documents: {self.embeddings_matrix.shape[0]}")
        print(f"  • Vocabulary size: {self.embeddings_matrix.shape[1]}")
        print(f"  • Matrix density: {(self.embeddings_matrix.nnz / (self.embeddings_matrix.shape[0] * self.embeddings_matrix.shape[1])) * 100:.2f}%")

        self.save_embeddings()
        print(f"\n✅ Embeddings successfully saved to: {self.embeddings_path}")

    def save_embeddings(self):
        """Save embeddings and metadata to disk."""
        data = {
            'vectorizer': self.vectorizer,
            'embeddings_matrix': self.embeddings_matrix,
            'sections_data': self.sections_data
        }
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(data, f)

        size_mb = self.embeddings_path.stat().st_size / (1024 * 1024)
        print(f"Saved {self.embeddings_path.name} ({size_mb:.2f} MB)")

    def load_embeddings(self) -> bool:
        """Load precomputed embeddings."""
        if not self.embeddings_path.exists():
            print(f"Embeddings file not found: {self.embeddings_path}")
            return False

        with open(self.embeddings_path, 'rb') as f:
            data = pickle.load(f)

        self.vectorizer = data['vectorizer']
        self.embeddings_matrix = data['embeddings_matrix']
        self.sections_data = data['sections_data']

        print(f"Loaded embeddings for {len(self.sections_data)} sections.")
        return True

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> List[Dict]:
        """Semantic search for relevant sections."""
        if self.embeddings_matrix is None or self.vectorizer is None:
            print("Embeddings not loaded. Call load_embeddings() or create_embeddings() first.")
            return []

        query_embedding = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        ranked_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked_indices[:top_k]:
            score = similarities[idx]
            if score < min_similarity:
                continue
            section = self.sections_data[idx].copy()
            section['similarity_score'] = float(score)
            results.append(section)

        return results

    def get_statistics(self) -> Dict:
        """Return statistics about the embeddings."""
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
    """Main function to generate embeddings and print statistics."""
    script_dir = Path(__file__).parent.resolve()
    db_path = script_dir / "regulations.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Please run database.py first to create the database.")
        sys.exit(1)

    embeddings_system = RegulationEmbeddings(
        db_path=str(db_path),
        embeddings_path=str(script_dir / "section_embeddings.pkl")
    )

    print("\nStarting embedding creation...")
    embeddings_system.create_embeddings()

    print("\n=== Embedding Statistics ===")
    stats = embeddings_system.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n✅ Embedding generation completed successfully!")


if __name__ == "__main__":
    main()
