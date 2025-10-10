#!/usr/bin/env python3
"""
Optimized Embedding-Based Redundancy Analyzer for CPSC Regulations
Analyzes redundancy, overlap, and parity in regulation sections using efficient algorithms.

Performance Optimizations:
1. Sparse matrix operations for TF-IDF embeddings
2. Batch processing with configurable batch sizes
3. Early stopping for low-similarity pairs
4. Efficient chunking with difflib SequenceMatcher
5. Optional approximate nearest neighbors (FAISS) for large datasets
6. Progress tracking and time estimates
"""

import sqlite3
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from collections import defaultdict, Counter
import sys
from datetime import datetime, timedelta
sys.stdout.reconfigure(encoding='utf-8')


class OptimizedRedundancyAnalyzer:
    """
    Analyzes regulation sections for redundancy, overlap, and parity using optimized algorithms.

    Optimizations:
    - Sparse matrix operations (10-100x faster than dense)
    - Batch processing to control memory usage
    - Early stopping for pairs below minimum threshold
    - Efficient text similarity with quick pre-checks
    - Optional FAISS for approximate nearest neighbors (1000x faster for large datasets)
    """

    # Similarity thresholds
    REDUNDANT_THRESHOLD = 0.95  # Near-duplicate sections
    OVERLAP_THRESHOLD = 0.80    # Significantly overlapping sections
    PARITY_THRESHOLD = 0.70     # Semantically similar sections

    def __init__(self, db_path: str = "regulations.db", use_faiss: bool = False):
        self.db_path = Path(db_path)
        self.sections = []
        self.vectorizer = None
        self.embeddings = None
        self.use_faiss = use_faiss
        self.faiss_index = None

        # Results storage
        self.redundant_pairs = []
        self.overlapping_pairs = []
        self.parity_pairs = []

        # Performance tracking
        self.start_time = None
        self.comparisons_done = 0
        self.total_comparisons = 0

    def load_sections(self):
        """Load all sections from the database."""
        print("\n=== Loading Sections from Database ===")
        conn = sqlite3.connect(str(self.db_path))
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

        self.sections = [
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

        print(f"✓ Loaded {len(self.sections)} sections")

    def create_embeddings(self, max_features: int = 5000):
        """Create TF-IDF embeddings using sparse matrices for efficiency."""
        print("\n=== Creating Embeddings ===")

        documents = [
            ' '.join(filter(None, [
                s['section_number'], s['subject'], s['text']
            ]))
            for s in self.sections
        ]

        print(f"Processing {len(documents)} documents...")

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )

        # This creates a sparse matrix - much faster than dense
        self.embeddings = self.vectorizer.fit_transform(documents)

        density = (self.embeddings.nnz / (self.embeddings.shape[0] * self.embeddings.shape[1])) * 100
        print(f"✓ Created sparse TF-IDF matrix:")
        print(f"  • Shape: {self.embeddings.shape}")
        print(f"  • Density: {density:.2f}%")
        print(f"  • Memory saved: ~{100-density:.1f}% vs dense matrix")

        # Optional: Create FAISS index for very large datasets
        if self.use_faiss:
            self._create_faiss_index()

    def _create_faiss_index(self):
        """Create FAISS index for approximate nearest neighbor search."""
        try:
            import faiss

            print("\n=== Creating FAISS Index ===")
            # Convert sparse to dense for FAISS (only if reasonable size)
            if self.embeddings.shape[0] > 10000:
                print("Warning: Large dataset - FAISS conversion may use significant memory")

            dense_embeddings = self.embeddings.toarray().astype('float32')

            # Normalize vectors for cosine similarity
            faiss.normalize_L2(dense_embeddings)

            # Create index
            dimension = dense_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
            self.faiss_index.add(dense_embeddings)

            print(f"✓ FAISS index created with {self.faiss_index.ntotal} vectors")
        except ImportError:
            print("⚠ FAISS not available - falling back to batch processing")
            print("  Install with: pip install faiss-cpu")
            self.use_faiss = False

    def _quick_text_similarity(self, text1: str, text2: str) -> float:
        """Fast text similarity using SequenceMatcher with quick ratio check."""
        matcher = SequenceMatcher(None, text1, text2)
        # quick_ratio is much faster and provides upper bound
        quick = matcher.quick_ratio()
        if quick < self.PARITY_THRESHOLD:
            return quick
        # Only compute full ratio if quick check passes
        return matcher.ratio()

    def _find_overlapping_chunks(self, text1: str, text2: str,
                                 min_chunk_length: int = 50) -> List[Dict]:
        """Find overlapping text chunks between two sections."""
        matcher = SequenceMatcher(None, text1, text2)
        chunks = []

        for match in matcher.get_matching_blocks():
            if match.size >= min_chunk_length:
                chunk_text = text1[match.a:match.a + match.size]
                chunks.append({
                    'text': chunk_text[:200],  # Truncate for storage
                    'length': match.size,
                    'position1': match.a,
                    'position2': match.b,
                    'similarity': match.size / max(len(text1), len(text2))
                })

        return chunks

    def analyze_with_batches(self, batch_size: int = 100):
        """Analyze redundancy using batch processing for memory efficiency."""
        print("\n=== Analyzing Redundancy (Batch Mode) ===")

        n_sections = len(self.sections)
        self.total_comparisons = (n_sections * (n_sections - 1)) // 2
        self.start_time = time.time()
        self.comparisons_done = 0

        print(f"Total comparisons to perform: {self.total_comparisons:,}")
        print(f"Batch size: {batch_size}")
        print(f"Minimum similarity threshold: {self.PARITY_THRESHOLD}")

        # Process in batches
        for i in range(0, n_sections, batch_size):
            batch_end = min(i + batch_size, n_sections)
            self._process_batch(i, batch_end, n_sections)

            # Print progress
            elapsed = time.time() - self.start_time
            progress = (self.comparisons_done / self.total_comparisons) * 100
            eta = self._estimate_eta(elapsed, progress)

            print(f"Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta}")

    def _process_batch(self, batch_start: int, batch_end: int, n_sections: int):
        """Process a batch of sections against all other sections."""

        # Get embeddings for this batch
        batch_embeddings = self.embeddings[batch_start:batch_end]

        # Compare batch against all sections (but avoid duplicates)
        for local_idx in range(batch_embeddings.shape[0]):
            i = batch_start + local_idx

            # Only compare with sections after this one to avoid duplicates
            if i + 1 < n_sections:
                remaining_embeddings = self.embeddings[i+1:]

                # Compute similarities (vectorized - very fast!)
                similarities = cosine_similarity(
                    batch_embeddings[local_idx:local_idx+1],
                    remaining_embeddings
                )[0]

                # Process only pairs above minimum threshold (early stopping)
                high_similarity_indices = np.where(similarities >= self.PARITY_THRESHOLD)[0]

                for local_j in high_similarity_indices:
                    j = i + 1 + local_j
                    embedding_sim = similarities[local_j]

                    # Compute text similarity for classification
                    text_sim = self._quick_text_similarity(
                        self.sections[i]['text'],
                        self.sections[j]['text']
                    )

                    max_sim = max(embedding_sim, text_sim)

                    # Categorize the pair
                    self._categorize_pair(i, j, embedding_sim, text_sim, max_sim)

                self.comparisons_done += len(remaining_embeddings)

    def analyze_with_faiss(self, k: int = 50):
        """Analyze redundancy using FAISS approximate nearest neighbors."""
        print("\n=== Analyzing Redundancy (FAISS Mode) ===")

        if not self.use_faiss or self.faiss_index is None:
            print("Error: FAISS index not available")
            return

        n_sections = len(self.sections)
        self.start_time = time.time()

        print(f"Searching for top {k} similar sections per document")
        print(f"This is approximately {k * n_sections:,} comparisons vs {(n_sections * (n_sections - 1)) // 2:,} full comparisons")

        # Search for k nearest neighbors for each section
        dense_embeddings = self.embeddings.toarray().astype('float32')
        import faiss
        faiss.normalize_L2(dense_embeddings)

        # Search all at once (very fast!)
        similarities, indices = self.faiss_index.search(dense_embeddings, k + 1)  # +1 because each doc matches itself

        # Process results
        seen_pairs = set()

        for i in range(n_sections):
            for local_j in range(1, k + 1):  # Skip first result (self)
                j = indices[i][local_j]
                embedding_sim = float(similarities[i][local_j])

                # Skip if below threshold
                if embedding_sim < self.PARITY_THRESHOLD:
                    continue

                # Avoid duplicate pairs
                pair_key = tuple(sorted([i, j]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Compute text similarity
                text_sim = self._quick_text_similarity(
                    self.sections[i]['text'],
                    self.sections[j]['text']
                )

                max_sim = max(embedding_sim, text_sim)
                self._categorize_pair(i, j, embedding_sim, text_sim, max_sim)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{n_sections} sections...")

        elapsed = time.time() - self.start_time
        print(f"✓ FAISS analysis completed in {elapsed:.1f}s")

    def _categorize_pair(self, i: int, j: int, embedding_sim: float,
                        text_sim: float, max_sim: float):
        """Categorize a pair of sections based on similarity."""

        pair_data = {
            'section1_id': self.sections[i]['section_id'],
            'section1_label': self.sections[i]['section_number'],
            'section1_heading': self.sections[i]['subject'],
            'section2_id': self.sections[j]['section_id'],
            'section2_label': self.sections[j]['section_number'],
            'section2_heading': self.sections[j]['subject'],
            'embedding_similarity': float(embedding_sim),
            'text_similarity': float(text_sim),
            'max_similarity': float(max_sim),
            'text1_snippet': self.sections[i]['text'][:300],
            'text2_snippet': self.sections[j]['text'][:300]
        }

        if max_sim >= self.REDUNDANT_THRESHOLD:
            self.redundant_pairs.append(pair_data)
        elif max_sim >= self.OVERLAP_THRESHOLD:
            # Find overlapping chunks for overlap category
            chunks = self._find_overlapping_chunks(
                self.sections[i]['text'],
                self.sections[j]['text']
            )
            pair_data['overlapping_chunks'] = chunks[:5]  # Limit to top 5
            pair_data['num_overlaps'] = len(chunks)
            self.overlapping_pairs.append(pair_data)
        else:  # PARITY_THRESHOLD <= max_sim < OVERLAP_THRESHOLD
            self.parity_pairs.append(pair_data)

    def _estimate_eta(self, elapsed: float, progress: float) -> str:
        """Estimate time remaining."""
        if progress < 1:
            return "calculating..."

        total_time = elapsed / (progress / 100)
        remaining = total_time - elapsed

        if remaining < 60:
            return f"{remaining:.0f}s"
        elif remaining < 3600:
            return f"{remaining/60:.1f}m"
        else:
            return f"{remaining/3600:.1f}h"

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        print("\n=== Generating Report ===")

        # Count sections with issues
        sections_with_issues = set()
        for pair in self.redundant_pairs + self.overlapping_pairs + self.parity_pairs:
            sections_with_issues.add(pair['section1_id'])
            sections_with_issues.add(pair['section2_id'])

        # Count issues per section
        section_counts = defaultdict(lambda: {
            'redundancy_count': 0,
            'overlap_count': 0,
            'parity_count': 0
        })

        for pair in self.redundant_pairs:
            section_counts[pair['section1_id']]['redundancy_count'] += 1
            section_counts[pair['section2_id']]['redundancy_count'] += 1

        for pair in self.overlapping_pairs:
            section_counts[pair['section1_id']]['overlap_count'] += 1
            section_counts[pair['section2_id']]['overlap_count'] += 1

        for pair in self.parity_pairs:
            section_counts[pair['section1_id']]['parity_count'] += 1
            section_counts[pair['section2_id']]['parity_count'] += 1

        # Build section counts list
        section_counts_list = []
        for section in self.sections:
            sid = section['section_id']
            if sid in section_counts:
                counts = section_counts[sid]
                section_counts_list.append({
                    'section_id': sid,
                    'section_label': section['section_number'],
                    'heading': section['subject'],
                    'part': section['part_heading'],
                    'redundancy_count': counts['redundancy_count'],
                    'overlap_count': counts['overlap_count'],
                    'parity_count': counts['parity_count'],
                    'total_similarity_issues': sum(counts.values())
                })

        # Sort by total issues
        section_counts_list.sort(key=lambda x: x['total_similarity_issues'], reverse=True)

        # Generate recommendations
        recommendations = self._generate_recommendations()

        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_sections_analyzed': len(self.sections),
                'embedding_dimensions': self.embeddings.shape[1] if self.embeddings is not None else 0,
                'analysis_method': 'FAISS' if self.use_faiss else 'Batch Processing',
                'thresholds': {
                    'redundant': self.REDUNDANT_THRESHOLD,
                    'overlap': self.OVERLAP_THRESHOLD,
                    'parity': self.PARITY_THRESHOLD
                },
                'performance': {
                    'total_time_seconds': time.time() - self.start_time if self.start_time else 0,
                    'comparisons_performed': self.comparisons_done if not self.use_faiss else 'approximate'
                }
            },
            'summary': {
                'redundant_pairs': len(self.redundant_pairs),
                'overlapping_pairs': len(self.overlapping_pairs),
                'parity_pairs': len(self.parity_pairs),
                'sections_with_issues': len(sections_with_issues),
                'sections_analyzed': len(self.sections)
            },
            'redundant_sections': sorted(self.redundant_pairs,
                                        key=lambda x: x['max_similarity'], reverse=True),
            'overlapping_sections': sorted(self.overlapping_pairs,
                                          key=lambda x: x['max_similarity'], reverse=True),
            'parity_sections': sorted(self.parity_pairs,
                                     key=lambda x: x['max_similarity'], reverse=True),
            'section_counts': section_counts_list[:100],  # Top 100
            'recommendations': recommendations
        }

        print(f"✓ Report generated")
        print(f"  • Redundant pairs: {len(self.redundant_pairs)}")
        print(f"  • Overlapping pairs: {len(self.overlapping_pairs)}")
        print(f"  • Parity pairs: {len(self.parity_pairs)}")
        print(f"  • Sections with issues: {len(sections_with_issues)}")

        return report

    def _generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []

        if self.redundant_pairs:
            recommendations.append({
                'type': 'redundancy',
                'priority': 'HIGH',
                'affected_sections': len(set(
                    [p['section1_id'] for p in self.redundant_pairs] +
                    [p['section2_id'] for p in self.redundant_pairs]
                )),
                'affected_pairs': len(self.redundant_pairs),
                'recommendation': 'Remove or consolidate near-duplicate sections',
                'details': 'These sections have >95% similarity and should be reviewed for consolidation.'
            })

        if self.overlapping_pairs:
            recommendations.append({
                'type': 'overlap',
                'priority': 'MEDIUM',
                'affected_sections': len(set(
                    [p['section1_id'] for p in self.overlapping_pairs] +
                    [p['section2_id'] for p in self.overlapping_pairs]
                )),
                'affected_pairs': len(self.overlapping_pairs),
                'recommendation': 'Extract common content into shared definitions',
                'details': 'These sections share 80-95% content. Consider creating reference sections.'
            })

        if self.parity_pairs:
            recommendations.append({
                'type': 'parity',
                'priority': 'LOW',
                'affected_sections': len(set(
                    [p['section1_id'] for p in self.parity_pairs] +
                    [p['section2_id'] for p in self.parity_pairs]
                )),
                'affected_pairs': len(self.parity_pairs),
                'recommendation': 'Add cross-references between related sections',
                'details': 'These sections are semantically similar (70-80%). Add "See also" references.'
            })

        return recommendations

    def save_report(self, output_path: str = "redundancy_analysis_results.json"):
        """Save analysis report to JSON file."""
        report = self.generate_report()

        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ Report saved to: {output_file}")
        print(f"  File size: {size_mb:.2f} MB")


def main():
    """Main function to run redundancy analysis."""
    script_dir = Path(__file__).parent.resolve()
    db_path = script_dir / "regulations.db"

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        print("Please run database.py first to create the database.")
        sys.exit(1)

    print("=" * 70)
    print("CPSC Regulations Redundancy Analyzer (Optimized)")
    print("=" * 70)

    # Ask user for analysis method
    print("\nChoose analysis method:")
    print("1. Batch processing (accurate, memory efficient)")
    print("2. FAISS approximate search (very fast for large datasets)")

    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    use_faiss = (choice == '2')

    if use_faiss:
        try:
            import faiss
        except ImportError:
            print("\n⚠ FAISS not installed. Falling back to batch processing.")
            print("To use FAISS: pip install faiss-cpu")
            use_faiss = False

    # Initialize analyzer
    analyzer = OptimizedRedundancyAnalyzer(db_path=str(db_path), use_faiss=use_faiss)

    # Load data
    analyzer.load_sections()

    # Create embeddings
    analyzer.create_embeddings()

    # Run analysis
    total_start = time.time()

    if use_faiss:
        analyzer.analyze_with_faiss(k=50)
    else:
        analyzer.analyze_with_batches(batch_size=100)

    total_time = time.time() - total_start

    # Save report
    analyzer.save_report()

    print("\n" + "=" * 70)
    print(f"✓ Analysis completed in {total_time:.1f} seconds")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Review redundancy_analysis_results.json")
    print("2. Start with HIGH priority recommendations")
    print("3. Use section_counts to identify problematic sections")


if __name__ == "__main__":
    main()
