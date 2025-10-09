#!/usr/bin/env python3
"""
CPSC Regulations Redundancy Analyzer using Embeddings
Detects redundant, overlapping, and parity sections using semantic embeddings.
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import re
from difflib import SequenceMatcher

class RegulationRedundancyAnalyzer:
    """Analyzer for detecting redundancy, overlap, and parity in regulations."""

    def __init__(self, db_path='regulations.db'):
        self.db_path = db_path
        self.sections = []
        self.embeddings = []

        # Thresholds for classification
        self.REDUNDANT_THRESHOLD = 0.95  # Near-duplicate (>95% similar)
        self.OVERLAP_THRESHOLD = 0.80    # Partial overlap (80-95% similar)
        self.PARITY_THRESHOLD = 0.70     # Semantic parity (70-80% similar)

    def load_sections(self):
        """Load all sections from the database with their metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                s.section_id,
                c.chapter_name,
                sc.subchapter_name,
                p.heading AS part_heading,
                s.section_number,
                s.subject AS heading,
                s.text,
                s.citation
            FROM sections s
            JOIN parts p ON s.part_id = p.part_id
            JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
            JOIN chapters c ON sc.chapter_id = c.chapter_id
            WHERE s.text IS NOT NULL AND s.text != ''
        """)

        for row in cursor.fetchall():
            self.sections.append({
                'section_id': row[0],
                'year': '2025',  # From CFR-2025 file
                'title': '16',   # Title 16
                'chapter': row[1],
                'subchapter': row[2],
                'part': row[3],
                'section_label': row[4],
                'heading': row[5],
                'text': row[6],
                'citations': row[7] if row[7] else ''
            })

        conn.close()
        print(f"Loaded {len(self.sections)} sections from database")

    def create_embeddings(self):
        """
        Create embeddings for all sections.
        For this implementation, we'll use TF-IDF-based embeddings.
        In production, use sentence-transformers or OpenAI embeddings.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [self._get_full_text(section) for section in self.sections]

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )

        self.embeddings = vectorizer.fit_transform(texts).toarray()
        print(f"Created embeddings with shape {self.embeddings.shape}")

        return vectorizer

    def _get_full_text(self, section):
        """Get full text including heading for embedding."""
        return f"{section['heading']} {section['text']}"

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def calculate_text_similarity(self, text1, text2):
        """Calculate direct text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def find_overlapping_chunks(self, text1, text2, min_chunk_size=50):
        """Find overlapping text chunks between two sections."""
        words1 = text1.split()
        words2 = text2.split()

        overlaps = []

        # Use sliding window to find common chunks
        for chunk_size in range(min_chunk_size, min(len(words1), len(words2)) + 1):
            for i in range(len(words1) - chunk_size + 1):
                chunk1 = ' '.join(words1[i:i + chunk_size])

                for j in range(len(words2) - chunk_size + 1):
                    chunk2 = ' '.join(words2[j:j + chunk_size])

                    similarity = self.calculate_text_similarity(chunk1, chunk2)

                    if similarity > 0.85:  # High chunk similarity
                        overlaps.append({
                            'text': chunk1[:200] + '...' if len(chunk1) > 200 else chunk1,
                            'length': len(chunk1.split()),
                            'similarity': similarity,
                            'position1': i,
                            'position2': j
                        })

        # Remove duplicate overlaps (keep largest)
        unique_overlaps = []
        seen_positions = set()

        for overlap in sorted(overlaps, key=lambda x: x['length'], reverse=True):
            pos_key = (overlap['position1'], overlap['position2'])
            if pos_key not in seen_positions:
                unique_overlaps.append(overlap)
                seen_positions.add(pos_key)

        return unique_overlaps[:5]  # Return top 5 overlaps

    def analyze_redundancy(self):
        """
        Perform comprehensive redundancy analysis.
        Returns redundant, overlapping, and parity sections.
        """
        print("\nAnalyzing section similarities...")

        redundant_pairs = []
        overlapping_pairs = []
        parity_pairs = []

        n = len(self.sections)

        # Compare each pair of sections
        for i in range(n):
            if i % 100 == 0:
                print(f"  Processing section {i}/{n}...")

            for j in range(i + 1, n):
                # Calculate embedding similarity
                embedding_sim = self.cosine_similarity(
                    self.embeddings[i],
                    self.embeddings[j]
                )

                # Calculate text similarity for verification
                text_sim = self.calculate_text_similarity(
                    self.sections[i]['text'],
                    self.sections[j]['text']
                )

                # Use the maximum of both similarities
                max_similarity = max(embedding_sim, text_sim)

                section_pair = {
                    'section1_id': self.sections[i]['section_id'],
                    'section1_label': self.sections[i]['section_label'],
                    'section1_heading': self.sections[i]['heading'],
                    'section1_part': self.sections[i]['part'],
                    'section2_id': self.sections[j]['section_id'],
                    'section2_label': self.sections[j]['section_label'],
                    'section2_heading': self.sections[j]['heading'],
                    'section2_part': self.sections[j]['part'],
                    'embedding_similarity': float(embedding_sim),
                    'text_similarity': float(text_sim),
                    'max_similarity': float(max_similarity)
                }

                # Classify by similarity level
                if max_similarity >= self.REDUNDANT_THRESHOLD:
                    # Near-duplicate sections
                    section_pair['text1_snippet'] = self.sections[i]['text'][:300]
                    section_pair['text2_snippet'] = self.sections[j]['text'][:300]
                    redundant_pairs.append(section_pair)

                elif max_similarity >= self.OVERLAP_THRESHOLD:
                    # Overlapping sections - find specific overlaps
                    overlaps = self.find_overlapping_chunks(
                        self.sections[i]['text'],
                        self.sections[j]['text']
                    )
                    section_pair['overlapping_chunks'] = overlaps
                    section_pair['num_overlaps'] = len(overlaps)
                    overlapping_pairs.append(section_pair)

                elif max_similarity >= self.PARITY_THRESHOLD:
                    # Parity sections (semantic similarity)
                    section_pair['text1_snippet'] = self.sections[i]['text'][:200]
                    section_pair['text2_snippet'] = self.sections[j]['text'][:200]
                    parity_pairs.append(section_pair)

        print(f"\nFound:")
        print(f"  - {len(redundant_pairs)} redundant pairs")
        print(f"  - {len(overlapping_pairs)} overlapping pairs")
        print(f"  - {len(parity_pairs)} parity pairs")

        return redundant_pairs, overlapping_pairs, parity_pairs

    def calculate_section_counts(self, redundant_pairs, overlapping_pairs, parity_pairs):
        """Calculate counts per section."""
        counts = defaultdict(lambda: {
            'redundancy_count': 0,
            'overlap_count': 0,
            'parity_count': 0,
            'total_similarity_issues': 0
        })

        # Count redundancies
        for pair in redundant_pairs:
            counts[pair['section1_id']]['redundancy_count'] += 1
            counts[pair['section2_id']]['redundancy_count'] += 1

        # Count overlaps
        for pair in overlapping_pairs:
            counts[pair['section1_id']]['overlap_count'] += 1
            counts[pair['section2_id']]['overlap_count'] += 1

        # Count parity
        for pair in parity_pairs:
            counts[pair['section1_id']]['parity_count'] += 1
            counts[pair['section2_id']]['parity_count'] += 1

        # Calculate totals
        for section_id in counts:
            c = counts[section_id]
            c['total_similarity_issues'] = (
                c['redundancy_count'] +
                c['overlap_count'] +
                c['parity_count']
            )

        # Add section metadata
        section_map = {s['section_id']: s for s in self.sections}

        counts_with_metadata = []
        for section_id, count_data in counts.items():
            section = section_map.get(section_id, {})
            counts_with_metadata.append({
                'section_id': section_id,
                'section_label': section.get('section_label', ''),
                'heading': section.get('heading', ''),
                'part': section.get('part', ''),
                **count_data
            })

        # Sort by total issues descending
        counts_with_metadata.sort(
            key=lambda x: x['total_similarity_issues'],
            reverse=True
        )

        return counts_with_metadata

    def generate_recommendations(self, redundant_pairs, overlapping_pairs, parity_pairs, section_counts):
        """Generate actionable recommendations for consolidation."""
        recommendations = []

        # 1. Redundant sections - recommend removal or consolidation
        if redundant_pairs:
            redundant_sections = set()
            for pair in redundant_pairs:
                redundant_sections.add(pair['section1_id'])
                redundant_sections.add(pair['section2_id'])

            recommendations.append({
                'type': 'redundancy',
                'priority': 'HIGH',
                'affected_sections': len(redundant_sections),
                'recommendation': 'Remove or consolidate near-duplicate sections',
                'details': [
                    {
                        'section1': f"{pair['section1_label']} - {pair['section1_heading'][:50]}",
                        'section2': f"{pair['section2_label']} - {pair['section2_heading'][:50]}",
                        'similarity': f"{pair['max_similarity']:.2%}",
                        'action': 'Consider keeping only one section or consolidating both into a single, comprehensive section'
                    }
                    for pair in sorted(redundant_pairs, key=lambda x: x['max_similarity'], reverse=True)[:10]
                ]
            })

        # 2. Overlapping sections - recommend extracting common content
        if overlapping_pairs:
            recommendations.append({
                'type': 'overlap',
                'priority': 'MEDIUM',
                'affected_sections': len(set(
                    [p['section1_id'] for p in overlapping_pairs] +
                    [p['section2_id'] for p in overlapping_pairs]
                )),
                'recommendation': 'Extract common overlapping content into shared definitions or base sections',
                'details': [
                    {
                        'section1': f"{pair['section1_label']} - {pair['section1_heading'][:50]}",
                        'section2': f"{pair['section2_label']} - {pair['section2_heading'][:50]}",
                        'overlap_count': pair['num_overlaps'],
                        'similarity': f"{pair['max_similarity']:.2%}",
                        'action': 'Extract common text into a referenced definition section'
                    }
                    for pair in sorted(overlapping_pairs, key=lambda x: x['num_overlaps'], reverse=True)[:10]
                ]
            })

        # 3. Parity sections - recommend cross-referencing
        if parity_pairs:
            recommendations.append({
                'type': 'parity',
                'priority': 'LOW',
                'affected_sections': len(set(
                    [p['section1_id'] for p in parity_pairs] +
                    [p['section2_id'] for p in parity_pairs]
                )),
                'recommendation': 'Add cross-references between semantically similar sections',
                'details': [
                    {
                        'section1': f"{pair['section1_label']} - {pair['section1_heading'][:50]}",
                        'section2': f"{pair['section2_label']} - {pair['section2_heading'][:50]}",
                        'similarity': f"{pair['max_similarity']:.2%}",
                        'action': 'Add "See also" cross-references to improve navigation'
                    }
                    for pair in sorted(parity_pairs, key=lambda x: x['max_similarity'], reverse=True)[:10]
                ]
            })

        # 4. Sections with multiple issues
        high_issue_sections = [s for s in section_counts if s['total_similarity_issues'] >= 5]
        if high_issue_sections:
            recommendations.append({
                'type': 'multiple_issues',
                'priority': 'HIGH',
                'affected_sections': len(high_issue_sections),
                'recommendation': 'Review sections with multiple similarity issues for potential restructuring',
                'details': [
                    {
                        'section': f"{s['section_label']} - {s['heading'][:50]}",
                        'redundancy': s['redundancy_count'],
                        'overlap': s['overlap_count'],
                        'parity': s['parity_count'],
                        'total': s['total_similarity_issues'],
                        'action': 'High-priority review needed - may indicate core definitions that should be centralized'
                    }
                    for s in high_issue_sections[:15]
                ]
            })

        return recommendations

    def run_full_analysis(self, output_file='redundancy_analysis_results.json'):
        """Run complete analysis and save results."""
        print("=" * 80)
        print("CPSC Regulations Redundancy Analysis (Embedding-Based)")
        print("=" * 80)

        # Load data
        self.load_sections()

        # Create embeddings
        print("\nCreating embeddings...")
        self.create_embeddings()

        # Analyze redundancy
        redundant_pairs, overlapping_pairs, parity_pairs = self.analyze_redundancy()

        # Calculate section counts
        print("\nCalculating per-section counts...")
        section_counts = self.calculate_section_counts(
            redundant_pairs, overlapping_pairs, parity_pairs
        )

        # Generate recommendations
        print("\nGenerating recommendations...")
        recommendations = self.generate_recommendations(
            redundant_pairs, overlapping_pairs, parity_pairs, section_counts
        )

        # Compile results
        results = {
            'metadata': {
                'total_sections_analyzed': len(self.sections),
                'embedding_dimensions': self.embeddings.shape[1],
                'thresholds': {
                    'redundant': self.REDUNDANT_THRESHOLD,
                    'overlap': self.OVERLAP_THRESHOLD,
                    'parity': self.PARITY_THRESHOLD
                }
            },
            'summary': {
                'redundant_pairs': len(redundant_pairs),
                'overlapping_pairs': len(overlapping_pairs),
                'parity_pairs': len(parity_pairs),
                'sections_with_issues': len(section_counts)
            },
            'redundant_sections': redundant_pairs,
            'overlapping_sections': overlapping_pairs,
            'parity_sections': parity_pairs,
            'section_counts': section_counts,
            'recommendations': recommendations
        }

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 80}")
        print(f"Analysis complete! Results saved to: {output_file}")
        print(f"{'=' * 80}")

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results):
        """Print analysis summary."""
        print("\nSUMMARY:")
        print("-" * 80)
        print(f"Total sections analyzed: {results['metadata']['total_sections_analyzed']}")
        print(f"\nRedundant pairs (≥{results['metadata']['thresholds']['redundant']:.0%} similar): {results['summary']['redundant_pairs']}")
        print(f"Overlapping pairs ({results['metadata']['thresholds']['overlap']:.0%}-{results['metadata']['thresholds']['redundant']:.0%} similar): {results['summary']['overlapping_pairs']}")
        print(f"Parity pairs ({results['metadata']['thresholds']['parity']:.0%}-{results['metadata']['thresholds']['overlap']:.0%} similar): {results['summary']['parity_pairs']}")
        print(f"Sections with similarity issues: {results['summary']['sections_with_issues']}")

        print(f"\nRecommendations generated: {len(results['recommendations'])}")
        for rec in results['recommendations']:
            print(f"  - [{rec['priority']}] {rec['type'].upper()}: {rec['recommendation']}")


def main():
    """Main entry point."""
    try:
        analyzer = RegulationRedundancyAnalyzer('regulations.db')
        results = analyzer.run_full_analysis('redundancy_analysis_results.json')
        return results
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
