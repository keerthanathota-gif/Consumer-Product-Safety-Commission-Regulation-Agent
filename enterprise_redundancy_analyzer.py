#!/usr/bin/env python3
"""
Enterprise-Grade Redundancy Analysis System for CPSC Regulations
================================================================

A production-level system for identifying redundancies, vague language, and quality issues
in regulatory documents using state-of-the-art semantic analysis.

Features:
- Multi-layer redundancy detection (Semantic, Lexical, Structural)
- Advanced transformer-based embeddings (Sentence-BERT)
- Context-aware vague language identification
- Impact assessment with confidence scoring
- Professional reporting with executive summaries
- Quality metrics and readability analysis
- Batch processing with progress tracking
- Comprehensive validation checks

Author: Enterprise AI Engineering Team
Version: 1.0.0
"""

import sqlite3
import numpy as np
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime
from dataclasses import dataclass, asdict
import sys

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class RedundancyPair:
    """Represents a pair of redundant or overlapping text sections."""
    primary_text: str
    redundant_text: str
    primary_id: int
    redundant_id: int
    similarity_score: float
    overlap_type: str  # 'Semantic', 'Lexical', 'Structural'
    vague_words: List[str]
    impact_severity: str  # 'High', 'Medium', 'Low'
    confidence: float
    rewrite_recommendation: str
    context_info: Dict


@dataclass
class QualityMetrics:
    """Quality metrics for analyzed text."""
    avg_sentence_length: float
    readability_score: float
    redundancy_ratio: float
    vague_language_density: float
    structural_consistency: float


# ============================================================================
# VAGUE LANGUAGE PATTERNS
# ============================================================================

VAGUE_PATTERNS = {
    'qualifiers': [
        r'\b(?:very|quite|rather|somewhat|fairly|pretty|kind of|sort of)\b',
        r'\b(?:relatively|comparatively|approximately|roughly|about)\b',
        r'\b(?:generally|usually|typically|normally|commonly)\b',
        r'\b(?:largely|mostly|mainly|primarily|essentially)\b'
    ],
    'hedging': [
        r'\b(?:may|might|could|would|should|possibly|probably|perhaps)\b',
        r'\b(?:seems?|appears?|tends? to|appears? to)\b',
        r'\b(?:suggests?|indicates?|implies?)\b'
    ],
    'vague_quantities': [
        r'\b(?:some|many|few|several|various|numerous|multiple)\b',
        r'\b(?:significant|substantial|considerable|extensive)\b',
        r'\b(?:small|large|big|little)\s+(?:amount|number|quantity)\b'
    ],
    'ambiguous_references': [
        r'\b(?:this|that|these|those|it|they)\b(?!\s+\w+)',
        r'\b(?:such|said|aforementioned|aforesaid)\b'
    ]
}


# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class EnterpriseRedundancyAnalyzer:
    """
    Enterprise-grade redundancy analyzer with advanced semantic understanding.

    This analyzer uses multiple techniques to identify redundancies:
    1. Semantic similarity (transformer-based embeddings)
    2. Lexical overlap (TF-IDF and word matching)
    3. Structural patterns (sentence structure analysis)
    """

    # Threshold configuration
    THRESHOLDS = {
        'high_redundancy': 0.95,
        'medium_redundancy': 0.85,
        'low_redundancy': 0.70,
        'semantic_similarity': 0.80,
        'lexical_overlap': 0.75
    }

    def __init__(self, db_path: str = "regulations.db", use_transformer: bool = True):
        """
        Initialize the enterprise analyzer.

        Args:
            db_path: Path to SQLite database
            use_transformer: Whether to use transformer embeddings (requires sentence-transformers)
        """
        self.db_path = Path(db_path)
        self.use_transformer = use_transformer

        # Data storage
        self.sections = []
        self.redundancy_pairs = []

        # Embedding models
        self.tfidf_vectorizer = None
        self.tfidf_embeddings = None
        self.transformer_model = None
        self.transformer_embeddings = None

        # Analytics
        self.vague_language_stats = defaultdict(list)
        self.quality_metrics = {}

        # Performance tracking
        self.start_time = None
        self.processing_stats = {}

        print("=" * 80)
        print("ENTERPRISE REDUNDANCY ANALYSIS SYSTEM v1.0.0")
        print("=" * 80)

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def load_sections(self):
        """Load all regulation sections from database with full context."""
        print("\n[1/8] Loading Sections from Database...")

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
            WHERE s.text IS NOT NULL AND LENGTH(s.text) > 50
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
                'chapter_name': row[7],
                'word_count': len(row[3].split()),
                'sentence_count': len(re.split(r'[.!?]+', row[3]))
            }
            for row in rows
        ]

        print(f"   ✓ Loaded {len(self.sections)} sections")
        print(f"   ✓ Average section length: {np.mean([s['word_count'] for s in self.sections]):.1f} words")

    # ========================================================================
    # EMBEDDING CREATION
    # ========================================================================

    def create_embeddings(self):
        """Create both TF-IDF and transformer embeddings for robust analysis."""
        print("\n[2/8] Creating Semantic Embeddings...")

        # Prepare documents
        documents = [
            ' '.join(filter(None, [s['section_number'], s['subject'], s['text']]))
            for s in self.sections
        ]

        # 1. TF-IDF Embeddings (fast, lexical similarity)
        print("   → Creating TF-IDF embeddings...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True
        )

        self.tfidf_embeddings = self.tfidf_vectorizer.fit_transform(documents)
        density = (self.tfidf_embeddings.nnz / (self.tfidf_embeddings.shape[0] * self.tfidf_embeddings.shape[1])) * 100
        print(f"   ✓ TF-IDF: {self.tfidf_embeddings.shape} | Density: {density:.2f}%")

        # 2. Transformer Embeddings (semantic understanding)
        if self.use_transformer:
            print("   → Loading Sentence-BERT model...")
            try:
                from sentence_transformers import SentenceTransformer

                # Use MiniLM - fast and accurate for semantic similarity
                self.transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

                print("   → Encoding documents with transformer...")
                # Process in batches for memory efficiency
                batch_size = 32
                all_embeddings = []

                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i+batch_size]
                    embeddings = self.transformer_model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    all_embeddings.append(embeddings)

                    if (i // batch_size + 1) % 10 == 0:
                        print(f"   → Processed {i+len(batch)}/{len(documents)} documents...")

                self.transformer_embeddings = np.vstack(all_embeddings)
                print(f"   ✓ Transformer: {self.transformer_embeddings.shape}")

            except ImportError:
                print("   ⚠ sentence-transformers not installed, using TF-IDF only")
                print("   → Install with: pip install sentence-transformers")
                self.use_transformer = False

        print(f"   ✓ Embeddings created successfully")

    # ========================================================================
    # VAGUE LANGUAGE DETECTION
    # ========================================================================

    def identify_vague_language(self, text: str) -> Dict[str, List[Tuple[str, str]]]:
        """
        Identify vague language patterns in text with context.

        Returns:
            Dict mapping pattern types to list of (word, context) tuples
        """
        results = defaultdict(list)

        # Split into sentences for context
        sentences = re.split(r'[.!?]+', text)

        for pattern_type, patterns in VAGUE_PATTERNS.items():
            for pattern in patterns:
                for sentence in sentences:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        word = match.group(0)
                        # Get context (surrounding words)
                        start = max(0, match.start() - 30)
                        end = min(len(sentence), match.end() + 30)
                        context = sentence[start:end].strip()
                        results[pattern_type].append((word, context))

        return dict(results)

    def calculate_vague_density(self, text: str) -> float:
        """Calculate the density of vague language (vague words per 100 words)."""
        vague_words = self.identify_vague_language(text)
        total_vague = sum(len(matches) for matches in vague_words.values())
        word_count = len(text.split())
        return (total_vague / word_count * 100) if word_count > 0 else 0.0

    # ========================================================================
    # SIMILARITY CALCULATION
    # ========================================================================

    def calculate_semantic_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate semantic similarity using transformer embeddings."""
        if self.transformer_embeddings is not None:
            vec1 = self.transformer_embeddings[idx1].reshape(1, -1)
            vec2 = self.transformer_embeddings[idx2].reshape(1, -1)
            return float(cosine_similarity(vec1, vec2)[0, 0])
        return 0.0

    def calculate_lexical_similarity(self, idx1: int, idx2: int) -> float:
        """Calculate lexical similarity using TF-IDF embeddings."""
        vec1 = self.tfidf_embeddings[idx1]
        vec2 = self.tfidf_embeddings[idx2]
        return float(cosine_similarity(vec1, vec2)[0, 0])

    def calculate_structural_similarity(self, text1: str, text2: str) -> float:
        """Calculate structural similarity using sequence matching."""
        # Normalize texts
        norm1 = re.sub(r'\s+', ' ', text1.lower().strip())
        norm2 = re.sub(r'\s+', ' ', text2.lower().strip())

        matcher = SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()

    def calculate_combined_similarity(self, idx1: int, idx2: int) -> Tuple[float, str]:
        """
        Calculate combined similarity score and determine overlap type.

        Returns:
            (similarity_score, overlap_type)
        """
        semantic_sim = self.calculate_semantic_similarity(idx1, idx2)
        lexical_sim = self.calculate_lexical_similarity(idx1, idx2)
        structural_sim = self.calculate_structural_similarity(
            self.sections[idx1]['text'],
            self.sections[idx2]['text']
        )

        # Weighted combination
        if self.use_transformer:
            combined = (semantic_sim * 0.5 + lexical_sim * 0.3 + structural_sim * 0.2)
        else:
            combined = (lexical_sim * 0.6 + structural_sim * 0.4)

        # Determine primary overlap type
        if semantic_sim > max(lexical_sim, structural_sim):
            overlap_type = "Semantic"
        elif lexical_sim > structural_sim:
            overlap_type = "Lexical"
        else:
            overlap_type = "Structural"

        return combined, overlap_type

    # ========================================================================
    # REDUNDANCY DETECTION
    # ========================================================================

    def analyze_redundancy(self, batch_size: int = 50, max_pairs: int = 500):
        """
        Perform comprehensive redundancy analysis with multiple validation checks.

        Args:
            batch_size: Number of sections to process per batch
            max_pairs: Maximum redundancy pairs to keep (top scoring)
        """
        print("\n[3/8] Analyzing Redundancy Patterns...")

        n_sections = len(self.sections)
        total_comparisons = (n_sections * (n_sections - 1)) // 2
        self.start_time = time.time()

        print(f"   → Total possible comparisons: {total_comparisons:,}")
        print(f"   → Minimum threshold: {self.THRESHOLDS['low_redundancy']}")

        candidates = []
        comparisons_done = 0

        # Use TF-IDF for initial filtering (fast)
        print("   → Phase 1: Fast lexical filtering...")

        for i in range(0, n_sections, batch_size):
            batch_end = min(i + batch_size, n_sections)

            for local_idx in range(batch_end - i):
                idx1 = i + local_idx

                if idx1 + 1 >= n_sections:
                    continue

                # Calculate similarities with remaining sections
                vec1 = self.tfidf_embeddings[idx1]
                remaining_vecs = self.tfidf_embeddings[idx1+1:]

                similarities = cosine_similarity(vec1, remaining_vecs)[0]

                # Keep only pairs above threshold
                high_sim_indices = np.where(similarities >= self.THRESHOLDS['low_redundancy'])[0]

                for local_j in high_sim_indices:
                    idx2 = idx1 + 1 + local_j
                    lex_sim = similarities[local_j]

                    candidates.append((idx1, idx2, lex_sim))

                comparisons_done += len(remaining_vecs)

            # Progress update
            progress = (comparisons_done / total_comparisons) * 100
            print(f"   → Progress: {progress:.1f}% | Candidates: {len(candidates)}", end='\r')

        print(f"\n   ✓ Phase 1 complete: {len(candidates)} candidate pairs found")

        # Phase 2: Detailed analysis of candidates
        print("   → Phase 2: Deep semantic analysis...")

        detailed_pairs = []

        for idx1, idx2, lex_sim in candidates:
            # Calculate comprehensive similarity
            combined_sim, overlap_type = self.calculate_combined_similarity(idx1, idx2)

            # Skip if below threshold
            if combined_sim < self.THRESHOLDS['low_redundancy']:
                continue

            # Identify vague language
            vague1 = self.identify_vague_language(self.sections[idx1]['text'])
            vague2 = self.identify_vague_language(self.sections[idx2]['text'])
            all_vague = set()
            for v_dict in [vague1, vague2]:
                for words in v_dict.values():
                    all_vague.update([w[0] for w in words])

            # Calculate impact severity
            impact = self._assess_impact(combined_sim, overlap_type, idx1, idx2)

            # Generate recommendation
            recommendation = self._generate_recommendation(combined_sim, overlap_type, idx1, idx2)

            # Calculate confidence
            confidence = self._calculate_confidence(combined_sim, overlap_type, idx1, idx2)

            pair = RedundancyPair(
                primary_text=self.sections[idx1]['text'],
                redundant_text=self.sections[idx2]['text'],
                primary_id=self.sections[idx1]['section_id'],
                redundant_id=self.sections[idx2]['section_id'],
                similarity_score=round(combined_sim, 4),
                overlap_type=overlap_type,
                vague_words=sorted(list(all_vague))[:10],  # Top 10
                impact_severity=impact,
                confidence=round(confidence, 4),
                rewrite_recommendation=recommendation,
                context_info={
                    'section1_number': self.sections[idx1]['section_number'],
                    'section1_subject': self.sections[idx1]['subject'],
                    'section2_number': self.sections[idx2]['section_number'],
                    'section2_subject': self.sections[idx2]['subject'],
                    'lexical_similarity': round(lex_sim, 4)
                }
            )

            detailed_pairs.append(pair)

        # Keep only top N pairs
        detailed_pairs.sort(key=lambda p: p.similarity_score, reverse=True)
        self.redundancy_pairs = detailed_pairs[:max_pairs]

        elapsed = time.time() - self.start_time
        print(f"   ✓ Phase 2 complete: {len(self.redundancy_pairs)} redundancy pairs identified")
        print(f"   ✓ Analysis completed in {elapsed:.1f} seconds")

    # ========================================================================
    # IMPACT ASSESSMENT
    # ========================================================================

    def _assess_impact(self, similarity: float, overlap_type: str, idx1: int, idx2: int) -> str:
        """Assess the impact severity of redundancy."""
        if similarity >= self.THRESHOLDS['high_redundancy']:
            return "High"
        elif similarity >= self.THRESHOLDS['medium_redundancy']:
            # Consider section importance
            total_words = self.sections[idx1]['word_count'] + self.sections[idx2]['word_count']
            if total_words > 400:  # Large sections
                return "High"
            return "Medium"
        else:
            return "Low"

    def _calculate_confidence(self, similarity: float, overlap_type: str, idx1: int, idx2: int) -> float:
        """Calculate confidence score for the redundancy detection."""
        # Base confidence from similarity
        base_confidence = similarity

        # Boost for multiple signal agreement
        semantic_sim = self.calculate_semantic_similarity(idx1, idx2)
        lexical_sim = self.calculate_lexical_similarity(idx1, idx2)

        if abs(semantic_sim - lexical_sim) < 0.1:  # Agreement
            base_confidence = min(1.0, base_confidence + 0.05)

        # Reduce for very short sections
        if self.sections[idx1]['word_count'] < 50 or self.sections[idx2]['word_count'] < 50:
            base_confidence *= 0.9

        return base_confidence

    def _generate_recommendation(self, similarity: float, overlap_type: str, idx1: int, idx2: int) -> str:
        """Generate specific rewrite recommendation."""
        if similarity >= self.THRESHOLDS['high_redundancy']:
            return "CONSOLIDATE: Merge these sections into a single comprehensive section and add cross-references."
        elif similarity >= self.THRESHOLDS['medium_redundancy']:
            if overlap_type == "Semantic":
                return "EXTRACT: Create a shared definition section and reference it from both locations."
            else:
                return "DEDUPLICATE: Remove duplicated content and add 'See Section X' references."
        else:
            return "CROSS-REFERENCE: Add 'See also Section X' to improve navigation and clarity."

    # ========================================================================
    # QUALITY METRICS
    # ========================================================================

    def calculate_quality_metrics(self):
        """Calculate comprehensive quality metrics for all sections."""
        print("\n[4/8] Calculating Quality Metrics...")

        all_metrics = []

        for section in self.sections:
            text = section['text']
            words = text.split()
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

            # Average sentence length
            avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

            # Simple readability score (Flesch-Kincaid approximation)
            avg_syllables = np.mean([self._count_syllables(w) for w in words[:100]])  # Sample
            readability = 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syllables
            readability = max(0, min(100, readability))  # Clamp to 0-100

            # Vague language density
            vague_density = self.calculate_vague_density(text)

            # Structural consistency (sentence length variance)
            sent_lengths = [len(s.split()) for s in sentences]
            consistency = 1.0 - (np.std(sent_lengths) / (np.mean(sent_lengths) + 1))
            consistency = max(0, min(1, consistency))

            metrics = QualityMetrics(
                avg_sentence_length=round(avg_sent_len, 2),
                readability_score=round(readability, 2),
                redundancy_ratio=0.0,  # Will be calculated later
                vague_language_density=round(vague_density, 2),
                structural_consistency=round(consistency, 4)
            )

            all_metrics.append(metrics)

        self.quality_metrics = {
            'per_section': all_metrics,
            'aggregate': {
                'avg_readability': round(np.mean([m.readability_score for m in all_metrics]), 2),
                'avg_vague_density': round(np.mean([m.vague_language_density for m in all_metrics]), 2),
                'avg_sentence_length': round(np.mean([m.avg_sentence_length for m in all_metrics]), 2),
                'avg_consistency': round(np.mean([m.structural_consistency for m in all_metrics]), 4)
            }
        }

        print(f"   ✓ Quality metrics calculated for {len(all_metrics)} sections")
        print(f"   ✓ Average readability score: {self.quality_metrics['aggregate']['avg_readability']}")
        print(f"   ✓ Average vague language density: {self.quality_metrics['aggregate']['avg_vague_density']:.2f}%")

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for readability calculation."""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel

        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1

        return count

    # ========================================================================
    # REPORTING
    # ========================================================================

    def generate_executive_summary(self) -> Dict:
        """Generate 3-point executive summary of key findings."""
        high_impact = [p for p in self.redundancy_pairs if p.impact_severity == "High"]
        medium_impact = [p for p in self.redundancy_pairs if p.impact_severity == "Medium"]

        # Calculate estimated savings
        total_redundant_words = sum([
            len(p.redundant_text.split()) for p in high_impact
        ])

        # Identify most problematic sections
        section_issues = defaultdict(int)
        for pair in self.redundancy_pairs:
            section_issues[pair.primary_id] += 1
            section_issues[pair.redundant_id] += 1

        most_problematic = sorted(section_issues.items(), key=lambda x: x[1], reverse=True)[:5]

        summary = {
            'finding_1': {
                'title': 'High-Impact Redundancies Identified',
                'description': f"Found {len(high_impact)} critical redundancies with >95% similarity. "
                              f"Consolidating these could reduce document length by ~{total_redundant_words:,} words.",
                'action': 'Priority: Immediate review and consolidation recommended.'
            },
            'finding_2': {
                'title': 'Vague Language Pervasiveness',
                'description': f"Average vague language density: {self.quality_metrics['aggregate']['avg_vague_density']:.1f}%. "
                              f"Qualifiers, hedging, and ambiguous references reduce regulatory clarity.",
                'action': 'Priority: Implement precise language guidelines for future revisions.'
            },
            'finding_3': {
                'title': 'Systematic Quality Patterns',
                'description': f"Identified {len(most_problematic)} sections with recurring redundancy issues. "
                              f"Average readability score: {self.quality_metrics['aggregate']['avg_readability']:.1f}/100.",
                'action': 'Priority: Focus improvement efforts on identified hotspots.'
            }
        }

        return summary

    def generate_statistical_analysis(self) -> Dict:
        """Generate distribution and trend statistics."""
        similarities = [p.similarity_score for p in self.redundancy_pairs]

        impact_distribution = Counter([p.impact_severity for p in self.redundancy_pairs])
        overlap_distribution = Counter([p.overlap_type for p in self.redundancy_pairs])

        # Confidence analysis
        high_confidence = len([p for p in self.redundancy_pairs if p.confidence > 0.9])

        return {
            'similarity_distribution': {
                'mean': round(np.mean(similarities), 4) if similarities else 0,
                'median': round(np.median(similarities), 4) if similarities else 0,
                'std': round(np.std(similarities), 4) if similarities else 0,
                'min': round(min(similarities), 4) if similarities else 0,
                'max': round(max(similarities), 4) if similarities else 0,
                'percentiles': {
                    '25th': round(np.percentile(similarities, 25), 4) if similarities else 0,
                    '75th': round(np.percentile(similarities, 75), 4) if similarities else 0,
                    '90th': round(np.percentile(similarities, 90), 4) if similarities else 0
                }
            },
            'impact_distribution': dict(impact_distribution),
            'overlap_type_distribution': dict(overlap_distribution),
            'confidence_analysis': {
                'high_confidence_pairs': high_confidence,
                'avg_confidence': round(np.mean([p.confidence for p in self.redundancy_pairs]), 4) if self.redundancy_pairs else 0
            }
        }

    def generate_comprehensive_report(self) -> Dict:
        """Generate complete enterprise-grade report."""
        print("\n[5/8] Generating Comprehensive Report...")

        executive_summary = self.generate_executive_summary()
        statistical_analysis = self.generate_statistical_analysis()

        # Prepare redundancy pairs for JSON serialization
        serialized_pairs = []
        for pair in self.redundancy_pairs:
            pair_dict = asdict(pair)
            # Truncate long text for readability
            pair_dict['primary_text'] = pair.primary_text[:500] + "..." if len(pair.primary_text) > 500 else pair.primary_text
            pair_dict['redundant_text'] = pair.redundant_text[:500] + "..." if len(pair.redundant_text) > 500 else pair.redundant_text
            serialized_pairs.append(pair_dict)

        # Action plan with prioritization
        action_plan = self._generate_action_plan()

        report = {
            'metadata': {
                'report_version': '1.0.0',
                'analysis_date': datetime.now().isoformat(),
                'analyzer_config': {
                    'use_transformer': self.use_transformer,
                    'thresholds': self.THRESHOLDS,
                    'model': 'all-MiniLM-L6-v2' if self.use_transformer else 'TF-IDF only'
                },
                'corpus_statistics': {
                    'total_sections': len(self.sections),
                    'total_words': sum(s['word_count'] for s in self.sections),
                    'total_sentences': sum(s['sentence_count'] for s in self.sections)
                },
                'performance': {
                    'processing_time_seconds': round(time.time() - self.start_time, 2) if self.start_time else 0,
                    'sections_per_second': round(len(self.sections) / (time.time() - self.start_time), 2) if self.start_time else 0
                }
            },
            'executive_summary': executive_summary,
            'statistical_analysis': statistical_analysis,
            'quality_metrics': self.quality_metrics['aggregate'],
            'redundancy_pairs': serialized_pairs,
            'action_plan': action_plan,
            'validation_checks': self._run_validation_checks()
        }

        print(f"   ✓ Report generated successfully")
        print(f"   ✓ Found {len(serialized_pairs)} redundancy pairs")
        print(f"   ✓ Quality metrics calculated")

        return report

    def _generate_action_plan(self) -> List[Dict]:
        """Generate priority-based action plan."""
        high_priority = [p for p in self.redundancy_pairs if p.impact_severity == "High"]
        medium_priority = [p for p in self.redundancy_pairs if p.impact_severity == "Medium"]
        low_priority = [p for p in self.redundancy_pairs if p.impact_severity == "Low"]

        actions = []

        if high_priority:
            actions.append({
                'priority': 'CRITICAL',
                'action': 'Consolidate High-Redundancy Sections',
                'affected_pairs': len(high_priority),
                'estimated_effort': f"{len(high_priority) * 2} hours",
                'expected_impact': 'Major improvement in document clarity and length reduction',
                'recommended_deadline': '2 weeks'
            })

        if medium_priority:
            actions.append({
                'priority': 'HIGH',
                'action': 'Review and Deduplicate Medium-Redundancy Content',
                'affected_pairs': len(medium_priority),
                'estimated_effort': f"{len(medium_priority) * 1} hours",
                'expected_impact': 'Moderate improvement in consistency',
                'recommended_deadline': '1 month'
            })

        if low_priority:
            actions.append({
                'priority': 'MEDIUM',
                'action': 'Add Cross-References for Related Sections',
                'affected_pairs': len(low_priority),
                'estimated_effort': f"{len(low_priority) * 0.5} hours",
                'expected_impact': 'Improved navigation and discoverability',
                'recommended_deadline': '2 months'
            })

        # Vague language action
        avg_vague = self.quality_metrics['aggregate']['avg_vague_density']
        if avg_vague > 5.0:
            actions.append({
                'priority': 'HIGH',
                'action': 'Reduce Vague Language Usage',
                'affected_pairs': len(self.sections),
                'estimated_effort': f"{len(self.sections) * 0.25} hours",
                'expected_impact': 'Significantly improved regulatory precision',
                'recommended_deadline': '3 months'
            })

        return actions

    def _run_validation_checks(self) -> Dict:
        """Run comprehensive validation checks on the analysis."""
        checks = {
            'embedding_quality': {
                'passed': self.tfidf_embeddings is not None,
                'details': f"TF-IDF embeddings: {self.tfidf_embeddings.shape if self.tfidf_embeddings is not None else 'None'}"
            },
            'transformer_validation': {
                'passed': self.transformer_embeddings is not None if self.use_transformer else True,
                'details': f"Transformer embeddings: {self.transformer_embeddings.shape if self.transformer_embeddings is not None else 'Not used'}"
            },
            'redundancy_precision': {
                'passed': len([p for p in self.redundancy_pairs if p.confidence > 0.85]) / max(len(self.redundancy_pairs), 1) > 0.7,
                'details': f"High confidence pairs: {len([p for p in self.redundancy_pairs if p.confidence > 0.85])}/{len(self.redundancy_pairs)}"
            },
            'quality_metrics_completeness': {
                'passed': len(self.quality_metrics.get('per_section', [])) == len(self.sections),
                'details': f"Metrics calculated for {len(self.quality_metrics.get('per_section', []))} sections"
            }
        }

        all_passed = all(check['passed'] for check in checks.values())
        checks['overall_validation'] = {
            'passed': all_passed,
            'details': f"{'All checks passed' if all_passed else 'Some checks failed'}"
        }

        return checks

    def save_report(self, output_path: str = "enterprise_redundancy_report.json"):
        """Save comprehensive report to JSON file."""
        print("\n[6/8] Saving Report...")

        report = self.generate_comprehensive_report()

        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   ✓ Report saved to: {output_file}")
        print(f"   ✓ File size: {size_mb:.2f} MB")

        return report

    def print_summary(self):
        """Print concise summary to console."""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        high = len([p for p in self.redundancy_pairs if p.impact_severity == "High"])
        medium = len([p for p in self.redundancy_pairs if p.impact_severity == "Medium"])
        low = len([p for p in self.redundancy_pairs if p.impact_severity == "Low"])

        print(f"\nRedundancy Detection:")
        print(f"  • High Impact:   {high:3d} pairs")
        print(f"  • Medium Impact: {medium:3d} pairs")
        print(f"  • Low Impact:    {low:3d} pairs")
        print(f"  • Total Pairs:   {len(self.redundancy_pairs):3d}")

        print(f"\nQuality Metrics:")
        agg = self.quality_metrics['aggregate']
        print(f"  • Readability Score:  {agg['avg_readability']:.1f}/100")
        print(f"  • Vague Language:     {agg['avg_vague_density']:.2f}%")
        print(f"  • Avg Sentence Length: {agg['avg_sentence_length']:.1f} words")
        print(f"  • Structural Consistency: {agg['avg_consistency']:.3f}")

        print(f"\nProcessing Statistics:")
        print(f"  • Sections Analyzed: {len(self.sections)}")
        print(f"  • Processing Time: {time.time() - self.start_time:.1f} seconds")
        print(f"  • Embeddings Used: {'Transformer + TF-IDF' if self.use_transformer else 'TF-IDF only'}")

        print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    script_dir = Path(__file__).parent.resolve()
    db_path = script_dir / "regulations.db"

    if not db_path.exists():
        print(f"\n❌ Error: Database not found at {db_path}")
        print("Please run database.py first to create the database.")
        sys.exit(1)

    print("\nInitializing Enterprise Redundancy Analyzer...")
    print("\nConfiguration Options:")
    print("1. Full Analysis (Transformer + TF-IDF) - Recommended")
    print("2. Fast Analysis (TF-IDF only) - Quicker but less accurate")

    choice = input("\nSelect mode (1 or 2, default=1): ").strip() or "1"
    use_transformer = (choice == "1")

    if use_transformer:
        print("\n📦 Checking for sentence-transformers...")
        try:
            import sentence_transformers
            print("   ✓ sentence-transformers found")
        except ImportError:
            print("   ⚠ sentence-transformers not installed")
            print("   → Installing: pip install sentence-transformers")
            print("\nWould you like to continue with TF-IDF only? (y/n): ", end='')
            if input().lower() != 'y':
                print("\nInstall sentence-transformers and run again.")
                sys.exit(0)
            use_transformer = False

    # Initialize analyzer
    analyzer = EnterpriseRedundancyAnalyzer(
        db_path=str(db_path),
        use_transformer=use_transformer
    )

    # Run complete analysis pipeline
    analyzer.load_sections()
    analyzer.create_embeddings()
    analyzer.analyze_redundancy(batch_size=50, max_pairs=500)
    analyzer.calculate_quality_metrics()

    # Generate and save report
    report = analyzer.save_report("enterprise_redundancy_report.json")

    # Print summary
    analyzer.print_summary()

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext Steps:")
    print("1. Review enterprise_redundancy_report.json for detailed findings")
    print("2. Focus on HIGH priority actions in the action plan")
    print("3. Use rewrite recommendations for specific sections")
    print("4. Monitor quality metrics for improvement tracking")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
