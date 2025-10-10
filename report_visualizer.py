#!/usr/bin/env python3
"""
Enterprise Report Visualizer
============================

Creates professional visualizations from the enterprise redundancy analysis report.
Generates charts, graphs, and insights for executive presentations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ReportVisualizer:
    """Generate visualizations from enterprise redundancy report."""

    def __init__(self, report_path: str = "enterprise_redundancy_report.json"):
        """Initialize visualizer with report data."""
        self.report_path = Path(report_path)
        self.report = self.load_report()

        if not self.report:
            raise ValueError(f"Could not load report from {report_path}")

        print(f"✓ Loaded report from {self.report_path}")

    def load_report(self) -> Dict:
        """Load the JSON report."""
        if not self.report_path.exists():
            print(f"Error: Report not found at {self.report_path}")
            return {}

        with open(self.report_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_similarity_distribution(self, output_path: str = "similarity_distribution.png"):
        """Create histogram of similarity scores."""
        pairs = self.report.get('redundancy_pairs', [])
        if not pairs:
            print("No redundancy pairs to visualize")
            return

        similarities = [p['similarity_score'] for p in pairs]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Histogram
        ax.hist(similarities, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=0.95, color='red', linestyle='--', linewidth=2, label='High Redundancy (0.95)')
        ax.axvline(x=0.85, color='orange', linestyle='--', linewidth=2, label='Medium Redundancy (0.85)')
        ax.axvline(x=0.70, color='green', linestyle='--', linewidth=2, label='Low Redundancy (0.70)')

        ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Pairs', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Redundancy Similarity Scores', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved similarity distribution: {output_path}")
        plt.close()

    def create_impact_severity_chart(self, output_path: str = "impact_severity.png"):
        """Create pie chart of impact severity distribution."""
        pairs = self.report.get('redundancy_pairs', [])
        if not pairs:
            return

        severity_counts = Counter([p['impact_severity'] for p in pairs])

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = {'High': '#d9534f', 'Medium': '#f0ad4e', 'Low': '#5bc0de'}
        sizes = [severity_counts.get('High', 0), severity_counts.get('Medium', 0), severity_counts.get('Low', 0)]
        labels = [f"High\n({severity_counts.get('High', 0)} pairs)",
                  f"Medium\n({severity_counts.get('Medium', 0)} pairs)",
                  f"Low\n({severity_counts.get('Low', 0)} pairs)"]
        color_list = [colors['High'], colors['Medium'], colors['Low']]

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=color_list,
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})

        ax.set_title('Redundancy Impact Severity Distribution', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved impact severity chart: {output_path}")
        plt.close()

    def create_overlap_type_chart(self, output_path: str = "overlap_types.png"):
        """Create bar chart of overlap types."""
        pairs = self.report.get('redundancy_pairs', [])
        if not pairs:
            return

        overlap_counts = Counter([p['overlap_type'] for p in pairs])

        fig, ax = plt.subplots(figsize=(10, 6))

        types = list(overlap_counts.keys())
        counts = list(overlap_counts.values())
        colors_map = {'Semantic': '#5cb85c', 'Lexical': '#5bc0de', 'Structural': '#f0ad4e'}
        bar_colors = [colors_map.get(t, 'gray') for t in types]

        bars = ax.bar(types, counts, color=bar_colors, edgecolor='black', alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.set_xlabel('Overlap Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Pairs', fontsize=12, fontweight='bold')
        ax.set_title('Redundancy Overlap Type Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved overlap type chart: {output_path}")
        plt.close()

    def create_quality_metrics_dashboard(self, output_path: str = "quality_dashboard.png"):
        """Create dashboard of quality metrics."""
        metrics = self.report.get('quality_metrics', {})
        if not metrics:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Document Quality Metrics Dashboard', fontsize=16, fontweight='bold', y=0.995)

        # 1. Readability Score
        ax1 = axes[0, 0]
        readability = metrics.get('avg_readability', 0)
        colors_read = ['#d9534f' if readability < 30 else '#f0ad4e' if readability < 60 else '#5cb85c']
        ax1.barh(['Readability'], [readability], color=colors_read, edgecolor='black', height=0.5)
        ax1.set_xlim(0, 100)
        ax1.set_xlabel('Score (0-100)', fontweight='bold')
        ax1.set_title('Average Readability Score', fontweight='bold')
        ax1.text(readability + 3, 0, f'{readability:.1f}', va='center', fontweight='bold', fontsize=12)
        ax1.grid(axis='x', alpha=0.3)

        # 2. Vague Language Density
        ax2 = axes[0, 1]
        vague = metrics.get('avg_vague_density', 0)
        colors_vague = ['#5cb85c' if vague < 3 else '#f0ad4e' if vague < 7 else '#d9534f']
        ax2.barh(['Vague Language'], [vague], color=colors_vague, edgecolor='black', height=0.5)
        ax2.set_xlim(0, 15)
        ax2.set_xlabel('Density (%)', fontweight='bold')
        ax2.set_title('Average Vague Language Density', fontweight='bold')
        ax2.text(vague + 0.3, 0, f'{vague:.2f}%', va='center', fontweight='bold', fontsize=12)
        ax2.grid(axis='x', alpha=0.3)

        # 3. Sentence Length
        ax3 = axes[1, 0]
        sent_len = metrics.get('avg_sentence_length', 0)
        colors_sent = ['#5cb85c' if sent_len < 20 else '#f0ad4e' if sent_len < 30 else '#d9534f']
        ax3.barh(['Sentence Length'], [sent_len], color=colors_sent, edgecolor='black', height=0.5)
        ax3.set_xlim(0, 50)
        ax3.set_xlabel('Words per Sentence', fontweight='bold')
        ax3.set_title('Average Sentence Length', fontweight='bold')
        ax3.text(sent_len + 1.5, 0, f'{sent_len:.1f}', va='center', fontweight='bold', fontsize=12)
        ax3.grid(axis='x', alpha=0.3)

        # 4. Structural Consistency
        ax4 = axes[1, 1]
        consistency = metrics.get('avg_consistency', 0)
        colors_cons = ['#d9534f' if consistency < 0.5 else '#f0ad4e' if consistency < 0.7 else '#5cb85c']
        ax4.barh(['Consistency'], [consistency], color=colors_cons, edgecolor='black', height=0.5)
        ax4.set_xlim(0, 1)
        ax4.set_xlabel('Score (0-1)', fontweight='bold')
        ax4.set_title('Structural Consistency', fontweight='bold')
        ax4.text(consistency + 0.03, 0, f'{consistency:.3f}', va='center', fontweight='bold', fontsize=12)
        ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved quality dashboard: {output_path}")
        plt.close()

    def create_confidence_analysis(self, output_path: str = "confidence_analysis.png"):
        """Create confidence score analysis."""
        pairs = self.report.get('redundancy_pairs', [])
        if not pairs:
            return

        confidences = [p['confidence'] for p in pairs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Confidence Analysis', fontsize=14, fontweight='bold')

        # Histogram
        ax1.hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='purple')
        ax1.axvline(x=0.9, color='green', linestyle='--', linewidth=2, label='High Confidence (0.9)')
        ax1.set_xlabel('Confidence Score', fontweight='bold')
        ax1.set_ylabel('Number of Pairs', fontweight='bold')
        ax1.set_title('Confidence Score Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Box plot by severity
        high_conf = [p['confidence'] for p in pairs if p['impact_severity'] == 'High']
        medium_conf = [p['confidence'] for p in pairs if p['impact_severity'] == 'Medium']
        low_conf = [p['confidence'] for p in pairs if p['impact_severity'] == 'Low']

        box_data = [high_conf, medium_conf, low_conf]
        bp = ax2.boxplot(box_data, labels=['High', 'Medium', 'Low'],
                         patch_artist=True, showmeans=True)

        colors = ['#d9534f', '#f0ad4e', '#5bc0de']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_xlabel('Impact Severity', fontweight='bold')
        ax2.set_ylabel('Confidence Score', fontweight='bold')
        ax2.set_title('Confidence by Impact Severity', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confidence analysis: {output_path}")
        plt.close()

    def create_vague_words_wordcloud(self, output_path: str = "vague_words_top20.png"):
        """Create bar chart of most common vague words."""
        pairs = self.report.get('redundancy_pairs', [])
        if not pairs:
            return

        # Collect all vague words
        all_vague = []
        for pair in pairs:
            all_vague.extend(pair.get('vague_words', []))

        if not all_vague:
            print("No vague words found")
            return

        # Count frequency
        vague_counts = Counter(all_vague)
        top_20 = vague_counts.most_common(20)

        words, counts = zip(*top_20)

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.barh(words, counts, color='coral', edgecolor='black', alpha=0.8)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2.,
                   f'{int(width)}',
                   ha='left', va='center', fontweight='bold')

        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vague Words', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Most Common Vague Words', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved vague words chart: {output_path}")
        plt.close()

    def generate_all_visualizations(self, output_dir: str = "visualizations"):
        """Generate all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        self.create_similarity_distribution(str(output_path / "similarity_distribution.png"))
        self.create_impact_severity_chart(str(output_path / "impact_severity.png"))
        self.create_overlap_type_chart(str(output_path / "overlap_types.png"))
        self.create_quality_metrics_dashboard(str(output_path / "quality_dashboard.png"))
        self.create_confidence_analysis(str(output_path / "confidence_analysis.png"))
        self.create_vague_words_wordcloud(str(output_path / "vague_words_top20.png"))

        print("\n" + "=" * 60)
        print(f"✅ All visualizations saved to: {output_path.resolve()}")
        print("=" * 60)


def main():
    """Main execution."""
    script_dir = Path(__file__).parent.resolve()
    report_path = script_dir / "enterprise_redundancy_report.json"

    if not report_path.exists():
        print(f"\n❌ Error: Report not found at {report_path}")
        print("Please run enterprise_redundancy_analyzer.py first.")
        sys.exit(1)

    try:
        visualizer = ReportVisualizer(str(report_path))
        visualizer.generate_all_visualizations("visualizations")

        print("\n📊 Visualizations ready for:")
        print("  • Executive presentations")
        print("  • Quality reports")
        print("  • Trend analysis")
        print("  • Stakeholder communication")

    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
