#!/usr/bin/env python3
"""
Example usage of the Embeddings and Retrieval System
Run this after creating embeddings with embeddings_retrieval.py
"""

from embeddings_retrieval import RegulationEmbeddings
from pathlib import Path


def example_1_basic_search():
    """Example 1: Basic semantic search"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Semantic Search")
    print("="*80)

    system = RegulationEmbeddings()

    # Try to load existing embeddings, create if they don't exist
    if not system.load_embeddings():
        print("Embeddings not found. Creating new embeddings...")
        system.create_embeddings()

    # Search for toy safety
    print("\nSearching for: 'toy safety requirements for children'")
    results = system.retrieve(
        query="toy safety requirements for children",
        top_k=3,
        verbose=True
    )


def example_2_specific_topics():
    """Example 2: Search for specific regulatory topics"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Searching Specific Topics")
    print("="*80)

    system = RegulationEmbeddings()
    system.load_embeddings()

    topics = [
        "flammable fabrics and clothing standards",
        "lead content in paint",
        "choking hazards for small children",
        "product recall procedures"
    ]

    for topic in topics:
        print(f"\n--- Searching: '{topic}' ---")
        results = system.search(topic, top_k=2, min_similarity=0.1)

        for i, result in enumerate(results, 1):
            print(f"{i}. {result['section_number']} - {result['subject']}")
            print(f"   Score: {result['similarity_score']:.4f}")


def example_3_find_similar():
    """Example 3: Find sections similar to a specific section"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Find Similar Sections")
    print("="*80)

    system = RegulationEmbeddings()
    system.load_embeddings()

    # Pick a section (let's say section_id = 1)
    section_id = 1
    reference_section = system.get_section_by_id(section_id)

    if reference_section:
        print(f"\nReference Section:")
        print(f"  {reference_section['section_number']} - {reference_section['subject']}")
        print(f"  Text: {reference_section['text'][:150]}...")

        print(f"\nFinding similar sections...")
        similar = system.find_similar_sections(section_id, top_k=3)

        for i, sec in enumerate(similar, 1):
            print(f"\n{i}. {sec['section_number']} - {sec['subject']}")
            print(f"   Similarity: {sec['similarity_score']:.4f}")
            print(f"   Text: {sec['text'][:150]}...")


def example_4_statistics():
    """Example 4: Display system statistics"""
    print("\n" + "="*80)
    print("EXAMPLE 4: System Statistics")
    print("="*80)

    system = RegulationEmbeddings()
    system.load_embeddings()

    stats = system.get_statistics()

    print("\nDatabase Coverage:")
    print(f"  Total Sections: {stats['total_sections']}")
    print(f"  Total Parts: {stats['total_parts']}")
    print(f"  Total Subchapters: {stats['total_subchapters']}")
    print(f"  Total Chapters: {stats['total_chapters']}")

    print("\nEmbeddings Information:")
    print(f"  Vocabulary Size: {stats['vocabulary_size']:,} terms")
    print(f"  Embedding Dimensions: {stats['embedding_dimensions']:,}")

    print("\nText Statistics:")
    print(f"  Average Text Length: {stats['avg_text_length']:.0f} characters")
    print(f"  Shortest Section: {stats['min_text_length']} characters")
    print(f"  Longest Section: {stats['max_text_length']:,} characters")


def example_5_custom_search():
    """Example 5: Interactive search"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Interactive Search")
    print("="*80)

    system = RegulationEmbeddings()

    if not system.load_embeddings():
        print("Creating embeddings first...")
        system.create_embeddings()

    print("\nInteractive Search Mode")
    print("Enter your search query (or 'quit' to exit)")

    while True:
        query = input("\nSearch query: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        results = system.search(query, top_k=5, min_similarity=0.1)

        if not results:
            print("No relevant results found.")
            continue

        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['similarity_score']:.3f}] {result['section_number']}")
            print(f"   {result['subject']}")
            print(f"   {result['text'][:200]}...")


def main():
    """Run all examples"""
    print("="*80)
    print("CPSC Regulation Embeddings - Usage Examples")
    print("="*80)

    try:
        # Run examples
        example_1_basic_search()
        example_2_specific_topics()
        example_3_find_similar()
        example_4_statistics()

        # Uncomment for interactive mode
        # example_5_custom_search()

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
