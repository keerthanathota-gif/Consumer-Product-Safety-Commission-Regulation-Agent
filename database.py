#!/usr/bin/env python3
"""
Script to create SQLite database from chapter_subchapter_part_sections JSON file.
Creates a normalized database with proper foreign key relationships.
Windows compatible version.
"""

import sqlite3
import json
import sys
import os
from pathlib import Path


def create_database_schema(conn):
    """Create the database schema with proper relationships."""
    cursor = conn.cursor()
    
    # Enable foreign key support
    cursor.execute("PRAGMA foreign_keys = ON;")
    
    # Create chapters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            chapter_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_name TEXT NOT NULL
        );
    """)
    
    # Create subchapters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subchapters (
            subchapter_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id INTEGER NOT NULL,
            subchapter_name TEXT NOT NULL,
            FOREIGN KEY (chapter_id) REFERENCES chapters(chapter_id)
        );
    """)
    
    # Create parts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parts (
            part_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subchapter_id INTEGER NOT NULL,
            heading TEXT NOT NULL,
            FOREIGN KEY (subchapter_id) REFERENCES subchapters(subchapter_id)
        );
    """)
    
    # Create sections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            section_id INTEGER PRIMARY KEY AUTOINCREMENT,
            part_id INTEGER NOT NULL,
            section_number TEXT,
            subject TEXT,
            text TEXT,
            citation TEXT,
            FOREIGN KEY (part_id) REFERENCES parts(part_id)
        );
    """)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_subchapters_chapter ON subchapters(chapter_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_subchapter ON parts(subchapter_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sections_part ON sections(part_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sections_number ON sections(section_number);")
    
    conn.commit()
    print("✓ Database schema created successfully")


def import_data(conn, json_file_path):
    """Import data from JSON file into the database."""
    cursor = conn.cursor()
    
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chapters = data.get('chapters', [])
    print(f"Found {len(chapters)} chapters to import")
    
    chapter_count = 0
    subchapter_count = 0
    part_count = 0
    section_count = 0
    
    # Import data with proper relationships
    for chapter in chapters:
        chapter_name = chapter.get('chapter_name', '')
        
        # Insert chapter
        cursor.execute(
            "INSERT INTO chapters (chapter_name) VALUES (?)",
            (chapter_name,)
        )
        chapter_id = cursor.lastrowid
        chapter_count += 1
        
        # Process subchapters
        for subchapter in chapter.get('subchapters', []):
            subchapter_name = subchapter.get('subchapter_name', '')
            
            # Insert subchapter
            cursor.execute(
                "INSERT INTO subchapters (chapter_id, subchapter_name) VALUES (?, ?)",
                (chapter_id, subchapter_name)
            )
            subchapter_id = cursor.lastrowid
            subchapter_count += 1
            
            # Process parts
            for part in subchapter.get('parts', []):
                heading = part.get('heading', '')
                
                # Insert part
                cursor.execute(
                    "INSERT INTO parts (subchapter_id, heading) VALUES (?, ?)",
                    (subchapter_id, heading)
                )
                part_id = cursor.lastrowid
                part_count += 1
                
                # Process sections
                for section in part.get('sections', []):
                    section_number = section.get('section_number', '')
                    subject = section.get('subject', '')
                    text = section.get('text', '')
                    citation = section.get('citation', '')
                    
                    # Insert section
                    cursor.execute(
                        """INSERT INTO sections 
                           (part_id, section_number, subject, text, citation) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (part_id, section_number, subject, text, citation)
                    )
                    section_count += 1
    
    conn.commit()
    
    print(f"\n✓ Data import completed successfully:")
    print(f"  - Chapters: {chapter_count}")
    print(f"  - Subchapters: {subchapter_count}")
    print(f"  - Parts: {part_count}")
    print(f"  - Sections: {section_count}")


def create_views(conn):
    """Create helpful views for querying the database."""
    cursor = conn.cursor()
    
    # Create a comprehensive view joining all tables
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS full_hierarchy AS
        SELECT 
            c.chapter_id,
            c.chapter_name,
            sc.subchapter_id,
            sc.subchapter_name,
            p.part_id,
            p.heading AS part_heading,
            s.section_id,
            s.section_number,
            s.subject AS section_subject,
            s.text AS section_text,
            s.citation
        FROM chapters c
        LEFT JOIN subchapters sc ON c.chapter_id = sc.chapter_id
        LEFT JOIN parts p ON sc.subchapter_id = p.subchapter_id
        LEFT JOIN sections s ON p.part_id = s.part_id;
    """)
    
    conn.commit()
    print("✓ Database views created successfully")


def verify_database(conn):
    """Verify the database was created correctly."""
    cursor = conn.cursor()
    
    print("\n--- Database Statistics ---")
    
    # Count records in each table
    tables = ['chapters', 'subchapters', 'parts', 'sections']
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table.capitalize()}: {count}")
    
    # Show sample data
    print("\n--- Sample Data ---")
    cursor.execute("""
        SELECT chapter_name, subchapter_name, part_heading, section_number, section_subject
        FROM full_hierarchy
        WHERE section_number IS NOT NULL AND section_number != ''
        LIMIT 3
    """)
    
    rows = cursor.fetchall()
    for row in rows:
        print(f"\nChapter: {row[0][:60]}...")
        print(f"Subchapter: {row[1][:60]}...")
        print(f"Part: {row[2][:60]}...")
        print(f"Section: {row[3]} - {row[4][:60]}...")


def main():
    """Main function to create and populate the database."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # File paths - look in the same directory as the script
    json_file = script_dir / "chapter_subchapter_part_sections_no_metadata.json"
    db_file = script_dir / "regulations.db"
    
    # Check if JSON file exists
    if not json_file.exists():
        print(f"Error: JSON file not found at {json_file}")
        print(f"\nPlease place 'chapter_subchapter_part_sections_no_metadata.json' in the same directory as this script.")
        print(f"Script directory: {script_dir}")
        sys.exit(1)
    
    # Remove existing database if it exists
    if db_file.exists():
        db_file.unlink()
        print(f"Removed existing database: {db_file}")
    
    print(f"Creating new database: {db_file}\n")
    
    # Create database connection
    conn = sqlite3.connect(str(db_file))
    
    try:
        # Create schema
        create_database_schema(conn)
        
        # Import data
        import_data(conn, json_file)
        
        # Create views
        create_views(conn)
        
        # Verify database
        verify_database(conn)
        
        print(f"\n✓ Database created successfully at: {db_file}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        conn.close()


if __name__ == "__main__":
    main()