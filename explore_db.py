#!/usr/bin/env python3
import sqlite3

def explore_database():
    conn = sqlite3.connect('regulations.db')
    cursor = conn.cursor()
    
    # Get total sections
    cursor.execute('SELECT COUNT(*) FROM sections')
    total_sections = cursor.fetchone()[0]
    print(f'Total sections: {total_sections}')
    
    # Get sample sections
    cursor.execute('SELECT section_number, subject FROM sections LIMIT 5')
    print('\nSample sections:')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1][:50]}...')
    
    # Get database schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f'\nDatabase tables: {[t[0] for t in tables]}')
    
    # Get hierarchy counts
    cursor.execute('SELECT COUNT(*) FROM chapters')
    chapters = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM subchapters')
    subchapters = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM parts')
    parts = cursor.fetchone()[0]
    
    print(f'\nHierarchy counts:')
    print(f'  Chapters: {chapters}')
    print(f'  Subchapters: {subchapters}')
    print(f'  Parts: {parts}')
    print(f'  Sections: {total_sections}')
    
    conn.close()

if __name__ == "__main__":
    explore_database()