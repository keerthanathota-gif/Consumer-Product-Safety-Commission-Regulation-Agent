#!/usr/bin/env python3
"""
Data loader for CPSC Regulation Knowledge Graph
Loads data from SQLite database and JSON files
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from entities import (
    RegulationSection, RegulationPart, RegulationChapter,
    create_regulation_section_from_db, create_regulation_part_from_db, 
    create_regulation_chapter_from_db
)
from config import DATABASE_CONFIG, DATA_ROOT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadedData:
    """Container for loaded data"""
    sections: List[RegulationSection]
    parts: List[RegulationPart]
    chapters: List[RegulationChapter]
    metadata: Dict[str, Any]

class DataLoader:
    """Loads regulation data from various sources"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_CONFIG["sqlite_path"]
        self.data_root = DATA_ROOT
        
    def load_from_sqlite(self) -> LoadedData:
        """Load data from SQLite database"""
        logger.info(f"Loading data from SQLite database: {self.db_path}")
        
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        
        try:
            # Load chapters
            chapters = self._load_chapters(conn)
            logger.info(f"Loaded {len(chapters)} chapters")
            
            # Load parts
            parts = self._load_parts(conn)
            logger.info(f"Loaded {len(parts)} parts")
            
            # Load sections
            sections = self._load_sections(conn)
            logger.info(f"Loaded {len(sections)} sections")
            
            # Load metadata
            metadata = self._load_metadata(conn)
            
            return LoadedData(
                sections=sections,
                parts=parts,
                chapters=chapters,
                metadata=metadata
            )
            
        finally:
            conn.close()
    
    def _load_chapters(self, conn: sqlite3.Connection) -> List[RegulationChapter]:
        """Load chapters from database"""
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.*, 
                   COUNT(DISTINCT sc.subchapter_id) as subchapter_count,
                   COUNT(DISTINCT p.part_id) as part_count,
                   COUNT(DISTINCT s.section_id) as section_count
            FROM chapters c
            LEFT JOIN subchapters sc ON c.chapter_id = sc.chapter_id
            LEFT JOIN parts p ON sc.subchapter_id = p.subchapter_id
            LEFT JOIN sections s ON p.part_id = s.part_id
            GROUP BY c.chapter_id
        """)
        
        chapters = []
        for row in cursor.fetchall():
            chapter_data = dict(row)
            chapter = create_regulation_chapter_from_db(chapter_data)
            chapter.subchapter_count = chapter_data.get('subchapter_count', 0)
            chapter.part_count = chapter_data.get('part_count', 0)
            chapter.section_count = chapter_data.get('section_count', 0)
            chapters.append(chapter)
        
        return chapters
    
    def _load_parts(self, conn: sqlite3.Connection) -> List[RegulationPart]:
        """Load parts from database"""
        cursor = conn.cursor()
        cursor.execute("""
            SELECT p.*, 
                   sc.chapter_id,
                   COUNT(s.section_id) as section_count
            FROM parts p
            JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
            LEFT JOIN sections s ON p.part_id = s.part_id
            GROUP BY p.part_id
        """)
        
        parts = []
        for row in cursor.fetchall():
            part_data = dict(row)
            part = create_regulation_part_from_db(part_data)
            part.section_count = part_data.get('section_count', 0)
            parts.append(part)
        
        return parts
    
    def _load_sections(self, conn: sqlite3.Connection) -> List[RegulationSection]:
        """Load sections from database"""
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.*, 
                   p.subchapter_id,
                   sc.chapter_id
            FROM sections s
            JOIN parts p ON s.part_id = p.part_id
            JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
            ORDER BY s.section_id
        """)
        
        sections = []
        for row in cursor.fetchall():
            section_data = dict(row)
            section = create_regulation_section_from_db(section_data)
            sections.append(section)
        
        return sections
    
    def _load_metadata(self, conn: sqlite3.Connection) -> Dict[str, Any]:
        """Load database metadata"""
        cursor = conn.cursor()
        
        # Get table counts
        tables = ['chapters', 'subchapters', 'parts', 'sections']
        counts = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        
        # Get database info
        cursor.execute("PRAGMA table_info(sections)")
        section_columns = [col[1] for col in cursor.fetchall()]
        
        return {
            "table_counts": counts,
            "section_columns": section_columns,
            "database_path": self.db_path,
            "load_timestamp": pd.Timestamp.now().isoformat()
        }
    
    def load_from_json(self, json_path: str = None) -> LoadedData:
        """Load data from JSON file (alternative to SQLite)"""
        json_path = json_path or str(self.data_root / "chapter_subchapter_part_sections_no_metadata.json")
        logger.info(f"Loading data from JSON file: {json_path}")
        
        if not Path(json_path).exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chapters = []
        parts = []
        sections = []
        
        # Process hierarchical data
        for chapter_data in data.get('chapters', []):
            chapter_id = len(chapters) + 1
            chapter = RegulationChapter(
                id=f"chapter_{chapter_id}",
                name=chapter_data['chapter_name'],
                description=chapter_data['chapter_name'],
                chapter_number=str(chapter_id),
                chapter_name=chapter_data['chapter_name']
            )
            chapters.append(chapter)
            
            for subchapter_data in chapter_data.get('subchapters', []):
                subchapter_id = len(parts) + 1
                
                for part_data in subchapter_data.get('parts', []):
                    part_id = len(parts) + 1
                    part = RegulationPart(
                        id=f"part_{part_id}",
                        name=part_data['heading'],
                        description=part_data['heading'],
                        part_number=str(part_id),
                        heading=part_data['heading'],
                        subchapter_id=subchapter_id,
                        chapter_id=chapter_id
                    )
                    parts.append(part)
                    
                    for section_data in part_data.get('sections', []):
                        section_id = len(sections) + 1
                        section = RegulationSection(
                            id=f"section_{section_id}",
                            name=section_data['section_number'],
                            description=section_data['subject'],
                            section_number=section_data['section_number'],
                            subject=section_data['subject'],
                            text=section_data['text'],
                            citation=section_data.get('citation', ''),
                            part_id=part_id,
                            chapter_id=chapter_id,
                            subchapter_id=subchapter_id,
                            word_count=len(section_data['text'].split()) if section_data['text'] else 0,
                            sentence_count=len([s for s in section_data['text'].split('.') if s.strip()]) if section_data['text'] else 0
                        )
                        sections.append(section)
        
        # Update counts
        for chapter in chapters:
            chapter.subchapter_count = len([p for p in parts if p.chapter_id == int(chapter.chapter_number)])
            chapter.part_count = len([p for p in parts if p.chapter_id == int(chapter.chapter_number)])
            chapter.section_count = len([s for s in sections if s.chapter_id == int(chapter.chapter_number)])
        
        for part in parts:
            part.section_count = len([s for s in sections if s.part_id == int(part.part_number)])
        
        metadata = {
            "source": "json",
            "file_path": json_path,
            "load_timestamp": pd.Timestamp.now().isoformat(),
            "table_counts": {
                "chapters": len(chapters),
                "parts": len(parts),
                "sections": len(sections)
            }
        }
        
        logger.info(f"Loaded {len(chapters)} chapters, {len(parts)} parts, {len(sections)} sections from JSON")
        
        return LoadedData(
            sections=sections,
            parts=parts,
            chapters=chapters,
            metadata=metadata
        )
    
    def get_section_by_number(self, section_number: str, conn: sqlite3.Connection = None) -> Optional[RegulationSection]:
        """Get a specific section by section number"""
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            should_close = True
        else:
            should_close = False
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, p.subchapter_id, sc.chapter_id
                FROM sections s
                JOIN parts p ON s.part_id = p.part_id
                JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
                WHERE s.section_number = ?
            """, (section_number,))
            
            row = cursor.fetchone()
            if row:
                section_data = dict(row)
                return create_regulation_section_from_db(section_data)
            return None
            
        finally:
            if should_close:
                conn.close()
    
    def search_sections(self, query: str, limit: int = 20) -> List[RegulationSection]:
        """Search sections by text content"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, p.subchapter_id, sc.chapter_id
                FROM sections s
                JOIN parts p ON s.part_id = p.part_id
                JOIN subchapters sc ON p.subchapter_id = sc.subchapter_id
                WHERE s.text LIKE ? OR s.subject LIKE ?
                ORDER BY s.section_id
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))
            
            sections = []
            for row in cursor.fetchall():
                section_data = dict(row)
                section = create_regulation_section_from_db(section_data)
                sections.append(section)
            
            return sections
            
        finally:
            conn.close()
    
    def get_hierarchical_structure(self) -> Dict[str, Any]:
        """Get the complete hierarchical structure"""
        data = self.load_from_sqlite()
        
        structure = {
            "chapters": [],
            "total_sections": len(data.sections),
            "total_parts": len(data.parts),
            "total_chapters": len(data.chapters)
        }
        
        for chapter in data.chapters:
            chapter_data = {
                "id": chapter.id,
                "name": chapter.chapter_name,
                "subchapters": []
            }
            
            # Get parts for this chapter
            chapter_parts = [p for p in data.parts if p.chapter_id == int(chapter.chapter_number)]
            
            # Group parts by subchapter
            subchapters = {}
            for part in chapter_parts:
                if part.subchapter_id not in subchapters:
                    subchapters[part.subchapter_id] = {
                        "id": f"subchapter_{part.subchapter_id}",
                        "parts": []
                    }
                
                part_data = {
                    "id": part.id,
                    "name": part.heading,
                    "sections": []
                }
                
                # Get sections for this part
                part_sections = [s for s in data.sections if s.part_id == int(part.part_number)]
                for section in part_sections:
                    section_data = {
                        "id": section.id,
                        "number": section.section_number,
                        "subject": section.subject,
                        "text_preview": section.text[:200] + "..." if len(section.text) > 200 else section.text
                    }
                    part_data["sections"].append(section_data)
                
                subchapters[part.subchapter_id]["parts"].append(part_data)
            
            chapter_data["subchapters"] = list(subchapters.values())
            structure["chapters"].append(chapter_data)
        
        return structure

def main():
    """Test the data loader"""
    loader = DataLoader()
    
    try:
        # Test SQLite loading
        data = loader.load_from_sqlite()
        print(f"✅ Loaded {len(data.sections)} sections, {len(data.parts)} parts, {len(data.chapters)} chapters")
        
        # Test search
        search_results = loader.search_sections("safety", limit=5)
        print(f"✅ Found {len(search_results)} sections containing 'safety'")
        
        # Test hierarchical structure
        structure = loader.get_hierarchical_structure()
        print(f"✅ Hierarchical structure: {structure['total_chapters']} chapters, {structure['total_parts']} parts, {structure['total_sections']} sections")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Data loading failed: {e}")

if __name__ == "__main__":
    main()