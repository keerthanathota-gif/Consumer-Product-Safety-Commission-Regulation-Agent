from lxml import etree
import json
import csv
import os

def parse_chapter_subchapter_part_sections(xml_file):
    """Extract Chapter → Subchapter → Part → Sections hierarchy, excluding subchapters with no parts and metadata."""
    # Parse XML with lxml
    tree = etree.parse(xml_file)
    
    # Initialize results dictionary for JSON
    results = {
        "chapters": []
    }
    
    # Find all chapters
    chapters = tree.xpath("//CHAPTER")
    for chapter in chapters:
        # Get chapter name from TOCHD/HD or CHAPTER/HD
        chapter_name = chapter.xpath("TOC/TOCHD/HD[@SOURCE='HED']/text() | HD[@SOURCE='HED']/text()")
        chapter_name = chapter_name[0].strip() if chapter_name else "Unknown Chapter"
        chapter_data = {
            "chapter_name": chapter_name,
            "subchapters": []
        }
        
        # Find subchapters with at least one part
        subchapters = chapter.xpath(".//SUBCHAP[.//PART]")
        for subchapter in subchapters:
            subchapter_name = subchapter.xpath("HD[@SOURCE='HED']/text()")[0].strip() if subchapter.xpath("HD[@SOURCE='HED']/text()") else "Unknown Subchapter"
            subchapter_data = {
                "subchapter_name": subchapter_name,
                "parts": []
            }
            
            # Find all parts within the subchapter
            parts = subchapter.xpath(".//PART")
            for part in parts:
                part_heading = part.xpath("HD[@SOURCE='HED']/text()")[0].strip() if part.xpath("HD[@SOURCE='HED']/text()") else "Unknown Part"
                part_data = {
                    "heading": part_heading,
                    "sections": []
                }
                
                # Find all sections within the part
                sections = part.xpath("SECTION")
                for section in sections:
                    sectno = section.xpath("SECTNO/text()")
                    subject = section.xpath("SUBJECT/text()")
                    citation=section.xpath("CITA/text()")
                    text_elements = section.xpath(".//P/text()|.//NOTE/P/text()|.//LIST/LI/text()")
                    text = " ".join([t.strip() for t in text_elements if t.strip()])
                    part_data["sections"].append({
                        "section_number": sectno[0] if sectno else "",
                        "subject": subject[0] if subject else "",
                        "text": text,
                        "citation": citation[0] if citation else ""
                    })
                
                subchapter_data["parts"].append(part_data)
            
            chapter_data["subchapters"].append(subchapter_data)
        
        # Only add chapter if it has subchapters with parts
        if chapter_data["subchapters"]:
            results["chapters"].append(chapter_data)
    
    return results

def save_json(data, output_file):
    """Save data to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_csv(data, output_file):
    """Save data to CSV file."""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Chapter Name", "Subchapter Name", "Part Heading", "Section Number", "Section Subject", "Section Text"])
        for chapter in data["chapters"]:
            chapter_name = chapter["chapter_name"]
            for subchapter in chapter["subchapters"]:
                subchapter_name = subchapter["subchapter_name"]
                for part in subchapter["parts"]:
                    part_heading = part["heading"]
                    for section in part["sections"]:
                        writer.writerow([
                            chapter_name,
                            subchapter_name,
                            part_heading,
                            section["section_number"],
                            section["subject"],
                            section["text"],
                            section["citation"]
                        ])

# Example usage
if __name__ == "__main__":
    xml_file = "CFR-2025-title16-vol2.xml"  # Replace with actual file path
    if os.path.exists(xml_file):
        parsed_data = parse_chapter_subchapter_part_sections(xml_file)
        save_json(parsed_data, "chapter_subchapter_part_sections_no_metadata.json")
        save_csv(parsed_data, "chapter_subchapter_part_sections_no_metadata.csv")
        print("XPath queries executed. Results saved to chapter_subchapter_part_sections_no_metadata.json and chapter_subchapter_part_sections_no_metadata.csv")
    else:
        print(f"XML file '{xml_file}' not found.")