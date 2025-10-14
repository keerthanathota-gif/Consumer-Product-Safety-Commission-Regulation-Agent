#!/usr/bin/env python3
"""
Pydantic entity models for CPSC Regulation Knowledge Graph
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class EntityType(str, Enum):
    """Entity types in the regulation knowledge graph"""
    REGULATION_SECTION = "regulation_section"
    REGULATION_PART = "regulation_part"
    REGULATION_CHAPTER = "regulation_chapter"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"
    SAFETY_STANDARD = "safety_standard"
    PRODUCT_CATEGORY = "product_category"
    HAZARD_TYPE = "hazard_type"

class RelationshipType(str, Enum):
    """Relationship types in the regulation knowledge graph"""
    # Hierarchical relationships
    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"
    PART_OF = "part_of"
    
    # Semantic relationships
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    REFERENCES = "references"
    
    # Compliance relationships
    REQUIRES = "requires"
    PROHIBITS = "prohibits"
    MANDATES = "mandates"
    
    # Temporal relationships
    SUPERSEDES = "supersedes"
    AMENDS = "amends"
    REPLACES = "replaces"
    
    # Cross-reference relationships
    CITES = "cites"
    DEFINES = "defines"

class BaseEntity(BaseModel):
    """Base entity class with common fields"""
    id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Entity name")
    description: Optional[str] = Field(None, description="Entity description")
    entity_type: EntityType = Field(..., description="Type of entity")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RegulationSection(BaseEntity):
    """Represents a single regulation section"""
    section_number: str = Field(..., description="Section number (e.g., '§ 1000.1')")
    subject: str = Field(..., description="Section subject/title")
    text: str = Field(..., description="Full regulation text")
    citation: Optional[str] = Field(None, description="Legal citation")
    part_id: int = Field(..., description="ID of the parent part")
    chapter_id: int = Field(..., description="ID of the parent chapter")
    subchapter_id: int = Field(..., description="ID of the parent subchapter")
    
    # Analysis fields
    word_count: int = Field(default=0, description="Number of words in text")
    sentence_count: int = Field(default=0, description="Number of sentences")
    readability_score: Optional[float] = Field(None, description="Flesch readability score")
    vague_words: List[str] = Field(default_factory=list, description="Detected vague words")
    
    # Compliance fields
    compliance_requirements: List[str] = Field(default_factory=list, description="Extracted compliance requirements")
    safety_standards: List[str] = Field(default_factory=list, description="Referenced safety standards")
    product_categories: List[str] = Field(default_factory=list, description="Applicable product categories")
    hazard_types: List[str] = Field(default_factory=list, description="Addressed hazard types")
    
    entity_type: EntityType = Field(default=EntityType.REGULATION_SECTION)

class RegulationPart(BaseEntity):
    """Represents a regulation part"""
    part_number: str = Field(..., description="Part number (e.g., '1000')")
    heading: str = Field(..., description="Part heading")
    subchapter_id: int = Field(..., description="ID of the parent subchapter")
    chapter_id: int = Field(..., description="ID of the parent chapter")
    section_count: int = Field(default=0, description="Number of sections in this part")
    
    # Summary fields
    summary: Optional[str] = Field(None, description="Part summary")
    key_topics: List[str] = Field(default_factory=list, description="Key topics covered")
    
    entity_type: EntityType = Field(default=EntityType.REGULATION_PART)

class RegulationChapter(BaseEntity):
    """Represents a regulation chapter"""
    chapter_number: str = Field(..., description="Chapter number")
    chapter_name: str = Field(..., description="Chapter name")
    subchapter_count: int = Field(default=0, description="Number of subchapters")
    part_count: int = Field(default=0, description="Number of parts")
    section_count: int = Field(default=0, description="Number of sections")
    
    # Summary fields
    summary: Optional[str] = Field(None, description="Chapter summary")
    scope: Optional[str] = Field(None, description="Chapter scope")
    
    entity_type: EntityType = Field(default=EntityType.REGULATION_CHAPTER)

class ComplianceRequirement(BaseEntity):
    """Represents a compliance requirement"""
    requirement_text: str = Field(..., description="Full requirement text")
    requirement_type: str = Field(..., description="Type of requirement (mandatory, prohibited, etc.)")
    applicable_products: List[str] = Field(default_factory=list, description="Applicable product categories")
    effective_date: Optional[datetime] = Field(None, description="When requirement becomes effective")
    expiration_date: Optional[datetime] = Field(None, description="When requirement expires")
    source_sections: List[str] = Field(default_factory=list, description="Source regulation sections")
    
    # Compliance fields
    enforcement_level: str = Field(default="standard", description="Enforcement level")
    penalty_amount: Optional[float] = Field(None, description="Penalty amount for non-compliance")
    
    entity_type: EntityType = Field(default=EntityType.COMPLIANCE_REQUIREMENT)

class SafetyStandard(BaseEntity):
    """Represents a safety standard"""
    standard_number: str = Field(..., description="Standard number")
    standard_name: str = Field(..., description="Standard name")
    standard_body: str = Field(..., description="Standards body (e.g., ASTM, ISO)")
    version: Optional[str] = Field(None, description="Standard version")
    effective_date: Optional[datetime] = Field(None, description="Effective date")
    applicable_products: List[str] = Field(default_factory=list, description="Applicable products")
    referenced_by: List[str] = Field(default_factory=list, description="Regulations that reference this standard")
    
    entity_type: EntityType = Field(default=EntityType.SAFETY_STANDARD)

class ProductCategory(BaseEntity):
    """Represents a product category"""
    category_name: str = Field(..., description="Product category name")
    description: str = Field(..., description="Category description")
    parent_category: Optional[str] = Field(None, description="Parent category")
    subcategories: List[str] = Field(default_factory=list, description="Subcategories")
    applicable_regulations: List[str] = Field(default_factory=list, description="Applicable regulations")
    hazard_risks: List[str] = Field(default_factory=list, description="Associated hazard risks")
    
    entity_type: EntityType = Field(default=EntityType.PRODUCT_CATEGORY)

class HazardType(BaseEntity):
    """Represents a hazard type"""
    hazard_name: str = Field(..., description="Hazard name")
    description: str = Field(..., description="Hazard description")
    severity_level: str = Field(..., description="Severity level (low, medium, high, critical)")
    affected_products: List[str] = Field(default_factory=list, description="Affected product categories")
    prevention_measures: List[str] = Field(default_factory=list, description="Prevention measures")
    applicable_regulations: List[str] = Field(default_factory=list, description="Applicable regulations")
    
    entity_type: EntityType = Field(default=EntityType.HAZARD_TYPE)

class Relationship(BaseModel):
    """Represents a relationship between entities"""
    id: str = Field(..., description="Unique relationship ID")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    weight: float = Field(default=1.0, description="Relationship weight/strength")
    confidence: float = Field(default=1.0, description="Confidence in relationship")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Context fields
    context: Optional[str] = Field(None, description="Relationship context")
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting relationship")

class GraphNode(BaseModel):
    """Represents a node in the knowledge graph"""
    entity: BaseEntity = Field(..., description="Entity data")
    neighbors: List[str] = Field(default_factory=list, description="Connected node IDs")
    degree: int = Field(default=0, description="Node degree (number of connections)")
    centrality_score: Optional[float] = Field(None, description="Centrality score")
    cluster_id: Optional[int] = Field(None, description="Cluster assignment")

class GraphEdge(BaseModel):
    """Represents an edge in the knowledge graph"""
    relationship: Relationship = Field(..., description="Relationship data")
    source_node: str = Field(..., description="Source node ID")
    target_node: str = Field(..., description="Target node ID")
    weight: float = Field(default=1.0, description="Edge weight")
    directed: bool = Field(default=True, description="Whether edge is directed")

class KnowledgeGraph(BaseModel):
    """Represents the complete knowledge graph"""
    nodes: Dict[str, GraphNode] = Field(default_factory=dict, description="Graph nodes")
    edges: Dict[str, GraphEdge] = Field(default_factory=dict, description="Graph edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Graph metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph"""
        self.nodes[node.entity.id] = node
        self.updated_at = datetime.utcnow()
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph"""
        self.edges[edge.relationship.id] = edge
        # Update node degrees
        if edge.source_node in self.nodes:
            self.nodes[edge.source_node].neighbors.append(edge.target_node)
            self.nodes[edge.source_node].degree += 1
        if edge.target_node in self.nodes:
            self.nodes[edge.target_node].neighbors.append(edge.source_node)
            self.nodes[edge.target_node].degree += 1
        self.updated_at = datetime.utcnow()
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str) -> List[GraphNode]:
        """Get all neighbors of a node"""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[neighbor_id] for neighbor_id in node.neighbors if neighbor_id in self.nodes]
    
    def get_edges(self, node_id: str) -> List[GraphEdge]:
        """Get all edges connected to a node"""
        return [edge for edge in self.edges.values() 
                if edge.source_node == node_id or edge.target_node == node_id]

# Utility functions for entity creation
def create_regulation_section_from_db(row: Dict[str, Any]) -> RegulationSection:
    """Create a RegulationSection from database row"""
    return RegulationSection(
        id=f"section_{row['section_id']}",
        name=row['section_number'],
        description=row['subject'],
        section_number=row['section_number'],
        subject=row['subject'],
        text=row['text'],
        citation=row.get('citation', ''),
        part_id=row['part_id'],
        chapter_id=row.get('chapter_id', 0),
        subchapter_id=row.get('subchapter_id', 0),
        word_count=len(row['text'].split()) if row['text'] else 0,
        sentence_count=len([s for s in row['text'].split('.') if s.strip()]) if row['text'] else 0
    )

def create_regulation_part_from_db(row: Dict[str, Any]) -> RegulationPart:
    """Create a RegulationPart from database row"""
    return RegulationPart(
        id=f"part_{row['part_id']}",
        name=row['heading'],
        description=row['heading'],
        part_number=str(row['part_id']),
        heading=row['heading'],
        subchapter_id=row['subchapter_id'],
        chapter_id=row.get('chapter_id', 0)
    )

def create_regulation_chapter_from_db(row: Dict[str, Any]) -> RegulationChapter:
    """Create a RegulationChapter from database row"""
    return RegulationChapter(
        id=f"chapter_{row['chapter_id']}",
        name=row['chapter_name'],
        description=row['chapter_name'],
        chapter_number=str(row['chapter_id']),
        chapter_name=row['chapter_name']
    )