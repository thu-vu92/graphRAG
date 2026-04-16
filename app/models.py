from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Ontology ───────────────────────────────────────────────────────────────────

class OntologyConfig(BaseModel):
    """Per-topic ontology. Defaults to a generic schema suitable for any domain."""

    entity_types: list[str] = Field(
        default=[
            "PERSON",
            "ORGANIZATION",
            "CONCEPT",
            "EVENT",
            "LOCATION",
            "DOCUMENT",
            "TECHNOLOGY",
            "PRODUCT",
        ]
    )
    relation_types: list[str] = Field(
        default=[
            "RELATED_TO",
            "PART_OF",
            "CREATED_BY",
            "LOCATED_IN",
            "REFERENCES",
            "OPPOSES",
            "SUPPORTS",
            "REGULATES",
        ]
    )


# ── Extraction models (used by pipeline + LLM structured predict) ──────────────

class ExtractedEntity(BaseModel):
    name: str = Field(description="Name of the entity, capitalized")
    type: str = Field(description="One of the allowed entity types")
    description: str = Field(description="Brief description of the entity and its role")


class ExtractedRelationship(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relation: str = Field(description="One of the allowed relationship types")
    description: str = Field(description="Sentence explaining the relationship")


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


# ── API request / response models ─────────────────────────────────────────────

class BuildRequest(BaseModel):
    ontology: Optional[OntologyConfig] = None


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    communities_checked: int
    relevant_communities: int


class TopicStatus(BaseModel):
    topic: str
    has_raw_files: bool
    has_graph: bool
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    community_count: Optional[int] = None
    build_status: str = "idle"   # idle | building | complete | error
    build_progress: Optional[str] = None
    build_error: Optional[str] = None


# ── Internal task state ────────────────────────────────────────────────────────

class TaskState(BaseModel):
    topic: str
    status: str = "building"   # building | complete | error
    progress: str = "Starting..."
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
