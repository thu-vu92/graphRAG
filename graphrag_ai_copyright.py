# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "graspologic",
#     "llama-index",
#     "llama-index-llms-openai",
#     "marimo>=0.25.2",
#     "nest-asyncio",
#     "networkx==3.7",
#     "pandas",
#     "pydantic==2.14.2",
#     "python-dotenv",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GraphRAG: AI Copyright & Governance Knowledge Graph

    This notebook builds a full **GraphRAG pipeline** on a pre-scraped dataset of news articles
    about AI copyright, governance, and intellectual property.

    ### Pipeline overview

    | Step | What happens |
    |---|---|
    | 1 | Load and prepare your scraped article dataset |
    | 2 | Define a domain ontology for AI copyright/governance |
    | 3 | Extract entities + relationships WITH descriptions using a custom extractor |
    | 4 | Build a knowledge graph with community detection baked in |
    | 5 | Generate LLM community summaries |
    | 6 | Visualize the graph interactively |
    | 7 | Query using GraphRAG |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 0. Install Dependencies
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: llama-index llama-index-llms-openai graspologic pandas nest-asyncio python-dotenv !pip install llama-index llama-index-llms-openai graspologic pandas nest-asyncio python-dotenv
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. Imports & Configuration
    """)
    return


@app.cell
def _():
    import asyncio
    import nest_asyncio
    import pandas as pd
    import networkx as nx
    from dotenv import load_dotenv

    from llama_index.core import Document, PropertyGraphIndex, Settings
    from llama_index.core.graph_stores.types import (
        EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
    )
    from llama_index.core.graph_stores import SimplePropertyGraphStore
    from llama_index.core.llms.llm import LLM
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.schema import TransformComponent, BaseNode
    from llama_index.core.async_utils import run_jobs
    from llama_index.core.query_engine import CustomQueryEngine
    from llama_index.llms.openai import OpenAI
    from graspologic.partition import hierarchical_leiden

    # Patch the event loop so LlamaIndex async calls work inside Jupyter
    nest_asyncio.apply()

    # Load API keys from .env file (create one with OPENAI_API_KEY=sk-...)
    load_dotenv()

    print("✅ Imports ready")
    return (
        BaseNode,
        CustomQueryEngine,
        Document,
        EntityNode,
        KG_NODES_KEY,
        KG_RELATIONS_KEY,
        LLM,
        OpenAI,
        PromptTemplate,
        PropertyGraphIndex,
        Relation,
        Settings,
        SimplePropertyGraphStore,
        TransformComponent,
        asyncio,
        hierarchical_leiden,
        nx,
        pd,
        run_jobs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. Configuration

    We use two different models for different parts of the pipeline:

    | Step | Model | Why |
    |---|---|---|
    | Entity & relationship extraction | `gpt-4o-mini` | High-volume, repetitive, cheap and fast for extraction. |
    | Community summarization | `gpt-4o-mini` | Summarization is straightforward, so mini handles it well. |
    | Final query synthesis | `gpt-4o` | Reasoning quality matters here, so larger model is better. |
    """)
    return


@app.cell
def _(OpenAI, Settings):
    # ── LLM Configuration ─────────────────────────────────────────────────────────
    EXTRACTION_LLM = OpenAI(model="gpt-4o-mini", temperature=0)  # For extraction + summaries
    QUERY_LLM      = OpenAI(model="gpt-4o",      temperature=0)  # For final answer synthesis

    # ── Pipeline Parameters ────────────────────────────────────────────────────────
    MAX_ARTICLES        = 50    # Number of articles to process
    MAX_PATHS_PER_CHUNK = 20    # Max entity-relationship-entity triplet|s extracted per chunk
    NUM_WORKERS         = 4     # Parallel workers for extraction (faster, same cost)
    MAX_CLUSTER_SIZE    = 10    # Max entities per community cluster (controls how big the clusters should be)
    GRAPH_OUTPUT_FILE   = "ai_copyright_graph.html"

    Settings.llm = EXTRACTION_LLM

    print(f"✅ Using {EXTRACTION_LLM.model} for extraction, {QUERY_LLM.model} for querying")
    return (
        EXTRACTION_LLM,
        GRAPH_OUTPUT_FILE,
        MAX_CLUSTER_SIZE,
        MAX_PATHS_PER_CHUNK,
        NUM_WORKERS,
        QUERY_LLM,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Define the Ontology

    The ontology is the **schema of our knowledge graph**. It controls exactly what
    kinds of entities and relationships the LLM is allowed to extract.

    ### Why it matters

    Without a defined schema, the LLM will:
    - Invent different entity types per document ("AI_COMPANY" vs "TECH_FIRM" vs "ORGANIZATION")
    - Use inconsistent relationship labels for the same concept
    - Create duplicate nodes for the same entity under slightly different names

    The ontology constrains extraction to a fixed vocabulary, so the graph stays
    consistent across all documents.

    ### For this particular use case - AI copyright & governance

    I defined below 7 entity types, 8 relationship type.

    But this really depends on your use case and what you want to research.
    """)
    return


@app.cell
def _():
    # ── Entity types ──────────────────────────────────────────────────────────────
    # We'll embed these directly into the extraction prompt below.

    ENTITY_TYPES = [
        "ORGANIZATION",  # Companies, labs, industry groups (OpenAI, Google, RIAA)
        "PERSON",        # Executives, policymakers, judges (Sam Altman, Thierry Breton)
        "LEGISLATION",   # Laws, acts, regulations (EU AI Act, DMCA, Copyright Act)
        "LEGAL_CASE",    # Lawsuits, court rulings (NYT v. OpenAI, Getty v. Stability AI)
        "CONCEPT",       # Abstract ideas: fair use, training data, IP rights
        "GOVERNMENT",    # Nations, regulatory bodies, courts (EU, US Copyright Office)
        "AI_SYSTEM",     # Specific models or products (GPT-4, Stable Diffusion, Gemini)
    ]

    # ── Relationship types ─────────────────────────────────────────────────────────
    RELATION_TYPES = [
        "FILED_AGAINST",  # Plaintiff Organization/Person → LEGAL_CASE
        "DEFENDANT_IN",   # Organization/Person → LEGAL_CASE
        "REGULATES",      # GOVERNMENT/LEGISLATION → ORGANIZATION/AI_SYSTEM
        "ADVOCATES_FOR",  # ORGANIZATION/PERSON → CONCEPT or policy position
        "TRAINED_ON",     # AI_SYSTEM → CONCEPT or dataset type
        "PART_OF",        # PERSON → ORGANIZATION
        "REFERENCES",     # LEGAL_CASE/LEGISLATION → CONCEPT
        "OPPOSES",        # ORGANIZATION/GOVERNMENT → LEGISLATION/CONCEPT
    ]

    print("✅ Ontology:")
    print(f"   Entity types:       {ENTITY_TYPES}")
    print(f"   Relationship types: {RELATION_TYPES}")
    return ENTITY_TYPES, RELATION_TYPES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Extraction Prompt

    This is where our ontology gets embedded into the LLM's instructions.

    Most GraphRAG implementations extract bare triplets: `(OpenAI, DEFENDANT_IN, NYT v. OpenAI)`.

    This prompt extracts **descriptions alongside every entity and relationship**:
    - Entity description: *"OpenAI is an AI research company that developed GPT-4, currently facing multiple copyright lawsuits"*
    - Relationship description: *"OpenAI is named as defendant in the New York Times lawsuit over allegedly using copyrighted articles as training data"*

    These descriptions flow through to the community summaries, making them far richer
    and enabling much better final answers.
    """)
    return


@app.cell
def _(ENTITY_TYPES, RELATION_TYPES):
    # Build the entity and relationship type strings dynamically from our ontology
    entity_types_str   = ", ".join(ENTITY_TYPES)
    relation_types_str = ", ".join(RELATION_TYPES)

    KG_TRIPLET_EXTRACT_TMPL = f"""
    -Goal-
    Given a news article about AI copyright, governance, or intellectual property,
    identify all entities mentioned in the article and their relationships.

    Extract up to {{max_knowledge_triplets}} entity-relation triplets.

    -Allowed Entity Types-
    {entity_types_str}

    -Allowed Relationship Types-
    {relation_types_str}

    -Steps-
    1. Identify ALL entities. For each entity extract:
       - name: Name of the entity, capitalized
       - type: One of the allowed entity types above
       - description: A brief description of the entity and its role in AI copyright/governance

    2. Identify relationships between entities. For each pair extract:
       - source: name of the source entity
       - target: name of the target entity
       - relation: one of the allowed relationship types above
       - description: a sentence explaining why and how these entities are related

    -Real Data-
    ######################
    text: {{text}}
    ######################
    """

    print("✅ Extraction prompt ready")
    print(f"\nPreview (first 300 chars):\n{KG_TRIPLET_EXTRACT_TMPL[:300]}...")
    return (KG_TRIPLET_EXTRACT_TMPL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5. Pydantic Extraction Models

    Instead of parsing raw LLM text with regex, we define Pydantic models that describe
    exactly what we want the LLM to return. LlamaIndex passes these to OpenAI as a
    function schema, so the response is structured JSON — validated and typed automatically.

    Three models are defined:

    - **`ExtractedEntity`** — a single entity with `name`, `type`, and `description`
    - **`ExtractedRelationship`** — a relationship between two entities: `source`, `target`, `relation`, `description`
    - **`ExtractionResult`** — the top-level wrapper containing a list of each

    `EntityTypeStr` and `RelationTypeStr` are `Literal` types built from the ontology defined in the previous cell.
    This means the LLM *cannot* return an invalid type — Pydantic will reject it before it ever reaches our code.
    """)
    return


@app.cell
def _():
    from pydantic import BaseModel, Field, field_validator
    from typing import Literal, List

    EntityTypeStr = Literal[
        "ORGANIZATION", "PERSON", "LEGISLATION", "LEGAL_CASE",
        "CONCEPT", "GOVERNMENT", "AI_SYSTEM"
    ]
    RelationTypeStr = Literal[
        "FILED_AGAINST", "DEFENDANT_IN", "REGULATES", "ADVOCATES_FOR",
        "TRAINED_ON", "PART_OF", "REFERENCES", "OPPOSES"
    ]

    class ExtractedEntity(BaseModel):
        name: str = Field(description="Name of the entity, capitalized")
        type: EntityTypeStr = Field(description="One of the allowed entity types")
        description: str = Field(description="Brief description of the entity and its role")

    class ExtractedRelationship(BaseModel):
        source: str = Field(description="Name of the source entity")
        target: str = Field(description="Name of the target entity")
        relation: RelationTypeStr = Field(description="One of the allowed relationship types")
        description: str = Field(description="Sentence explaining the relationship")

    class ExtractionResult(BaseModel):
        entities: List[ExtractedEntity] = Field(default_factory=list)
        relationships: List[ExtractedRelationship] = Field(default_factory=list)

    print("✅ Pydantic extraction models defined")
    return ExtractionResult, Field, List, field_validator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6. GraphRAGExtractor

    This is the core extraction component. It sends each text chunk to the LLM
    with our ontology-constrained prompt, parses the response, and stores the
    extracted entities and relationships as structured objects on each node.

    ### Why build a custom extractor?

    LlamaIndex's built-in `SchemaLLMPathExtractor` extracts entity/relationship labels
    but **drops descriptions**. By building our own extractor:

    - Every `EntityNode` carries a `entity_description` property
    - Every `Relation` carries a `relationship_description` property
    - These flow through to community summaries, making them far richer

    The extractor runs **asynchronously** with `num_workers=4` — processing 4 chunks
    in parallel, which speeds up extraction significantly without increasing API cost.
    """)
    return


@app.cell
def _(
    BaseNode,
    EntityNode,
    ExtractionResult,
    Field,
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    KG_TRIPLET_EXTRACT_TMPL,
    LLM,
    PromptTemplate,
    Relation,
    Settings,
    TransformComponent,
    asyncio,
    field_validator,
    run_jobs,
):
    class GraphRAGExtractor(TransformComponent):
        """
        Extracts entities and relationships WITH descriptions from each text chunk.

        Uses Pydantic structured output (via OpenAI function calling) to guarantee
        that every entity has a valid type and every relationship uses an allowed label.

        Data flow for a single chunk:
          Raw text
            → LLM (astructured_predict)
            → ExtractionResult(
                  entities=[
                      ExtractedEntity(name="OpenAI", type="ORGANIZATION", description="AI lab..."),
                      ExtractedEntity(name="NYT v. OpenAI", type="LEGAL_CASE", description="Copyright lawsuit..."),
                  ],
                  relationships=[
                      ExtractedRelationship(source="OpenAI", target="NYT v. OpenAI",
                                            relation="DEFENDANT_IN", description="OpenAI is named defendant..."),
                  ]
              )
            → EntityNode(name="OpenAI", label="ORGANIZATION", properties={...})
            → EntityNode(name="NYT v. OpenAI", label="LEGAL_CASE", properties={...})
            → Relation(label="DEFENDANT_IN", source_id=<OpenAI id>, target_id=<NYT v. OpenAI id>)
        """

        llm: LLM = Field(default_factory=lambda: Settings.llm)
        extract_prompt: PromptTemplate = Field(default_factory=lambda: PromptTemplate(KG_TRIPLET_EXTRACT_TMPL))
        num_workers: int = 4
        max_paths_per_chunk: int = 10

        @field_validator("extract_prompt", mode="before")
        @classmethod
        def coerce_to_prompt_template(cls, v):
            return PromptTemplate(v) if isinstance(v, str) else v

        def __call__(self, nodes, show_progress=False, **kwargs):
            return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

        async def _aextract(self, node: BaseNode) -> BaseNode:
            text = node.get_content(metadata_mode="llm")

            # ── Step 1: Call the LLM ───────────────────────────────────────────────
            # astructured_predict uses OpenAI function calling to return a validated ExtractionResult format
            # so entity types and relation labels are guaranteed valid.
            try:
                result = await self.llm.astructured_predict(
                    ExtractionResult,
                    self.extract_prompt,
                    text=text,
                    max_knowledge_triplets=self.max_paths_per_chunk,
                )
                entities      = result.entities
                relationships = result.relationships
        
            except Exception as e:
                print(f"Extraction error: {e}")
                entities, relationships = [], []

            existing_nodes     = node.metadata.pop(KG_NODES_KEY, [])
            existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
            base_metadata      = node.metadata.copy()

            # ── Step 2: Entities → EntityNode objects ─────────────────────────────
            # e.g. ExtractedEntity(name="OpenAI", type="ORGANIZATION", description="...")
            # becomes  EntityNode(name="OpenAI", label="ORGANIZATION", properties={...})
            existing_nodes += [
                EntityNode(
                    name=entity.name,
                    label=entity.type,
                    properties={**base_metadata, "entity_description": entity.description},
                )
                for entity in entities
            ]

            # ── Step 3: Relationships → Relation objects ───────────────────────────
            # entity_lookup lets us attach the correct label to relationship source and target nodes.
            # e.g. {"OpenAI": "ORGANIZATION", "NYT v. OpenAI": "LEGAL_CASE"}
            entity_lookup = {e.name: e.type for e in entities}

            for rel in relationships:
                source_node = EntityNode(
                    name=rel.source,
                    label=entity_lookup.get(rel.source, "ENTITY"),
                    properties=base_metadata,
                )
                target_node = EntityNode(
                    name=rel.target,
                    label=entity_lookup.get(rel.target, "ENTITY"),
                    properties=base_metadata,
                )
                # Only add endpoint nodes not already captured from the entity pass
                if rel.source not in entity_lookup:
                    existing_nodes.append(source_node)
                if rel.target not in entity_lookup:
                    existing_nodes.append(target_node)
                existing_relations.append(Relation(
                    label=rel.relation,
                    source_id=source_node.id,
                    target_id=target_node.id,
                    properties={**base_metadata, "relationship_description": rel.description},
                ))

            node.metadata[KG_NODES_KEY]     = existing_nodes
            node.metadata[KG_RELATIONS_KEY] = existing_relations
            return node

        async def acall(self, nodes, show_progress=False, **kwargs):
            """Process all nodes in parallel using num_workers async workers."""
            jobs = [self._aextract(node) for node in nodes]
            return await run_jobs(
                jobs,
                workers=self.num_workers,
                show_progress=show_progress,
                desc="Extracting triplets",
            )

    print("✅ GraphRAGExtractor defined")
    return (GraphRAGExtractor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7. GraphRAGStore

    `GraphRAGStore` extends LlamaIndex's `SimplePropertyGraphStore` with two additional
    capabilities: **community detection** and **community summary generation**.

    By bundling these into the store itself, the pipeline stays clean — after building
    the index you just call `graph_store.build_communities()` and everything is handled.

    ### How community detection works here

    1. Convert the property graph to a NetworkX graph
    2. Run hierarchical Leiden to find entity clusters
    3. For each cluster, collect all entities (+ descriptions) and relationships (+ descriptions)
    4. Ask the LLM to write a briefing note for each cluster

    The descriptions captured during extraction make these briefings significantly
    richer than if we'd only stored bare labels.
    """)
    return


@app.cell
def _(
    EXTRACTION_LLM,
    EntityNode,
    MAX_CLUSTER_SIZE,
    SimplePropertyGraphStore,
    hierarchical_leiden,
    nx,
):
    class GraphRAGStore(SimplePropertyGraphStore):
        """
        Extends SimplePropertyGraphStore with:
        - Leiden community detection
        - LLM-generated community summaries (using entity + relationship descriptions)

        After building the PropertyGraphIndex, call build_communities() to
        detect clusters and generate summaries. These summaries are then
        used by GraphRAGQueryEngine at query time.
        """

        community_summaries: dict = {}

        def build_communities(self):
            """
            Main entry point: run community detection and generate summaries.
            Call this once after PropertyGraphIndex is built.
            """
            print("Running community detection...")
            nx_graph = self._to_networkx()

            if not nx_graph.nodes:
                print("⚠️  Graph is empty — no communities to detect")
                return

            print(f"Graph has {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")

            # Run hierarchical Leiden algorithm
            clusters = hierarchical_leiden(nx_graph, max_cluster_size=MAX_CLUSTER_SIZE)
            num_communities = len(set(c.cluster for c in clusters))
            print(f"Found {num_communities} communities")

            # Collect entities and relationships per community
            community_info = self._collect_community_info(nx_graph, clusters)

            # Generate LLM summaries
            self._generate_summaries(community_info)
            print(f"\n✅ {len(self.community_summaries)} community summaries generated")

        def _to_networkx(self) -> nx.Graph:
            """
            Convert the LlamaIndex property graph to a NetworkX graph for Graspologic.
            Edges carry relationship label and description as attributes.
            """
            nx_graph = nx.Graph()

            # Add entity nodes
            for node in self.graph.nodes.values():
                if isinstance(node, EntityNode):
                    nx_graph.add_node(node.id)

            # Add edges with relationship metadata
            for relation in self.graph.relations.values():
                if relation.source_id in nx_graph and relation.target_id in nx_graph:
                    nx_graph.add_edge(
                        relation.source_id,
                        relation.target_id,
                        relationship=relation.label,
                        description=relation.properties.get("relationship_description", ""),
                    )

            return nx_graph

        def _collect_community_info(self, nx_graph, clusters) -> dict:
            """
            For each community cluster, collect:
            - Entity names, types, and descriptions
            - Relationship strings (with descriptions where available)

            This rich data is what makes the LLM summaries informative.
            """
            community_mapping = {item.node: item.cluster for item in clusters}

            # Build a lookup of node id -> {name, type, description}
            node_details = {}
            for node in self.graph.nodes.values():
                if not isinstance(node, EntityNode):
                    continue
                node_details[node.id] = {
                    "name":        node.name,
                    "type":        node.label,
                    "description": node.properties.get("entity_description", ""),
                }

            community_info = {}
            for item in clusters:
                cid, nid = item.cluster, item.node
                community_info.setdefault(cid, {"entities": [], "relationships": []})

                # Add entity details to community
                if nid in node_details:
                    community_info[cid]["entities"].append(node_details[nid])

                # Add relationships to neighbors in the same community
                for neighbor in nx_graph.neighbors(nid):
                    if community_mapping.get(neighbor) == cid:
                        edge = nx_graph.get_edge_data(nid, neighbor)
                        rel  = edge.get("relationship", "RELATED") if edge else "RELATED"
                        desc = edge.get("description",  "")        if edge else ""

                        src_name = node_details.get(nid,      {}).get("name", nid)
                        tgt_name = node_details.get(neighbor, {}).get("name", neighbor)

                        # Format: "Entity A --[RELATION]--> Entity B (description)"
                        entry = f"{src_name} --[{rel}]--> {tgt_name}"
                        if desc:
                            entry += f" ({desc})"
                        community_info[cid]["relationships"].append(entry)

            return community_info

        def _generate_summaries(self, community_info):
            """
            Ask the LLM to write a briefing note for each community cluster.
            Uses entity descriptions and relationship descriptions for richer context.
            """
            for community_id, data in community_info.items():
                if not data["relationships"] and not data["entities"]:
                    continue

                # Format entities with type and description
                entities_text = "\n".join([
                    f"- {e['name']} ({e['type']}): {e['description']}"
                    for e in data["entities"] if e.get("name")
                ])

                # Deduplicate and format relationships
                relationships_text = "\n".join(sorted(set(data["relationships"])))

                prompt = f"""You are analysing a cluster of entities from news articles about
                AI copyright, governance, and intellectual property.

                Entities in this cluster:
                {entities_text}

                Relationships:
                {relationships_text}

                Write a concise briefing (3-5 sentences) that:
                1. Identifies the main organizations, people, legal cases, or topics in this cluster
                2. Explains how they are connected and why — including legal or regulatory context
                3. Highlights any disputes, lawsuits, policy positions, or tensions
                4. Notes anything particularly relevant for understanding AI copyright or governance

                Briefing:"""

                response = EXTRACTION_LLM.complete(prompt)
                self.community_summaries[community_id] = response.text
                print(f"  Community {community_id}: {response.text[:100]}...")

        def get_community_summaries(self) -> dict:
            return self.community_summaries

    print("✅ GraphRAGStore defined")
    return (GraphRAGStore,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8. GraphRAGQueryEngine

    The query engine uses a two-step approach:

    1. **Per-community answering** — ask the LLM to answer the question from each
       community summary independently. If a summary isn't relevant, the LLM says so
       and we skip it. This avoids polluting the final answer with irrelevant content.

    2. **Aggregation** — combine all relevant partial answers into one final,
       non-redundant response using `QUERY_LLM` (the stronger model).
    """)
    return


@app.cell
def _(CustomQueryEngine, EXTRACTION_LLM, GraphRAGStore, LLM, List):
    class GraphRAGQueryEngine(CustomQueryEngine):
        """
        Queries all community summaries and synthesises a single answer.

        Step 1: For each community summary, ask EXTRACTION_LLM to answer the
                question based only on that summary. If not relevant, skip.
        Step 2: Aggregate all relevant partial answers using QUERY_LLM into
                one final, clear, non-redundant response.
        """

        graph_store: GraphRAGStore
        llm: LLM

        def custom_query(self, query_str: str) -> str:
            summaries = self.graph_store.get_community_summaries()

            if not summaries:
                return "No community summaries found. Run graph_store.build_communities() first."

            # Step 1: Get a partial answer from each community summary
            community_answers = [
                self._answer_from_community(summary, query_str)
                for summary in summaries.values()
            ]

            # Filter out empty/irrelevant responses
            relevant_answers = [a for a in community_answers if a.strip()]

            if not relevant_answers:
                return "I don't have enough information in the knowledge graph to answer that question."

            # Step 2: Aggregate into one final answer
            return self._aggregate(relevant_answers, query_str)

        def _answer_from_community(self, summary: str, query: str) -> str:
            """
            Ask EXTRACTION_LLM to answer the query from a single community summary.
            Returns empty string if the summary isn't relevant to the question.
            We use the cheaper model here since this runs once per community.
            """
            prompt = (
                f"Community summary:\n{summary}\n\n"
                f"Question: {query}\n\n"
                f"If this summary contains information relevant to the question, answer it. "
                f"If not relevant, reply exactly: 'No relevant information.'\n\n"
                f"Answer:"
            )
            response = EXTRACTION_LLM.complete(prompt)
            text = response.text.strip()

            # Filter out non-answers
            return "" if "no relevant information" in text.lower() else text

        def _aggregate(self, answers: List[str], query: str) -> str:
            """
            Synthesise all relevant partial answers into one final response.
            Uses QUERY_LLM (gpt-4o) for better reasoning quality on the final step.
            """
            combined = "\n\n---\n\n".join(answers)
            prompt = (
                f"You have received answers from multiple knowledge graph communities about this question:\n\n"
                f"Question: {query}\n\n"
                f"Community answers:\n{combined}\n\n"
                f"Synthesise these into a single, clear, well-structured final answer. "
                f"Remove redundancy, keep all important details, and ensure the answer "
                f"directly addresses the question.\n\n"
                f"Final Answer:"
            )
            return self.llm.complete(prompt).text

    print("✅ GraphRAGQueryEngine defined")
    return (GraphRAGQueryEngine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 9. Load Article Dataset

    Load the articles previously scraped with SerpApi.
    """)
    return


@app.cell
def _(pd):
    # Load from CSV ───────────────────────────────────────────────────
    df = pd.read_csv("ai_copyright_dataset.csv")
    print(f"Number of articles: {len(df)}")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 10. Wrap Articles as Documents

    We wrap each article as a single LlamaIndex `Document` — no chunking needed since
    the articles are short enough to fit in the LLM's context window.
    """)
    return


@app.cell
def _(Document, df):
    # Wrap each article as a LlamaIndex Document
    # full_text is the main content; everything else is metadata
    nodes = [
        Document(
            text=str(row["full_text"]),
            metadata={
                "title":  str(row.get("title",  "")),
                "source": str(row.get("source", "")),
                "date":   str(row.get("date",   "")),
            }
        )
        for _, row in df.iterrows()
    ]

    print(f"✅ Created {len(nodes)} LlamaIndex documents, stored as nodes")
    return (nodes,)


@app.cell
def _(nodes):
    print(nodes[3].text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 11. Build the Knowledge Graph

    Now we wire everything together and run the extraction pipeline.

    `PropertyGraphIndex` handles the full workflow:
    1. Passes each chunk to `GraphRAGExtractor`
    2. The extractor calls the LLM with our ontology-constrained prompt
    3. Parsed entities and relationships are stored in `GraphRAGStore`

    -> This is the most time-consuming step!
    """)
    return


@app.cell
def _(
    EXTRACTION_LLM,
    GraphRAGExtractor,
    GraphRAGStore,
    KG_TRIPLET_EXTRACT_TMPL,
    MAX_PATHS_PER_CHUNK,
    NUM_WORKERS,
    PropertyGraphIndex,
    nodes,
):
    # Instantiate the extractor with our ontology prompt
    kg_extractor = GraphRAGExtractor(
        llm=EXTRACTION_LLM,
        extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
        max_paths_per_chunk=MAX_PATHS_PER_CHUNK,
        num_workers=NUM_WORKERS,
    )

    # GraphRAGStore is both the graph database and the community detection engine
    graph_store = GraphRAGStore()

    print("Building knowledge graph... (this may take a few minutes)")

    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        embed_kg_nodes=False,             # we don't need vector similarity search for this GraphRAG pipeline
        show_progress=True,
    )
    return graph_store, kg_extractor


@app.cell
async def _(KG_NODES_KEY, KG_RELATIONS_KEY, kg_extractor, nodes):
    # Print out example of an article and the extracted entities & relationships
    import copy
    sample = nodes[3]
    # Change this index to inspect a different article
    print('=== Title ===')
    print(sample.metadata.get('title', 'untitled'))
    print()
    print('=== Raw text ===')
    print(sample.text)
    print()
    extracted = await kg_extractor._aextract(copy.deepcopy(sample))
    print('=== Extracted entities ===')
    for _node in extracted.metadata.get(KG_NODES_KEY, []):
    # Re-run extraction on a copy so we don't mutate the original
        print(f'  [{_node.label}] {_node.name}')
        print(f"    {_node.properties.get('entity_description', '')}")
    print()
    print('=== Extracted relationships ===')
    id_to_name = {n.id: n.name for n in extracted.metadata.get(KG_NODES_KEY, [])}
    for rel in extracted.metadata.get(KG_RELATIONS_KEY, []):
        _src = id_to_name.get(rel.source_id, rel.source_id)
        _tgt = id_to_name.get(rel.target_id, rel.target_id)
        print(f'  {_src} --[{rel.label}]--> {_tgt}')
    return


@app.cell
def _(EntityNode, graph_store):
    # ── Print out unique entities extracted ───────────────────────────────────────────────────
    from collections import defaultdict
    by_type = defaultdict(list)
    for _node in graph_store.graph.nodes.values():
        if isinstance(_node, EntityNode):
            by_type[_node.label].append(_node.name)
    for entity_type, names in sorted(by_type.items()):
        print(f'\n{entity_type} ({len(names)})')
        for name in sorted(names):
            print(f'  {name}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 12. Build Communities & Generate Summaries

    Now we run community detection and generate LLM summaries for each cluster.

    This is the step that enables big-picture queries. Instead of searching for
    similar chunks, GraphRAG queries these pre-built community briefings — each one
    a synthesised overview of a topic cluster in your data.

    For our AI copyright dataset, you should see communities forming around things like:
    - US copyright litigation (NYT, Getty, authors suing AI companies)
    - EU regulatory activity (EU AI Act, GDPR interactions)
    - Training data debates (fair use arguments, dataset licensing)
    - Specific AI systems and their legal exposure
    """)
    return


@app.cell
def _(graph_store):
    graph_store.build_communities()

    summaries = graph_store.get_community_summaries()
    print(f"\n✅ {len(summaries)} community summaries ready for querying")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 13. Visualize the Knowledge Graph

    Create a d3.js network graph visualisation from the graph we built.

    Open `ai_copyright_graph.html` in your browser after running this cell.
    """)
    return


@app.cell
def _(EntityNode, GRAPH_OUTPUT_FILE, GraphRAGStore, graph_store):
    import json
    import pathlib


    def export_graph_data(graph_store: GraphRAGStore, output_file: str = "graph_data.json"):
        """
        Export the knowledge graph to a JSON file compatible with the D3.js template.
        Run this once after graph_store.build_communities().

        Saves to disk so you can re-run the visualization without reprocessing
        the full pipeline — which is expensive.
        """

        nx_graph = graph_store._to_networkx()

        # ── Build node metadata lookup ─────────────────────────────────────────
        node_meta = {}
        for node in graph_store.graph.nodes.values():
            if isinstance(node, EntityNode):
                node_meta[node.id] = {
                    "id":          node.id,
                    "label":       node.name,
                    "type":        node.label if node.label else "OTHER",
                    "description": node.properties.get("entity_description", ""),
                }

        # ── Nodes — only those present in the NetworkX graph ──────────────────
        nodes_data = [
            node_meta[n] for n in nx_graph.nodes() if n in node_meta
        ]

        # ── Links ──────────────────────────────────────────────────────────────
        links_data = []
        for source, target, data in nx_graph.edges(data=True):
            links_data.append({
                "source":      source,
                "target":      target,
                "label":       data.get("relationship", ""),
                "description": data.get("description",  ""),
            })

        # ── Assemble and write ─────────────────────────────────────────────────
        graph_data = {
            "nodes":       nodes_data,
            "links":       links_data,
            "communities": len(graph_store.get_community_summaries()),
        }

        pathlib.Path(output_file).write_text(json.dumps(graph_data, indent=2))

        print(f"✅ Graph data exported to '{output_file}'")
        print(f"   Nodes:       {len(nodes_data)}")
        print(f"   Edges:       {len(links_data)}")
        print(f"   Communities: {graph_data['communities']}")

        return graph_data


    def visualize_graph(
        graph_store: GraphRAGStore = None,
        graph_data_file: str = "graph_data.json",
        template_file: str = "graph_template.html",
        output_file: str = GRAPH_OUTPUT_FILE,
    ):
        """
        Generate a D3.js interactive knowledge graph visualization.

        Can be called in two ways:

        1. Pass graph_store directly — exports fresh data then builds the HTML:
             visualize_graph(graph_store=graph_store)

        2. Load from a previously exported JSON file — no pipeline re-run needed:
             visualize_graph()  # loads graph_data.json automatically

        Features:
        - Color-coded nodes by entity type (from ontology)
        - Node size scales with degree (more connections = bigger)
        - Click a node to highlight its neighbourhood, dim everything else
        - Sidebar legend — click any entity type to toggle its visibility
        - Search bar to find and highlight any node by name
        - Sliders for link distance, charge strength, and edge label threshold
        - Hover tooltips with entity name, type, description, and connection count
        - Stats header showing total nodes, edges, and communities
        """

        # ── Get graph data ─────────────────────────────────────────────────────
        if graph_store is not None:
            # Export fresh data from the graph store and save to disk
            graph_data = export_graph_data(graph_store, graph_data_file)
        else:
            # Load from a previously saved JSON file
            data_path = pathlib.Path(graph_data_file)
            if not data_path.exists():
                print(f"⚠️  '{graph_data_file}' not found.")
                print(f"   Run export_graph_data(graph_store) first, or pass graph_store directly.")
                return
            graph_data = json.loads(data_path.read_text())

            print(f"✅ Loaded graph data from '{graph_data_file}'")
            print(f"   Nodes: {len(graph_data['nodes'])}  |  "
                  f"Edges: {len(graph_data['links'])}  |  "
                  f"Communities: {graph_data.get('communities', '—')}")

        # ── Load HTML template ─────────────────────────────────────────────────
        template_path = pathlib.Path(template_file)
        if not template_path.exists():
            print(f"⚠️  '{template_file}' not found.")
            print(f"   Make sure graph_template.html is in the same folder as this notebook.")
            return

        # ── Inject data into template and write output ─────────────────────────
        html = template_path.read_text()
        html = html.replace("GRAPH_DATA_PLACEHOLDER", json.dumps(graph_data))
        pathlib.Path(output_file).write_text(html)

        print(f"\n✅ Visualization saved to '{output_file}'")
        print(f"   Open it in your browser to explore.")


    # Export data and build visualization from the graph store
    visualize_graph(graph_store=graph_store)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 14. Query the System

    Now we run queries that would be impossible to answer reliably with standard RAG.

    The `GraphRAGQueryEngine` scores each community summary for relevance, filters
    irrelevant ones, and synthesizes a final answer from the best results — all using
    the community briefings generated in Step 12.
    """)
    return


@app.cell
def _(GraphRAGQueryEngine, QUERY_LLM, graph_store):
    query_engine = GraphRAGQueryEngine(
        graph_store=graph_store,
        llm=QUERY_LLM,  # gpt-4o for final synthesis
    )

    print("✅ Query engine ready")
    return (query_engine,)


@app.cell
def _(query_engine):
    # ── Query 1: Big-picture thematic question ────────────────────────────────────
    # Requires synthesising the main legal arguments across many articles.
    # Standard RAG would struggle — no single chunk contains the full picture.

    q1 = "What are the main legal arguments being made around AI copyright and training data?"
    print(f"Query: {q1}")
    print("=" * 70)
    print(query_engine.custom_query(q1))
    return


@app.cell
def _(query_engine):
    # ── Query 2: Cross-entity relationship question ────────────────────────────────
    # Requires tracing which companies appear in disputes and what positions they hold.
    # GraphRAG can follow DEFENDANT_IN, ADVOCATES_FOR, OPPOSES edges to answer this.

    q2 = "Which companies are involved in AI copyright or governance disputes, and what are their positions?"
    print(f"Query: {q2}")
    print("=" * 70)
    print(query_engine.custom_query(q2))
    return


@app.cell
def _(query_engine):
    # ── Query 3: Comparative policy question ──────────────────────────────────────
    # Requires reasoning across different governments and jurisdictions —
    # connecting EU, US, and UK regulatory approaches from different articles.

    q3 = "How are different governments (e.g. EU, US, and UK) approaching AI governance?"
    print(f"Query: {q3}")
    print("=" * 70)
    print(query_engine.custom_query(q3))
    return


@app.cell
def _(query_engine):
    # ── Query 4: Try your own question ────────────────────────────────────────────
    your_question = "What is the role of fair use in AI copyright disputes?"

    print(f"Query: {your_question}")
    print("=" * 70)
    print(query_engine.custom_query(your_question))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 15. Summary

    ### What we built

    | Step | Component | What it does |
    |---|---|---|
    | Ontology | Custom prompt | Constrains LLM to 7 entity types + 8 relationship types |
    | Extraction | `GraphRAGExtractor` | Extracts entities + relationships WITH descriptions |
    | Graph store | `GraphRAGStore` | Stores graph + runs Leiden community detection |
    | Summaries | `_generate_summaries()` | LLM briefings per community cluster |
    | Visualization | D3.js | Interactive color-coded graph by entity type |
    | Querying | `GraphRAGQueryEngine` | Per-community answering + synthesis |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
