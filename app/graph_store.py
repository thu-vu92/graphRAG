"""
GraphRAGStore — extends LlamaIndex's SimplePropertyGraphStore with:
  - Leiden community detection
  - LLM-generated community summaries
  - save() / load() for persistence between sessions
  - export_graph_data() for D3.js visualization
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Callable, Optional

import networkx as nx
from graspologic.partition import hierarchical_leiden
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation

logger = logging.getLogger(__name__)


class GraphRAGStore(SimplePropertyGraphStore):
    """
    Property graph store with community detection and LLM summarization.

    Usage:
        store = GraphRAGStore()
        # ... build index via PropertyGraphIndex ...
        store.build_communities(llm=extraction_llm, topic_name="climate_change")
        store.save(Path("graphs/climate_change"))

        # Later:
        store = GraphRAGStore.load(Path("graphs/climate_change"))
    """

    community_summaries: dict = {}

    # ── Community detection ────────────────────────────────────────────────────

    def build_communities(
        self,
        llm,
        topic_name: str = "",
        max_cluster_size: int = 10,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _progress(msg: str) -> None:
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        _progress("Converting graph to NetworkX...")
        nx_graph = self._to_networkx()

        if not nx_graph.nodes:
            _progress("Graph is empty — no communities to detect")
            return

        _progress(
            f"Graph has {nx_graph.number_of_nodes()} nodes, "
            f"{nx_graph.number_of_edges()} edges — running Leiden clustering..."
        )

        clusters = hierarchical_leiden(nx_graph, max_cluster_size=max_cluster_size)
        num_communities = len(set(c.cluster for c in clusters))
        _progress(f"Found {num_communities} communities — generating summaries...")

        community_info = self._collect_community_info(nx_graph, clusters)
        self._generate_summaries(community_info, llm, topic_name, progress_callback)

        _progress(f"Done — {len(self.community_summaries)} community summaries generated")

    def _to_networkx(self) -> nx.Graph:
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            if isinstance(node, EntityNode):
                nx_graph.add_node(node.id)
        for relation in self.graph.relations.values():
            if relation.source_id in nx_graph and relation.target_id in nx_graph:
                nx_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relationship=relation.label,
                    description=relation.properties.get("relationship_description", ""),
                )
        return nx_graph

    def _collect_community_info(self, nx_graph: nx.Graph, clusters) -> dict:
        community_mapping = {item.node: item.cluster for item in clusters}

        node_details: dict[str, dict] = {}
        for node in self.graph.nodes.values():
            if not isinstance(node, EntityNode):
                continue
            node_details[node.id] = {
                "name": node.name,
                "type": node.label,
                "description": node.properties.get("entity_description", ""),
            }

        community_info: dict[int, dict] = {}
        for item in clusters:
            cid, nid = item.cluster, item.node
            community_info.setdefault(cid, {"entities": [], "relationships": []})

            if nid in node_details:
                community_info[cid]["entities"].append(node_details[nid])

            for neighbor in nx_graph.neighbors(nid):
                if community_mapping.get(neighbor) != cid:
                    continue
                edge = nx_graph.get_edge_data(nid, neighbor) or {}
                rel = edge.get("relationship", "RELATED")
                desc = edge.get("description", "")
                src_name = node_details.get(nid, {}).get("name", nid)
                tgt_name = node_details.get(neighbor, {}).get("name", neighbor)
                entry = f"{src_name} --[{rel}]--> {tgt_name}"
                if desc:
                    entry += f" ({desc})"
                community_info[cid]["relationships"].append(entry)

        return community_info

    def _generate_summaries(
        self,
        community_info: dict,
        llm,
        topic_name: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        topic_label = topic_name.replace("_", " ") if topic_name else "the given topic"
        total = len(community_info)

        for idx, (community_id, data) in enumerate(community_info.items(), 1):
            if not data["relationships"] and not data["entities"]:
                continue

            entities_text = "\n".join(
                f"- {e['name']} ({e['type']}): {e['description']}"
                for e in data["entities"]
                if e.get("name")
            )
            relationships_text = "\n".join(sorted(set(data["relationships"])))

            prompt = f"""You are analysing a cluster of related entities extracted from documents about "{topic_label}".

Entities in this cluster:
{entities_text}

Relationships:
{relationships_text}

Write a concise briefing (3-5 sentences) that:
1. Identifies the main entities (people, organizations, concepts, events) in this cluster
2. Explains how they are connected and why
3. Highlights any disputes, collaborations, dependencies, or tensions
4. Notes anything particularly significant for understanding {topic_label}

Briefing:"""

            try:
                response = llm.complete(prompt)
                self.community_summaries[community_id] = response.text
                if progress_callback:
                    progress_callback(f"Summarized community {idx}/{total}")
            except Exception as exc:
                logger.error("Failed to summarize community %s: %s", community_id, exc)

    def get_community_summaries(self) -> dict:
        return self.community_summaries

    # ── Export for D3.js ───────────────────────────────────────────────────────

    def export_graph_data(self) -> dict:
        """
        Return a dict matching the JSON schema expected by the D3.js visualization:
          {nodes: [{id, label, type, description}], links: [{source, target, label, description}], communities: N}
        """
        nx_graph = self._to_networkx()

        node_meta: dict[str, dict] = {}
        for node in self.graph.nodes.values():
            if isinstance(node, EntityNode):
                node_meta[node.id] = {
                    "id": node.id,
                    "label": node.name,
                    "type": node.label or "OTHER",
                    "description": node.properties.get("entity_description", ""),
                }

        nodes_data = [node_meta[n] for n in nx_graph.nodes() if n in node_meta]

        links_data = [
            {
                "source": src,
                "target": tgt,
                "label": data.get("relationship", ""),
                "description": data.get("description", ""),
            }
            for src, tgt, data in nx_graph.edges(data=True)
        ]

        return {
            "nodes": nodes_data,
            "links": links_data,
            "communities": len(self.community_summaries),
        }

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # D3.js visualization data
        graph_data = self.export_graph_data()
        (directory / "graph_data.json").write_text(
            json.dumps(graph_data, indent=2), encoding="utf-8"
        )

        # Community summaries (human-readable, JSON)
        (directory / "communities.json").write_text(
            json.dumps(self.community_summaries, indent=2), encoding="utf-8"
        )

        # Full store pickle (for fast query engine reload)
        with open(directory / "store.pkl", "wb") as fh:
            pickle.dump(self, fh)

        logger.info(
            "Saved graph to %s  (nodes=%d, edges=%d, communities=%d)",
            directory,
            len(graph_data["nodes"]),
            len(graph_data["links"]),
            graph_data["communities"],
        )

    @classmethod
    def load(cls, directory: Path) -> "GraphRAGStore":
        directory = Path(directory)
        pkl_path = directory / "store.pkl"

        # Fast path: pickle
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as fh:
                    store = pickle.load(fh)
                logger.info("Loaded graph store from pickle: %s", pkl_path)
                return store
            except Exception as exc:
                logger.warning("Pickle load failed (%s) — falling back to JSON", exc)

        # Fallback: rebuild from communities.json (no re-extraction needed)
        communities_path = directory / "communities.json"
        if communities_path.exists():
            store = cls()
            store.community_summaries = json.loads(
                communities_path.read_text(encoding="utf-8")
            )
            logger.info("Loaded community summaries from JSON: %s", communities_path)
            return store

        raise FileNotFoundError(
            f"No saved graph found in {directory}. Build it first via POST /api/topics/{{topic}}/build"
        )
