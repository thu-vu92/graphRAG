"""
GraphRAGQueryEngine — two-phase community-based query answering.

Phase 1: Ask the cheaper LLM whether each community summary is relevant.
Phase 2: Aggregate relevant partial answers with the stronger LLM.
"""

from __future__ import annotations

import logging

from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import CustomQueryEngine

from .graph_store import GraphRAGStore

logger = logging.getLogger(__name__)


class GraphRAGQueryEngine(CustomQueryEngine):
    """
    Queries all community summaries and synthesises a single answer.

    graph_store:    the loaded GraphRAGStore containing community_summaries
    llm:            stronger model (e.g. gpt-4o) used for final synthesis
    community_llm:  cheaper model (e.g. gpt-4o-mini) for per-community relevance check
    """

    graph_store: GraphRAGStore
    llm: LLM
    community_llm: LLM

    def custom_query(self, query_str: str) -> tuple[str, int, int]:
        """
        Returns (answer, communities_checked, relevant_communities).
        """
        summaries = self.graph_store.get_community_summaries()

        if not summaries:
            return (
                "No community summaries found. Please build the graph first.",
                0,
                0,
            )

        communities_checked = len(summaries)
        community_answers = [
            self._answer_from_community(summary, query_str)
            for summary in summaries.values()
        ]

        relevant_answers = [a for a in community_answers if a.strip()]
        relevant_communities = len(relevant_answers)

        if not relevant_answers:
            return (
                "I don't have enough information in the knowledge graph to answer that question.",
                communities_checked,
                0,
            )

        answer = self._aggregate(relevant_answers, query_str)
        return answer, communities_checked, relevant_communities

    def _answer_from_community(self, summary: str, query: str) -> str:
        prompt = (
            f"Community summary:\n{summary}\n\n"
            f"Question: {query}\n\n"
            f"If this summary contains information relevant to the question, answer it based only on "
            f"the summary. If not relevant, reply exactly: 'No relevant information.'\n\n"
            f"Answer:"
        )
        try:
            response = self.community_llm.complete(prompt)
            text = response.text.strip()
            return "" if "no relevant information" in text.lower() else text
        except Exception as exc:
            logger.error("Community answer error: %s", exc)
            return ""

    def _aggregate(self, answers: list[str], query: str) -> str:
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
        try:
            return self.llm.complete(prompt).text
        except Exception as exc:
            logger.error("Aggregation error: %s", exc)
            return "\n\n".join(answers)
