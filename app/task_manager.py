"""
Async build task manager — tracks in-progress and completed graph builds.

Build tasks are expensive (minutes). This module launches each build as an
asyncio background task and exposes its status for the API polling endpoint.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from llama_index.llms.openai import OpenAI

from .config import Settings
from .graph_store import GraphRAGStore
from .models import OntologyConfig, TaskState
from .parser import DocumentParser
from .pipeline import build_topic_graph

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self, config: Settings):
        self.config = config
        self._tasks: dict[str, TaskState] = {}
        self._stores: dict[str, GraphRAGStore] = {}  # built store cache

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_status(self, topic: str) -> Optional[TaskState]:
        return self._tasks.get(topic)

    def get_store(self, topic: str) -> Optional[GraphRAGStore]:
        return self._stores.get(topic)

    def invalidate_store(self, topic: str) -> None:
        self._stores.pop(topic, None)

    async def start_build(
        self,
        topic: str,
        ontology: Optional[OntologyConfig],
        query_engines: dict,
    ) -> None:
        """
        Launch a background build task for the given topic.
        If a build is already running, raises RuntimeError.
        """
        existing = self._tasks.get(topic)
        if existing and existing.status == "building":
            raise RuntimeError(f"Build already in progress for topic '{topic}'")

        self._tasks[topic] = TaskState(topic=topic, status="building", progress="Starting...")
        self.invalidate_store(topic)
        query_engines.pop(topic, None)

        asyncio.create_task(
            self._run_build(topic, ontology or OntologyConfig(), query_engines)
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _run_build(
        self,
        topic: str,
        ontology: OntologyConfig,
        query_engines: dict,
    ) -> None:
        def _progress(msg: str) -> None:
            if topic in self._tasks:
                self._tasks[topic].progress = msg

        try:
            config = self.config
            parser = DocumentParser(config.raw_dir)

            _progress("Parsing documents...")
            documents = await asyncio.get_event_loop().run_in_executor(
                None, parser.parse_topic, topic
            )

            if not documents:
                raise ValueError(f"No parseable documents found in raw/{topic}/")

            _progress(f"Parsed {len(documents)} documents — building graph...")

            # Build LLM clients using the configured provider
            extraction_llm = _make_llm(config.extraction_model, config)
            community_llm = extraction_llm
            query_llm = _make_llm(config.query_model, config)

            store = await build_topic_graph(
                topic=topic,
                documents=documents,
                ontology=ontology,
                config=config,
                extraction_llm=extraction_llm,
                community_llm=community_llm,
                progress_callback=_progress,
            )

            self._stores[topic] = store
            self._tasks[topic].status = "complete"
            self._tasks[topic].progress = "Build complete"
            self._tasks[topic].completed_at = datetime.now(timezone.utc)
            logger.info("Build complete for topic '%s'", topic)

        except Exception as exc:
            logger.exception("Build failed for topic '%s': %s", topic, exc)
            if topic in self._tasks:
                self._tasks[topic].status = "error"
                self._tasks[topic].error = str(exc)
                self._tasks[topic].completed_at = datetime.now(timezone.utc)


def _make_llm(model: str, config: Settings):
    """
    Create an LLM client. Supports:
    - OpenAI (gpt-4o, gpt-4o-mini, etc.)
    - Any OpenAI-compatible API (LM Studio, Ollama, etc.) via LLM_BASE_URL
    """
    kwargs = {"model": model, "temperature": 0}

    if config.openai_api_key:
        kwargs["api_key"] = config.openai_api_key

    # Support for local/custom OpenAI-compatible endpoints
    if hasattr(config, "llm_base_url") and config.llm_base_url:
        kwargs["api_base"] = config.llm_base_url

    return OpenAI(**kwargs)
