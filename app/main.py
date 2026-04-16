"""
FastAPI application — multi-topic GraphRAG web service.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import settings
from .graph_store import GraphRAGStore
from .models import BuildRequest, QueryRequest, QueryResponse, TopicStatus
from .query_engine import GraphRAGQueryEngine
from .task_manager import TaskManager, _make_llm

logger = logging.getLogger(__name__)

app = FastAPI(title="GraphRAG Multi-Topic Explorer", version="1.0.0")

# ── Static files & templates ───────────────────────────────────────────────────
_here = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(_here / "static")), name="static")
templates = Jinja2Templates(directory=str(_here / "templates"))

# ── Singletons ─────────────────────────────────────────────────────────────────
task_manager = TaskManager(settings)
_query_engines: dict[str, GraphRAGQueryEngine] = {}


# ── Helper: discover topics ────────────────────────────────────────────────────

def _discover_topics() -> list[TopicStatus]:
    raw_dir = Path(settings.raw_dir)
    graphs_dir = Path(settings.graphs_dir)
    topics: dict[str, TopicStatus] = {}

    # Topics with raw files
    if raw_dir.exists():
        for d in sorted(raw_dir.iterdir()):
            if d.is_dir():
                topics[d.name] = TopicStatus(
                    topic=d.name,
                    has_raw_files=any(d.iterdir()),
                    has_graph=False,
                )

    # Topics with built graphs
    if graphs_dir.exists():
        for d in sorted(graphs_dir.iterdir()):
            if not d.is_dir():
                continue
            graph_data_path = d / "graph_data.json"
            if not graph_data_path.exists():
                continue

            meta_path = d / "build_meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            if d.name not in topics:
                topics[d.name] = TopicStatus(
                    topic=d.name,
                    has_raw_files=False,
                    has_graph=True,
                )
            else:
                topics[d.name].has_graph = True

            topics[d.name].node_count = meta.get("node_count")
            topics[d.name].edge_count = meta.get("edge_count")
            topics[d.name].community_count = meta.get("community_count")

    # Overlay live build status
    result = []
    for name, status in topics.items():
        task = task_manager.get_status(name)
        if task:
            status.build_status = task.status
            status.build_progress = task.progress
            status.build_error = task.error
        elif status.has_graph:
            status.build_status = "complete"
        result.append(status)

    return result


def _get_query_engine(topic: str) -> GraphRAGQueryEngine:
    if topic not in _query_engines:
        graphs_dir = Path(settings.graphs_dir)
        topic_dir = graphs_dir / topic
        if not topic_dir.exists():
            raise HTTPException(status_code=404, detail=f"Graph not found for topic '{topic}'")
        try:
            store = GraphRAGStore.load(topic_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        _query_engines[topic] = GraphRAGQueryEngine(
            graph_store=store,
            llm=_make_llm(settings.query_model, settings),
            community_llm=_make_llm(settings.extraction_model, settings),
        )
    return _query_engines[topic]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    topics = _discover_topics()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "topics": topics},
    )


@app.get("/api/topics")
async def list_topics() -> list[TopicStatus]:
    return _discover_topics()


@app.post("/api/topics/{topic}/build", status_code=202)
async def build_topic(topic: str, body: Optional[BuildRequest] = None):
    raw_dir = Path(settings.raw_dir) / topic
    if not raw_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Raw directory not found: raw/{topic}/. Create it and add documents first.",
        )

    ontology = (body.ontology if body else None)
    try:
        await task_manager.start_build(topic, ontology, _query_engines)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"status": "building", "topic": topic}


@app.get("/api/topics/{topic}/status")
async def topic_status(topic: str) -> TopicStatus:
    topics = {t.topic: t for t in _discover_topics()}
    if topic not in topics:
        raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found")
    return topics[topic]


@app.get("/api/topics/{topic}/graph")
async def get_graph(topic: str):
    graph_path = Path(settings.graphs_dir) / topic / "graph_data.json"
    if not graph_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Graph not built for topic '{topic}'. POST /api/topics/{topic}/build first.",
        )
    return Response(
        content=graph_path.read_text(encoding="utf-8"),
        media_type="application/json",
    )


@app.post("/api/topics/{topic}/query")
async def query_topic(topic: str, body: QueryRequest) -> QueryResponse:
    task = task_manager.get_status(topic)
    if task and task.status == "building":
        raise HTTPException(status_code=409, detail="Graph is still being built. Please wait.")

    engine = _get_query_engine(topic)
    answer, communities_checked, relevant_communities = engine.custom_query(body.query)

    return QueryResponse(
        answer=answer,
        communities_checked=communities_checked,
        relevant_communities=relevant_communities,
    )
