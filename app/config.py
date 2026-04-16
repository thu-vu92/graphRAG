from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM Provider ────────────────────────────────────────────────────────────
    # Supports OpenAI and any OpenAI-compatible API (LM Studio, Ollama, etc.)
    openai_api_key: str = ""

    # Set this to use a local/custom endpoint, e.g.:
    #   LLM_BASE_URL=http://localhost:1234/v1   (LM Studio)
    #   LLM_BASE_URL=http://localhost:11434/v1  (Ollama with OpenAI compat)
    llm_base_url: Optional[str] = None

    extraction_model: str = "gpt-4o-mini"
    query_model: str = "gpt-4o"

    # ── Data directories ────────────────────────────────────────────────────────
    raw_dir: Path = Path("./raw")
    graphs_dir: Path = Path("./graphs")

    # ── Pipeline parameters ─────────────────────────────────────────────────────
    max_paths_per_chunk: int = 20
    num_workers: int = 4
    max_cluster_size: int = 10
    max_documents: int = 50


settings = Settings()
