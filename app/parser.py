"""
Document parser — reads files from raw/<topic>/ and extracts plain text.

Supported formats: .pdf, .docx, .html/.htm, .txt, .md, .csv
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md", ".csv"}


class DocumentParser:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def parse_topic(self, topic: str) -> list[dict]:
        """
        Scan raw/<topic>/, parse each supported file, return a list of:
          [{"filename": str, "text": str, "metadata": dict}, ...]
        Unsupported extensions are skipped with a warning.
        """
        topic_dir = self.raw_dir / topic
        if not topic_dir.exists():
            raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

        documents = []
        for path in sorted(topic_dir.iterdir()):
            if not path.is_file():
                continue
            ext = path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported file: %s", path.name)
                continue

            try:
                text, extra_meta = self._dispatch(path, ext)
            except Exception as exc:
                logger.error("Failed to parse %s: %s", path.name, exc)
                continue

            if not text or not text.strip():
                logger.warning("Empty content after parsing: %s", path.name)
                continue

            documents.append(
                {
                    "filename": path.name,
                    "text": text.strip(),
                    "metadata": {"source": str(path), "filename": path.name, **extra_meta},
                }
            )

        logger.info("Parsed %d documents from topic '%s'", len(documents), topic)
        return documents

    # ── Dispatcher ─────────────────────────────────────────────────────────────

    def _dispatch(self, path: Path, ext: str) -> tuple[str, dict]:
        """Return (text, extra_metadata) for the given file."""
        if ext == ".pdf":
            return self._parse_pdf(path), {}
        if ext == ".docx":
            return self._parse_docx(path), {}
        if ext in (".html", ".htm"):
            return self._parse_html(path), {}
        if ext == ".txt":
            return self._parse_txt(path), {}
        if ext == ".md":
            return self._parse_txt(path), {}  # markdown is plain text
        if ext == ".csv":
            return self._parse_csv(path)
        raise ValueError(f"No parser for extension: {ext}")

    # ── Format-specific parsers ────────────────────────────────────────────────

    def _parse_pdf(self, path: Path) -> str:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages)

    def _parse_docx(self, path: Path) -> str:
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _parse_html(self, path: Path) -> str:
        import trafilatura

        raw_html = path.read_bytes()
        text = trafilatura.extract(raw_html, include_comments=False, include_tables=True)
        if not text:
            # Fallback to BeautifulSoup
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        return text or ""

    def _parse_txt(self, path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_bytes().decode("utf-8", errors="replace")

    def _parse_csv(self, path: Path) -> tuple[str, dict]:
        """
        CSV handling:
        - If a 'full_text' or 'text' column exists, treat each row as a separate document
          and concatenate them (backward-compatible with the original ai_copyright_dataset.csv).
        - Otherwise, stringify the entire CSV as a table.
        """
        import pandas as pd

        df = pd.read_csv(str(path))
        text_col = next((c for c in df.columns if c.lower() in ("full_text", "text", "content")), None)

        if text_col:
            rows = df[text_col].dropna().astype(str).tolist()
            text = "\n\n---\n\n".join(rows)
        else:
            text = df.to_string(index=False)

        return text, {"row_count": len(df)}
