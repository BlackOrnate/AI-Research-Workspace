"""Local paper resolver: answers only "Do I already have this paper?", from the paper metadata.

Version 1 matches titles and aliases from data/papers.json in code. It does not use the model,
and the paper list never goes into a prompt. Later versions can add semantic search over the
metadata behind the same resolve_local_paper() interface.
"""
import json
import re

from config import PAPERS_METADATA_PATH


def load_paper_metadata() -> list[dict]:
    return json.loads(PAPERS_METADATA_PATH.read_text(encoding="utf-8"))


paper_metadata = load_paper_metadata()

# Characters that can be part of a paper name. A match must not touch one on either side,
# so "SAM" is not found in "MedSAM" or "Cellpose-SAM", and "CellViT" is not found in "CellViT++"
NAME_CHAR = r"[a-z0-9+\-]"


def normalize(text: str) -> str:
    return text.lower().strip()


def mentions(query: str, name: str) -> bool:
    """Whether the query contains the name as a whole, ignoring case."""
    pattern = rf"(?<!{NAME_CHAR}){re.escape(normalize(name))}(?!{NAME_CHAR})"
    return re.search(pattern, normalize(query)) is not None


def resolve_local_paper(query: str) -> dict:
    """Find the local papers named in the query by their title or an alias.

    Returns {"found": True, "papers": [{"paper_id", "title"}, ...]} or {"found": False, "papers": []}.
    It returns every paper it finds, because questions like "Compare CellViT and HoVer-Net" name several.
    """
    papers = [
        {"paper_id": paper["paper_id"], "title": paper["title"]}
        for paper in paper_metadata
        if any(mentions(query, name) for name in [paper["full_title"], *paper["aliases"]])
    ]
    return {"found": bool(papers), "papers": papers}
