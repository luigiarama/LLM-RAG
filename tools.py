
from typing import Dict, Any
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "book_summaries.json"

def load_summaries() -> Dict[str, Dict[str, Any]]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)
    by_title = {it["title"]: it for it in items}
    return by_title

BOOKS = load_summaries()

def get_summary_by_title(title: str) -> str:
    """Returnează rezumatul complet pentru un titlu exact."""
    item = BOOKS.get(title)
    if not item:
        return "Nu am găsit acest titlu în baza locală."
    return item["summary_full"]
