import os
import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "book_summaries.json"

def load_summaries() -> dict:
    """Load book summaries from the JSON file."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return {item["title"].lower(): item for item in json.load(f)}

BOOK_SUMMARIES = load_summaries()

def normalize_text(text: str) -> str:
    """Normalize text by removing punctuation and converting to lowercase."""
    return re.sub(r"\W+", " ", text).strip().lower()

def extract_keywords(text: str) -> set:
    """Extract keywords from the text by removing stopwords and normalizing."""
    stopwords = {"vreau", "o", "carte", "despre", "si", "in", "pe", "cu", "la", "de", "un", "o", "este", "sunt"}
    words = re.findall(r"\b\w{3,}\b", text.lower())
    return set(words) - stopwords

def get_chat_response(query: str) -> str:
    """Generate a chatbot response based on available book summaries."""
    query_keywords = extract_keywords(query)
    for title, summary in BOOK_SUMMARIES.items():
        title_keywords = extract_keywords(title)
        short_summary_keywords = extract_keywords(summary['summary_short'])
        full_summary_keywords = extract_keywords(summary['summary_full'])
        themes_keywords = set(summary.get('themes', []))

        if query_keywords & (title_keywords | short_summary_keywords | full_summary_keywords | themes_keywords):
            return f"I found a match: {summary['title']}\n\nSummary: {summary['summary_full']}\n\nRecommendation: This book matches your query and is highly recommended!"

    return "I'm sorry, I couldn't find any books matching your query in the available summaries."
