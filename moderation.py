
import re

# Minimal list; extend as needed.
BANNED = [
    r"\bidiot\b",
    r"\bstupid\b",
    r"\btrash\b",
    r"\bcrap\b",
    r"\bjerk\b",
    r"\bshut up\b",
    r"\bfool\b",
    
]

def contains_profanity(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pat, lowered) for pat in BANNED)
