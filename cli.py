
import os, sys
from openai import OpenAI
from rag import index_if_empty, search
from tools import get_summary_by_title
from moderation import contains_profanity

def main():
    if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
        print("Set OPENAI_API_KEY in environment.", file=sys.stderr)
        sys.exit(1)

    index_if_empty()
    print("Smart Librarian CLI. Type your query (Ctrl+C to exit).")
    while True:
        try:
            q = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q.strip():
            continue
        if contains_profanity(q):
            print("Let's keep it polite. Try again.")
            continue
        hits = search(q, k=3)
        if not hits:
            print("No matches.")
            continue
        title = hits[0]["metadata"]["title"]
        print(f"Recommendation: {title}")
        print("Fetching detailed summary...")
        print(get_summary_by_title(title))
        print()

if __name__ == "__main__":
    main()
