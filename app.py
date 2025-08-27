import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from typing import Any, Dict, List

from rag import index_if_empty, search, generate_response_with_rag
from tools import get_summary_by_title
from moderation import contains_profanity
from chatbot import get_chat_response

st.set_page_config(page_title="Smart Librarian", page_icon="📚")
st.title("Smart Librarian — RAG + Tool")


st.caption("Demo: recomandă cărți după teme și oferă un rezumat detaliat printr-un tool.")

# Load .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY nu este setat în .env. Adăugați-l și reporniți aplicația.")
    st.stop()
client = OpenAI()

with st.spinner("Inițializez vector store..."):
    index_if_empty()

query = st.text_input("Ce fel de carte cauți? (ex: 'prietenie și magie', 'povești de război')")
if st.button("Caută recomandare") and query:
    if contains_profanity(query):
        st.warning("Hai să păstrăm conversația politicoasă. Încearcă să reformulezi 🙂")
        st.stop()

    # Get RAG-based response
    st.subheader("Răspuns RAG:")
    try:
        rag_response = generate_response_with_rag(query)
        st.write(rag_response)
    except Exception as e:
        st.error(f"Eroare la generarea răspunsului RAG: {e}")

    # Search for recommendations
    hits = search(query, k=3)
    if not hits:
        st.error("Nu am găsit nimic relevant.")
        st.stop()

    # Pick the top hit and extract title from metadata
    top = hits[0]["metadata"]
    title = top["title"]
    st.subheader(f"Recomandare: **{title}**")
    st.write("Pe baza temelor potrivite, aceasta pare o alegere bună.")

    # Tool call (simulated): fetch full summary
    full = get_summary_by_title(title)
    with st.expander("Rezumat detaliat"):
        st.write(full)

    # (Optional) Text-to-Speech placeholder
    if st.checkbox("Generează audio (TTS) [placeholder]"):
        st.info("Integrați TTS preferat (ex: OpenAI TTS, edge-tts).")
