"""Streamlit chat app for the Spectroscopy Use-Case Assessment Assistant.

Run with:  streamlit run streamlit_app.py

Features
  * Chat interface for describing a customer use case.
  * Upload PDF / Word / Excel documents to grow the knowledge base live
    (this is the "scalable" story: more documents -> smarter assistant).
  * Every answer shows the sources it was built from.
  * Refuses / asks follow-ups instead of hallucinating.
"""

from __future__ import annotations

import streamlit as st

from app import config
from app.assistant import SpectroscopyAssistant
from app.loaders import SUPPORTED_EXTENSIONS, extract_text
from app.vector_store import VectorStore

st.set_page_config(page_title="Spectroscopy Assistant", page_icon="🔬", layout="centered")


# --- Cached singletons ----------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner="Connecting to Gemini...")
def get_assistant(_store: VectorStore) -> SpectroscopyAssistant:
    return SpectroscopyAssistant(_store)


store = get_store()


# --- Sidebar: knowledge base + document upload ----------------------------
with st.sidebar:
    st.header("📚 Knowledge base")

    uploaded = st.file_uploader(
        "Upload documents (PDF, Word, Excel, text)",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Add to knowledge base", type="primary"):
        for f in uploaded:
            text = extract_text(f.name, f.getvalue())
            if not text.strip():
                st.warning(f"Could not extract text from {f.name}.")
                continue
            n = store.add_document(f.name, text)
            st.success(f"Added {f.name} ({n} chunks).")
        st.cache_resource.clear()  # refresh counts
        st.rerun()

    st.divider()
    st.caption(f"Chunks indexed: **{store.count()}**")
    sources = store.list_sources()
    if sources:
        st.caption("Documents:")
        for s in sources:
            st.caption(f"• {s}")
    else:
        st.info("No documents yet. Upload some to get started.")

    st.divider()
    if not config.GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not set. Add it to a .env file.")


# --- Main chat ------------------------------------------------------------
st.title("🔬 Spectroscopy Use-Case Assistant")
st.caption(
    "A first-pass technical assessment helper for AFS experts. "
    "Describe a customer's monitoring need; answers are grounded in the "
    "uploaded knowledge base and always cite their sources."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources used"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**{i}. {src['source']}**")
                    st.caption(src["text"][:600] + ("…" if len(src["text"]) > 600 else ""))

prompt = st.chat_input("Describe the customer's use case or ask a question…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not config.GEMINI_API_KEY:
            st.error("Cannot answer: GEMINI_API_KEY is not set in your .env file.")
        else:
            try:
                assistant = get_assistant(store)
                with st.spinner("Assessing…"):
                    reply = assistant.ask(prompt, history=st.session_state.messages)
                st.markdown(reply.answer)

                src_dicts = [
                    {"source": c.source, "text": c.text, "distance": c.distance}
                    for c in reply.sources
                ]
                if src_dicts:
                    with st.expander("📎 Sources used"):
                        for i, src in enumerate(src_dicts, start=1):
                            st.markdown(f"**{i}. {src['source']}**")
                            st.caption(
                                src["text"][:600]
                                + ("…" if len(src["text"]) > 600 else "")
                            )

                st.session_state.messages.append(
                    {"role": "assistant", "content": reply.answer, "sources": src_dicts}
                )
            except Exception as e:  # noqa: BLE001 - surface errors in the UI
                st.error(f"Something went wrong: {e}")
