"""Kapital Bank RAG assistant — Streamlit chat UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from kb_rag.config import get_settings
from kb_rag.rag.pipeline import get_pipeline

st.set_page_config(page_title="Kapital Bank RAG Assistant", page_icon="🏦", layout="centered")

SECTIONS = ["cards", "loans", "deposits", "money-transfers", "sigortalar",
            "corporate-banking", "birbank", "faq", "how-to", "news", "other",
            "locations", "kampaniyalar", "ferdi-bankciliq", "online-order", "insurance"]


@st.cache_resource(show_spinner="Loading embedding model and index…")
def load_pipeline():
    return get_pipeline()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role","content","sources":[...]}]


def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for src in sources:
            # reranker scores are probabilities in [0, 1]; show two decimals for
            # confident matches and a wider format for near-zero so the user
            # can tell "weak match" apart from "display bug"
            if src.score >= 0.01:
                score_str = f"{src.score:.2f}"
            else:
                score_str = f"{src.score:.4f}"
            st.markdown(
                f"**[{src.index}]** {src.section_path or src.title}  \n"
                f"[{src.url}]({src.url}) · similarity `{score_str}`"
            )


pipeline = load_pipeline()
settings = get_settings()
init_state()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("⚙️ Retrieval settings")
    lang = st.selectbox("Knowledge base language", ["auto (all)", "az", "en", "ru"])
    selected_sections = st.multiselect("Sections", SECTIONS, default=None)
    top_k = st.slider("Passages retrieved (top-k)", 2, 12, settings.retrieval.top_k)

    st.divider()
    n_chunks = pipeline.indexed_chunks
    st.metric("Indexed passages", f"{n_chunks:,}")
    if n_chunks == 0:
        st.warning("Index is empty — run `python -m scripts.scrape` then "
                   "`python -m scripts.build_index`.")
    st.caption(
        "Educational RAG project over public [kapitalbank.az](https://kapitalbank.az) pages. "
        "Answers cite the passages they use; always verify on the official site."
    )

# --------------------------------------------------------------------- chat
st.title("🏦 Kapital Bank RAG Assistant")
st.caption("Ask about cards, loans, deposits, transfers… — answers are grounded in scraped site content with inline citations.")

for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources"))

if question := st.chat_input("Your question…"):
    lang_arg = None if lang.startswith("auto") else lang

    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user", avatar="🧑").markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        answer = pipeline.answer(
            question,
            history=history,
            lang=lang_arg,
            section=selected_sections or None,
            top_k=top_k,
            stream=True,
        )
        full_text = st.write_stream(answer.text_stream or iter([]))
        render_sources(answer.sources)

    st.session_state.messages.append({
        "role": "assistant", "content": full_text, "sources": list(answer.sources),
    })
