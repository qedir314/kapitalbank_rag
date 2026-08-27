"""Kapital Bank RAG assistant — Streamlit chat UI.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from kb_rag.config import ROOT, get_settings
from kb_rag.feedback import record_feedback
from kb_rag.rag.pipeline import get_pipeline

st.set_page_config(page_title="Kapital Bank RAG Assistant", page_icon="🏦", layout="centered")

SECTIONS = ["cards", "loans", "deposits", "money-transfers", "sigortalar",
            "corporate-banking", "birbank", "faq", "how-to", "news", "other",
            "locations", "kampaniyalar", "ferdi-bankciliq", "online-order", "insurance"]

# KB_FEEDBACK_PATH lets the smoke test point 👎 capture at a temp file
FEEDBACK_PATH = Path(os.environ.get("KB_FEEDBACK_PATH") or ROOT / "data" / "feedback.jsonl")


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
            # plan 4.3: bank rates change — every passage states when it was
            # crawled; "" on chunks indexed before the freshness rollout
            fresh = f" · crawled `{src.crawled_at[:10]}`" if src.crawled_at else ""
            st.markdown(
                f"**[{src.index}]** {src.section_path or src.title}  \n"
                f"[{src.url}]({src.url}) · similarity `{score_str}`{fresh}"
            )


def _multilang_warning(flagged) -> str:
    nums = ", ".join(f"[{n}]" for n in flagged)
    return (
        f"⚠ Could not verify citation{'' if len(flagged) == 1 else 's'} {nums} "
        f"against the cited passage — double-check on the official site."
    )


def render_feedback_bar(msg_idx: int) -> None:
    """👍/👎 capture (plan 4.5) — appends to data/feedback.jsonl, never mutates
    the answer. One rating per message per session (stored on the message)."""
    messages = st.session_state.messages
    message = messages[msg_idx]
    if message.get("feedback") is not None:
        note = "👍 recorded — thank you!" if message["feedback"] > 0 \
            else "👎 recorded — we'll review this answer."
        st.caption(note)
        return
    left, right, _ = st.columns([1, 1, 4])
    rating = None
    if left.button("👍", key=f"fb_up_{msg_idx}", help="This answer helped"):
        rating = 1
    elif right.button("👎", key=f"fb_down_{msg_idx}", help="This answer was wrong or unhelpful"):
        rating = -1
    if rating is None:
        return
    # the question is the closest preceding user message
    question = next(
        (m["content"] for m in reversed(messages[:msg_idx]) if m["role"] == "user"),
        "",
    )
    meta = message.get("meta") or {}
    record_feedback(
        FEEDBACK_PATH,
        question=question,
        answer=message["content"],
        rating=rating,
        sources=message.get("sources") or [],
        lang=meta.get("lang"),
        sections=meta.get("sections"),
        top_k=meta.get("top_k"),
    )
    message["feedback"] = rating  # dicts are shared with session_state — flag persists


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
    as_of = pipeline.content_as_of()
    if as_of:
        # plan 4.3: rates move; dated content is honest content
        st.metric("Content as of", as_of[:10])
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

def render_answer_extras(msg_idx: int) -> None:
    """Everything an assistant answer carries beyond its text (plans 4.2–4.5)."""
    message = st.session_state.messages[msg_idx]
    if message.get("retrieval_query"):
        # plan 4.4: transparency — show what a follow-up was retrieved as
        st.caption(f"🔎 Retrieved as: *{message['retrieval_query']}*")
    if message.get("citation_flags"):
        # plan 4.2: runtime citation verification — honest warnings, not silence
        st.warning(_multilang_warning(message["citation_flags"]))
    render_sources(message.get("sources"))
    render_feedback_bar(msg_idx)


for idx, message in enumerate(st.session_state.messages):
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_answer_extras(idx)

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
        if answer.text is not None:
            # pipeline-level message (empty index / no passages under this
            # filter) — arrives whole, not streamed
            full_text = answer.text
            st.markdown(full_text)
        else:
            # write_stream consumes the generator to completion, which is when
            # the citation-verification wrapper attaches its report (4.2)
            full_text = st.write_stream(answer.text_stream or iter([]))
        flagged = tuple(answer.citations.flagged) if answer.citations else ()
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "sources": list(answer.sources),
            "citation_flags": flagged,
            "retrieval_query": answer.retrieval_query,
            "meta": {"lang": lang_arg, "sections": list(selected_sections or []),
                     "top_k": top_k},
        })
        render_answer_extras(len(st.session_state.messages) - 1)
