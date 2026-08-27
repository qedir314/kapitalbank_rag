"""Headless end-to-end smoke test of the Streamlit chat UI.

Runs app.py through Streamlit's AppTest harness — the same script execution a
real browser session triggers — then submits one question through the chat
input. Catches import errors, widget crashes and broken streaming that an
HTTP health check cannot see. The generated answer is read back from
``st.session_state.messages``, where app.py records every turn.

Usage:
    python -m scripts.smoke_app                      # default question
    python -m scripts.smoke_app "Kapital Bankın depozitləri"
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

DEFAULT_QUESTION = "What is Birbank Miles?"


def main() -> int:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    at = AppTest.from_file(str(APP_PATH), default_timeout=600)

    at.run()  # first render: loads embedding model + index
    if at.exception:
        print("FAILED on initial render:")
        for exc in at.exception:
            print(f"  {type(exc).__name__}: {exc.value}")
        return 1

    n_chunks = at.sidebar.metric[0].value if at.sidebar.metric else "?"
    print(f"[1/2] Initial render OK ({n_chunks} indexed passages)")

    at.chat_input[0].set_value(question)
    at.run()  # full RAG turn: retrieve -> DeepSeek -> stream
    if at.exception:
        print(f"FAILED on question {question!r}:")
        for exc in at.exception:
            print(f"  {type(exc).__name__}: {exc.value}")
        return 1

    messages = at.session_state["messages"]
    answers = [m for m in messages if m["role"] == "assistant"]
    if not answers or not answers[-1]["content"].strip():
        print("FAILED: assistant produced no visible text.")
        return 1

    last = answers[-1]
    print(f"[2/3] Answer received ({len(last['sources'])} sources):")
    print("  " + last["content"][:300].replace("\n", "\n  "))

    # multi-turn: a bare follow-up exercises history + query condensing (4.4),
    # and the citation-verification field (4.2) must be populated post-stream
    at.chat_input[0].set_value("And how much does that card cost?")
    at.run()
    if at.exception:
        print("FAILED on follow-up turn:")
        for exc in at.exception:
            print(f"  {type(exc).__name__}: {exc.value}")
        return 1
    messages = at.session_state["messages"]
    turn2 = [m for m in messages if m["role"] == "assistant"][-1]
    condensed = turn2.get("retrieval_query")
    print(f"[3/3] Follow-up OK — condensed query: {condensed!r}")
    if condensed is None:
        print("  (no standalone rewrite — condensing off, or query unchanged)")
    if "citation_flags" not in turn2:
        print("FAILED: citation_flags missing (4.2 wiring broken).")
        return 1
    # freshness (4.3): at least one source should carry a crawl date now
    dated = [s for s in turn2["sources"] if getattr(s, "crawled_at", "")]
    print(f"  sources carrying crawled_at: {len(dated)}/{len(turn2['sources'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
