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
    print(f"[2/2] Answer received ({len(last['sources'])} sources):")
    print("  " + last["content"][:300].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
