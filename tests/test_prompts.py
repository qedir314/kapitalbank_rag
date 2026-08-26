from kb_rag.ingest.store import RetrievedChunk
from kb_rag.rag.prompts import build_context_block, build_messages, build_system_prompt


def _chunk(i: int) -> RetrievedChunk:
    return RetrievedChunk(
        text=f"Passage text {i}",
        url=f"https://birbank.az/en/page-{i}",
        source_url=f"https://kapitalbank.az/en/page-{i}",
        title=f"Page {i}",
        lang="en",
        section="cards",
        section_path=f"Cards > Page {i}",
        score=0.9 - i * 0.01,
    )


def test_context_block_numbers_passages_and_links():
    block = build_context_block([_chunk(1), _chunk(2)])
    assert "[1] Cards > Page 1 (https://birbank.az/en/page-1)" in block
    assert "[2]" in block
    assert "Passage text 2" in block


def test_system_prompt_wraps_context_and_states_rules():
    prompt = build_system_prompt(build_context_block([_chunk(1)]))
    assert "<context>" in prompt and "</context>" in prompt
    lowered = prompt.lower()
    assert "only on the numbered context" in lowered   # grounding rule
    assert "cite" in lowered                            # citation rule


def test_build_messages_ordering():
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    messages = build_messages("SYSTEM", history, "What cards exist?")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "SYSTEM"
    assert [m["content"] for m in messages[1:]] == ["hi", "hello", "What cards exist?"]
    assert messages[-1]["role"] == "user"
