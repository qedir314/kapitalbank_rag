import pytest

from kb_rag.evaluation.generation_metrics import (
    JudgeResult,
    judge_answer,
    looks_like_refusal,
    parse_judge_json,
)


def test_refusal_markers_multilingual():
    assert looks_like_refusal("Sorry, I don't have that information.")
    assert looks_like_refusal("Təəssüf ki, bu barədə məlumat yoxdur.")
    assert looks_like_refusal("К сожалению, нет информации по этому вопросу.")
    assert not looks_like_refusal("Kapital Bank offers 6 premium cards [1].")
    assert looks_like_refusal(None)


def test_offdomain_gate_paraphrases_detected():
    """The Phase 4 A/B caught real off-domain refusals scoring as answered
    because the template was paraphrased differently — the marker set must
    cover the whole "I only help with bank topics" family, not one phrasing."""
    assert looks_like_refusal(
        "I can only help with questions about Kapital Bank / Birbank products. "
        "Solving math equations is outside my scope.")
    assert looks_like_refusal(
        "Я могу помочь только с вопросами о продуктах Kapital Bank. Написание "
        "кода на Python не относится к этой теме, не могу выполнить этот запрос.")


def test_parse_judge_json_variants():
    assert parse_judge_json('{"faithfulness": 5, "correctness": 4, "rationale": "ok"}')["correctness"] == 4
    fenced = '```json\n{"faithfulness": 3, "correctness": 3, "rationale": "meh"}\n```'
    assert parse_judge_json(fenced)["faithfulness"] == 3
    noisy = 'Sure! {"faithfulness": 2, "correctness": 2, "rationale": "weak"} hope that helps'
    assert parse_judge_json(noisy)["correctness"] == 2
    with pytest.raises(ValueError):
        parse_judge_json("no json here at all")


class FakeLLM:
    """Stands in for DeepSeekClient in tests — returns a canned completion."""

    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, messages, stream=False, model=None, **kwargs):
        self.calls.append({"messages": messages, "model": model})
        return self.payload


def test_judge_answer_clamps_out_of_range_scores():
    fake = FakeLLM('{"faithfulness": 9, "correctness": 0, "rationale": "wild"}')
    result = judge_answer(
        fake, judge_model="judge-x", question="Q?", context="CTX", answer="A"
    )
    assert isinstance(result, JudgeResult)
    assert result.faithfulness == 5      # clamped to rubric max
    assert result.correctness == 1       # clamped to rubric min
    assert fake.calls[0]["model"] == "judge-x"
    # the judge prompt must carry question, context and answer
    user_msg = fake.calls[0]["messages"][1]["content"]
    for fragment in ("Q?", "CTX", "A"):
        assert fragment in user_msg


def test_judge_answer_includes_reference_when_given():
    fake = FakeLLM('{"faithfulness": 4, "correctness": 4, "rationale": "fine"}')
    judge_answer(fake, judge_model="j", question="Q", context="C", answer="A",
                 reference="REF-ANSWER")
    assert "REF-ANSWER" in fake.calls[0]["messages"][1]["content"]


class FlakyJudgeLLM:
    """Returns garbage completions first, then valid judge JSON."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def complete(self, messages, stream=False, model=None, **kwargs):
        self.calls += 1
        return self.sequence.pop(0)


GOOD_JUDGE = '{"faithfulness": 4, "correctness": 5, "rationale": "fine"}'


def test_judge_answer_retries_on_unparseable_json():
    client = FlakyJudgeLLM(["not json at all", "also { broken", GOOD_JUDGE])
    result = judge_answer(
        client, judge_model="m", question="q", context="ctx", answer="a", retries=3,
    )
    assert result.correctness == 5
    assert client.calls == 3  # two parse failures, then success


def test_judge_answer_raises_after_exhausting_retries():
    client = FlakyJudgeLLM(["garbage"] * 3)
    with pytest.raises(ValueError):
        judge_answer(
            client, judge_model="m", question="q", context="ctx", answer="a", retries=3,
        )
    assert client.calls == 3


def test_judge_answer_no_retry_on_success():
    client = FlakyJudgeLLM([GOOD_JUDGE])
    judge_answer(client, judge_model="m", question="q", context="ctx", answer="a")
    assert client.calls == 1
