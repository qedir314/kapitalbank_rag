"""LLM-as-judge scoring of generated answers + refusal detection.

Faithfulness and correctness are scored 1-5 against a rubric by a judge model
(configurable separately from the answering model). Refusals on unanswerable
questions are detected with a multilingual phrase heuristic first, then judged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of Retrieval-Augmented Generation (RAG) answers \
about bank products. You will receive: a user question, the retrieved context passages the answer \
was based on, and the generated answer.

Score BOTH dimensions from 1 to 5:

faithfulness — how well the answer sticks to the context:
5 = every claim is directly supported by the context; no outside facts
3 = mostly supported but includes minor unsupported details or rounding
1 = contradicts the context or invents rates/amounts/policies

correctness — how well the answer addresses the question using the context:
5 = correct and complete w.r.t. the context, right language, proper citations
3 = partially correct or incomplete
1 = wrong, ignores the context, or answers something else entirely

Respond with ONLY a JSON object, no other text:
{"faithfulness": <int>, "correctness": <int>, "rationale": "<one short sentence>"}"""


# Multilingual refusal markers (en / az / ru). Includes both the
# "information missing" template family and the off-domain-gate template
# ("I can only help with bank topics…").
REFUSAL_MARKERS = (
    # english
    "don't have", "do not have", "couldn't find", "could not find",
    "cannot answer", "can't answer", "cannot provide", "not able to answer",
    "cannot find", "can't find", "cannot locate",
    "not enough information", "no information", "unable to answer",
    "not mentioned in", "not included in", "does not contain", "don't know",
    "i don't know", "isn't mentioned", "not specified",
    "only help with kapital bank", "only answer questions about kapital bank",
    "general-purpose chatbot", "general knowledge",
    # azerbaijani
    "məlumat yoxdur", "məlumat tapılmadı", "məlumatım yoxdur",
    "cavab vermək mümkün deyil", "dəqiq məlumat yoxdur", "tapmaq mümkün olmadı",
    "heç bir məlumat", "cavab verə bilmirəm", "cavab verə bilmərəm",
    "verə bilmirəm", "verə bilmərəm",   # …məlumat verə bilmirəm / cavabsız formalar
    "kömək edə bilmərəm", "yalnız kapital bank", "kontekstdə yoxdur",
    "tapmadım",
    # russian
    "не располагаю", "нет информации", "информация не найдена",
    "не могу ответить", "не могу найти", "не удалось найти",
    "не найдено", "не содержится", "нет данных",
    "невозможно получить информацию", "невозможно ответить",
    "только вопросы о капитал банк", "можу помочь только",
    "отсутствует в предоставленной информации", "відповісти",
)


@dataclass(frozen=True)
class JudgeResult:
    faithfulness: int
    correctness: int
    rationale: str
    raw: str


def looks_like_refusal(answer_text: str | None) -> bool:
    if not answer_text:
        return True
    lowered = answer_text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def parse_judge_json(text: str) -> dict:
    """Parse the judge's JSON, tolerating markdown fences / stray prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Judge returned non-JSON output: {text[:200]!r}")


def _clamp(score) -> int:
    return max(1, min(5, int(round(float(score)))))


def judge_answer(
    llm_client,
    judge_model: str,
    question: str,
    context: str,
    answer: str,
    reference: str | None = None,
) -> JudgeResult:
    """Ask the judge model to score faithfulness/correctness of ``answer``."""
    parts = [f"QUESTION:\n{question}", f"CONTEXT PASSAGES:\n{context}",
             f"GENERATED ANSWER:\n{answer}"]
    if reference:
        parts.append(f"REFERENCE ANSWER (additional ground truth):\n{reference}")

    raw = llm_client.complete(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n---\n\n".join(parts)},
        ],
        stream=False,
        model=judge_model,
        temperature=0.0,
        max_tokens=300,
    )
    data = parse_judge_json(raw)
    return JudgeResult(
        faithfulness=_clamp(data.get("faithfulness", 1)),
        correctness=_clamp(data.get("correctness", 1)),
        rationale=str(data.get("rationale", ""))[:300],
        raw=raw,
    )
