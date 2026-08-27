"""Offline tests for feedback capture + review aggregation (plan 4.5)."""
import json

from kb_rag.feedback import record_feedback
from kb_rag.rag.pipeline import Source
from scripts.review_feedback import _norm, aggregate, candidates, load_feedback


def _src(i=1, crawled="2026-08-26T09:22:33+00:00"):
    return Source(index=i, title=f"p{i}", url=f"https://birbank.az/page-{i}",
                  source_url="", section_path="cards", score=0.7,
                  crawled_at=crawled)


# ---------------------------------------------------------------- record_feedback
def test_record_appends_one_json_line_per_rating(tmp_path):
    path = tmp_path / "feedback.jsonl"
    assert record_feedback(path, question="kart qiymeti?", answer="Pulsuz [1].",
                           rating=1, sources=[_src(1), _src(2)])
    assert record_feedback(path, question="kart qiymeti?", answer="Pulsuz [1].",
                           rating=-1, sources=[])
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert [l["rating"] for l in lines] == [1, -1]
    assert lines[0]["sources"] == ["https://birbank.az/page-1", "https://birbank.az/page-2"]
    assert lines[0]["crawled_at"] == "2026-08-26T09:22:33+00:00"
    assert lines[1]["sources"] == []
    assert all(l["ts"] for l in lines)


def test_record_never_raises_on_io_error(tmp_path):
    # a directory can't be opened for append -> swallowed, returns False
    bad = tmp_path  # the path IS an existing directory
    assert record_feedback(bad, question="q", answer="a", rating=1, sources=[]) is False


def test_answer_is_truncated(tmp_path):
    path = tmp_path / "f.jsonl"
    record_feedback(path, question="q", answer="x" * 5000, rating=-1, sources=[])
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert len(rec["answer"]) == 800


# ---------------------------------------------------------------- review script
def _rec(question, rating, sources=None, lang="az"):
    return {"question": question, "rating": rating, "answer": "a",
            "sources": sources or [], "lang": lang}


def test_load_feedback_tolerates_torn_lines(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text('{"question":"q","rating":1}\n{torn\n\n', encoding="utf-8")
    assert len(load_feedback(path)) == 1
    assert load_feedback(tmp_path / "missing.jsonl") == []


def test_aggregate_groups_cosmetic_variants():
    recs = [
        _rec("Kart  qiyməti?", -1),        # double space
        _rec("kart  qiyməti?", -1),        # different case
        _rec("kart qiyməti?", 1),
    ]
    agg = aggregate(recs)
    assert len(agg) == 1                    # all three normalize together
    slot = next(iter(agg.values()))
    assert (slot["down"], slot["up"]) == (2, 1)


def test_candidates_require_majority_down_and_min_count():
    recs = [
        _rec("sual A", -1), _rec("sual A", -1),          # 2 down / 0 up -> in
        _rec("sual B", -1), _rec("sual B", 1), _rec("sual B", 1),  # out: up wins
        _rec("sual C", -1),                              # out: below min-down
    ]
    cands = candidates(aggregate(recs), min_down=2)
    assert [c["display"] for c in cands] == ["sual A"]


def test_candidate_order_worst_first():
    recs = ([_rec("az aşağı", -1)] * 2) + ([_rec("ən aşağı", -1)] * 3)
    cands = candidates(aggregate(recs), min_down=2)
    assert [c["down"] for c in cands] == [3, 2]
