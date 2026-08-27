"""Offline tests for runtime citation verification (plan 4.2).

Uses the same numbered-context format ``build_context_block`` produces. The
checker must agree with the eval-side ``citation_stats`` on the same inputs —
these tests pin that shared semantics.
"""
from kb_rag.rag.citations import verify_citations

CONTEXT = (
    "[1] Cards > Debet kartları (https://birbank.az/debet-kartlar)\n"
    "Birbank debet kartının qiyməti pulsuzdur və kart dərhal təsdiqlənir.\n\n"
    "[2] Loans > İstehlak krediti (https://birbank.az/credits)\n"
    "İstehlak kreditinin illik faiz dərəcəsi 18% təşkil edir, müddət 36 ay."
)


def test_supported_valid_citation_is_ok():
    answer = "Birbank debet kartı pulsuzdur [1]. Kart dərhal təsdiqlənir [1]."
    report = verify_citations(answer, CONTEXT)
    assert report.n_markers == 2
    assert report.ok, report.flagged


def test_marker_pointing_at_nonexistent_passage_is_invalid():
    report = verify_citations("Burada əla bir şey yazılıb [7].", CONTEXT)
    assert 7 in report.invalid
    assert 7 in report.flagged


def test_unrelated_cited_passage_is_flagged_unsupported():
    # sentence about loan rates citing the debit-card passage: no overlap
    report = verify_citations("İstehlak krediti faiz dərəcəsi 18% [1].", CONTEXT)
    assert report.invalid == ()          # the marker itself is real
    assert 1 in report.unsupported       # but passage 1 doesn't support it
    assert report.flagged == (1,)


def test_correctly_attributed_fact_is_not_flagged():
    report = verify_citations("İstehlak kreditinin illik faiz dərəcəsi 18% [2].", CONTEXT)
    assert report.ok


def test_short_citing_sentence_is_not_judged():
    # "Bəli [1]." has too few content tokens to test support — no verdict,
    # and crucially it must NOT be flagged as unsupported
    report = verify_citations("Bəli [1].", CONTEXT)
    assert report.n_markers == 1
    assert report.ok


def test_answer_without_markers_has_nothing_to_verify():
    # refusals legitimately carry no citations
    report = verify_citations("Məlumat tapılmadı. kapitalbank.az-a müraciət edin.", CONTEXT)
    assert report.n_markers == 0
    assert report.ok


def test_multi_marker_sentence_needs_only_one_supporting_passage():
    # max-overlap semantics: one genuinely supporting passage [1] is enough —
    # the co-cited unrelated passage [2] must not poison the whole sentence
    report = verify_citations("Kart pulsuzdur, təsdiq dərhal olur [1][2].", CONTEXT)
    assert report.ok
    assert report.n_markers == 2


def test_mixed_flags_union_sorted():
    answer = (
        "Birbank debet kartının qiyməti pulsuzdur [1]. "
        "Kapital Bankın filial sayı 500-dür [9]. "
        "Kredit faizi 99% təşkil edir [2]. Faiz dərəcəsi yüksəkdir [1]."
    )
    report = verify_citations(answer, CONTEXT)
    assert 9 in report.invalid
    # [2] here claims 99% against a 18% passage, but the overlap is lexical not
    # numeric; we only assert the flagged set is a sorted superset of invalid
    assert list(report.flagged) == sorted(set(report.flagged))
    assert set(report.invalid).issubset(set(report.flagged))
