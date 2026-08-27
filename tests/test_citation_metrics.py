"""Unit tests for the deterministic citation checker (plan 3.5) — pure strings, offline."""
from kb_rag.evaluation.citation_metrics import citation_stats, split_passages

CONTEXT = (
    "[1] Cards > BirKart (https://x/cards)\n"
    "BirKart Miles debit kartı 10 mil və təqaüd kartıdır.\n\n"
    "[2] Loans > Nağd kredit (https://x/loans)\n"
    "Nağd pul krediti 10.9% dərəcə ilə 24 ayadək verilir."
)


# --------------------------------------------------------------------------- passage parsing
def test_split_passages_keys_and_bodies():
    passages = split_passages(CONTEXT)
    assert set(passages) == {1, 2}
    assert "10.9%" in passages[2]
    assert passages[1].startswith("BirKart Miles")


def test_split_passages_empty_context():
    assert split_passages("") == {}
    assert split_passages(None) == {}


# --------------------------------------------------------------------------- stats
def test_citation_stats_no_markers_returns_nones():
    stats = citation_stats("Heç bir məlumat yoxdur.", CONTEXT)
    assert stats["n_citations"] == 0
    assert stats["citation_valid_frac"] is None
    assert stats["citation_coverage"] is None


def test_out_of_range_markers_drop_validity():
    answer = "BirKart 10 mil verir [1]. Kredit 24 aydır [7]."
    stats = citation_stats(answer, CONTEXT)
    assert stats["n_citations"] == 2
    assert stats["citation_valid_frac"] == 0.5  # [7] references nothing


def test_supported_sentence_scores_full_overlap_share():
    answer = "Nağd pul krediti 10.9% dərəcə ilə verilir [2]."
    stats = citation_stats(answer, CONTEXT)
    assert stats["citation_valid_frac"] == 1.0
    assert stats["citation_support_frac"] == 1.0  # most sentence tokens live in passage 2


def test_unsupported_citation_lowers_support_frac():
    # cites [1] but talks about a completely different subject — the staple
    # of a fabricated citation
    answer = "Bank our metro station opened in 1997 [1]."
    stats = citation_stats(answer, CONTEXT)
    assert stats["citation_valid_frac"] == 1.0  # [1] exists…
    assert stats["citation_support_frac"] == 0.0  # …but supports nothing here


def test_coverage_counts_citing_sentences():
    answer = "BirKart Miles debet kartıdır [1]. Təqaüd kartı da var [1]. Marja haqqında məlumat yoxdur."
    stats = citation_stats(answer, CONTEXT)
    assert stats["citation_coverage"] == 2 / 3


def test_multiple_markers_in_one_sentence():
    answer = "Kart 10 mil verir və kredit 10.9% dərəcəlidir [1][2]."
    stats = citation_stats(answer, CONTEXT)
    assert stats["n_citations"] == 2
    assert stats["citation_valid_frac"] == 1.0
    # max overlap over the two referenced passages decides support
    assert stats["citation_support_frac"] == 1.0


def test_short_citation_only_sentence_excluded_from_support():
    answer = "Bəli [1]. Nağd pul krediti 10.9% dərəcə ilə 24 ayadək verilir [2]."
    stats = citation_stats(answer, CONTEXT)
    assert stats["n_citations"] == 2
    assert stats["citation_support_frac"] == 1.0  # judged only the long sentence
