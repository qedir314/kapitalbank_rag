from kb_rag.config import ChunkingConfig
from kb_rag.ingest.chunking import (
    build_chunks,
    derive_section,
    sliding_window,
    split_sections,
)

CFG = ChunkingConfig(target_chars=200, overlap_chars=40, min_chars=50)


PAGE = {
    "url": "https://kapitalbank.az/cards/taksitCards/birkart",
    "final_url": "https://birbank.az/az/kartlar/taksitkartlar",
    "lang": "az",
    "title": "Birbank Taksit Kartı",
    "text": (
        "Intro paragraph before any heading with enough content to matter.\n\n"
        "# Faizlər\n"
        + ("Kartın faiz dərəcəsi 20%-dir. " * 12)
        + "\n\n## Komissiyalar\n"
        + ("İllik xidmət haqqı 24 AZN təşkil edir. " * 10)
    ),
}


def test_split_sections_by_headings():
    sections = split_sections(PAGE["text"])
    titles = [t for t, _ in sections]
    assert titles[0] == ""                      # pre-heading intro
    assert "Faizlər" in titles
    assert "Komissiyalar" in titles
    assert all(body.strip() for _, body in sections)


def test_sliding_window_respects_target_and_overlaps():
    text = "Cümlə birinci haqqında məlumat. " * 30   # ~1_080 chars
    windows = sliding_window(text, CFG)
    assert len(windows) >= 2
    for w in windows:
        assert len(w) <= CFG.target_chars * 3   # generous upper bound (long units)
    # overlap: the tail of window i appears at the start of window i+1
    assert windows[0][-CFG.overlap_chars:] in windows[1]


def test_build_chunks_metadata_and_filtering():
    chunks = build_chunks(PAGE, CFG)
    assert len(chunks) >= 2                     # at least interest + fees sections
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))            # unique

    first = chunks[0]
    assert first.metadata["url"] == PAGE["final_url"]      # final URL wins
    assert first.metadata["source_url"] == PAGE["url"]
    assert first.metadata["lang"] == "az"
    assert first.metadata["section"] == "cards"
    assert "Birbank Taksit Kartı" in first.metadata["section_path"]
    assert all(isinstance(c.metadata[k], str) for c in chunks for k in c.metadata)


def test_tiny_chunks_are_dropped():
    page = {**PAGE, "text": "# Boş bölmə\ngözə qısa"}
    assert build_chunks(page, CFG) == []


def test_derive_section_handles_lang_prefixes_and_unknown():
    assert derive_section("https://kapitalbank.az/en/loans/cash") == "loans"
    assert derive_section("https://kapitalbank.az/ru/deposits/x") == "deposits"
    assert derive_section("https://kapitalbank.az/faq") == "faq"
    assert derive_section("https://birbank.az/az/whatever/page") == "other"
