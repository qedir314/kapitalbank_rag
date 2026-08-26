from kb_rag.config import get_settings
from kb_rag.evaluation.dataset import load_dataset


def test_golden_set_loads_and_is_wellformed():
    items = load_dataset(get_settings().golden_set_path)
    assert len(items) >= 25                      # enough for meaningful aggregates
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))             # unique ids
    answerable = [i for i in items if not i.unanswerable]
    unanswerable = [i for i in items if i.unanswerable]
    assert answerable and unanswerable           # both behaviors are covered
    for item in answerable:
        assert item.expected_sources, f"{item.id} needs expected_sources"
    assert {i.lang for i in items} == {"az", "en", "ru"}   # trilingual coverage
