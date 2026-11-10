from citation_guard import citations_reference_sources


def test_accepts_known_citations():
    assert citations_reference_sources("Wear gloves [S1] and goggles [S2].", {"S1", "S2"})


def test_rejects_missing_citations():
    assert not citations_reference_sources("Wear chemical resistant gloves.", {"S1"})


def test_rejects_unknown_citations():
    assert not citations_reference_sources("Wear gloves [S3].", {"S1", "S2"})


def test_rejects_mixed_known_and_unknown_citations():
    assert not citations_reference_sources("Wear gloves [S1] [S9].", {"S1", "S2"})
