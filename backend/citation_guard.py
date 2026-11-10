"""Small validation helpers for source references in generated answers."""

import re


def citations_reference_sources(answer: str, source_ids: set[str]) -> bool:
    """Return true only when an answer cites known source IDs at least once."""
    cited = set(re.findall(r"\[(S\d+)\]", answer))
    return bool(cited) and cited.issubset(source_ids)
