"""Qualified semantic keys used by C1 alignment audits.

Raw labels are deliberately not treated as cross-source positives.  A key is
eligible only when its metadata column declares a canonical disease or brain
region namespace.  Source-native cell labels and unqualified free text remain
unavailable until an explicit cross-source mapping is frozen.
"""

from __future__ import annotations

from typing import Mapping


SEMANTIC_ALIGNMENT_POLICY_VERSION = "neuronsifter-c1-qualified-semantic-key-v1"


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "<na>"} else text


def qualified_semantic_alignment_key(row: Mapping[str, object]) -> str:
    """Return a namespace-and-axis-qualified key, or ``""`` if unavailable."""

    if _text(row.get("action_uid")):
        return ""
    # These fields are the only V1 metadata axes with an explicit canonical or
    # ontology namespace.  ``cell_type_id`` and bare disease/region labels are
    # source-native and must not create cross-source positives by string match.
    candidates = (
        ("disease_state_canonical", "disease_state_canonical"),
        ("brain_region_ontology_id", "brain_region_ontology_id"),
        ("brain_region_canonical", "brain_region_canonical"),
    )
    for field, namespace in candidates:
        value = _text(row.get(field))
        if value:
            return f"{namespace}|{value}"
    return ""
