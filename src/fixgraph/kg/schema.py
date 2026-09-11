"""KG schema (spec §7): node labels, relation types, and head/tail type constraints.

Used by: kg.validation (`relation_allowed` rejects type-violating relations). No fixgraph imports.
"""

from typing import Literal, get_args

NodeLabel = Literal[
    "Product",
    "ProductFamily",
    "OSVersion",
    "Component",
    "Feature",
    "Symptom",
    "ErrorCode",
    "Cause",
    "Fix",
    "Article",
]

# Entity types the LLM may extract (ProductFamily and Article are derived, not extracted).
ExtractedType = Literal[
    "Product", "OSVersion", "Component", "Feature", "Symptom", "ErrorCode", "Cause", "Fix"
]

RelationType = Literal[
    "IN_FAMILY",
    "RUNS",
    "DEPENDS_ON",
    "HAS_COMPONENT",
    "EXHIBITS",
    "INVOLVES",
    "CAUSED_BY",
    "RESOLVED_BY",
    "ADDRESSES",
    "REQUIRES",
    "APPLIES_TO",
    "SIGNALS",
]

# Relations the LLM may extract, with allowed (head types, tail types).
EXTRACTED_RELATIONS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "RUNS": (frozenset({"Product"}), frozenset({"OSVersion"})),
    "DEPENDS_ON": (frozenset({"Product"}), frozenset({"Product"})),
    "HAS_COMPONENT": (frozenset({"Product"}), frozenset({"Component"})),
    "EXHIBITS": (frozenset({"Product"}), frozenset({"Symptom"})),
    "INVOLVES": (frozenset({"Symptom"}), frozenset({"Component", "Feature"})),
    "CAUSED_BY": (frozenset({"Symptom"}), frozenset({"Cause"})),
    "RESOLVED_BY": (frozenset({"Symptom"}), frozenset({"Fix"})),
    "ADDRESSES": (frozenset({"Fix"}), frozenset({"Cause"})),
    "REQUIRES": (frozenset({"Fix"}), frozenset({"OSVersion", "Product", "Feature"})),
    "APPLIES_TO": (frozenset({"Fix", "Symptom"}), frozenset({"OSVersion"})),
    "SIGNALS": (frozenset({"ErrorCode"}), frozenset({"Symptom"})),
}

EXTRACTED_TYPES: tuple[str, ...] = get_args(ExtractedType)
ALL_RELATIONS: tuple[str, ...] = get_args(RelationType)


def relation_allowed(relation: str, head_type: str, tail_type: str) -> bool:
    spec = EXTRACTED_RELATIONS.get(relation)
    return spec is not None and head_type in spec[0] and tail_type in spec[1]
