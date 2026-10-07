"""Seed ontology loader and rule-based matchers (products, OS versions, components, features).

Loads configs/ontology.yaml. Used by: ingest.parse and ingest.select (product tagging, corpus
choice), kg.extraction / kg.run_extract / kg.resolve / kg.build (normalization, seed
edges), retrieval.linking (rule-based mentions), and the ingest/kg/bench CLIs.
Uses: core.paths.
"""

# Imports: regex matching and YAML loading; the ontology file lives in configs/.
import re
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from fixgraph.core.paths import CONFIGS_DIR

# Default location of the hand-written ontology YAML.
DEFAULT_ONTOLOGY = CONFIGS_DIR / "ontology.yaml"

# Finds OS versions like "iOS 17.2" or "macOS Ventura 13.5" in free text.
_OS_VERSION_RE = re.compile(
    # Optional marketing name between platform and number: "macOS Ventura 13.5".
    r"\b(iOS|iPadOS|watchOS|macOS|tvOS|visionOS|audioOS)\s+(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s+)?"
    r"(\d{1,2}(?:\.\d{1,2}){0,2})\b"
)


# A product family's alternative names and regex patterns for specific models.
class FamilySpec(BaseModel):
    aliases: list[str] = Field(default_factory=lambda: list[str]())
    patterns: list[str] = Field(default_factory=lambda: list[str]())


# Maps an OS platform (e.g. iOS) to the product family it runs on.
class OSPlatform(BaseModel):
    family: str


# A child-to-parent rule between ontology entries, used to add seed edges to the graph.
class Dependency(BaseModel):
    child: str
    parent: str


# The whole ontology from YAML, plus rule-based matchers that tag text with known names.
# Used by ingest for product tags and by kg/retrieval to normalize and spot mentions.
class Ontology(BaseModel):
    families: dict[str, FamilySpec]
    target_families: list[str]
    os_platforms: dict[str, OSPlatform]
    macos_names: dict[str, str]  # marketing name -> version, e.g. Catalina -> 10.15
    dependencies: list[Dependency]
    components: dict[str, list[str]]
    features: dict[str, list[str]]

    # Build one case-insensitive whole-word regex from a list of aliases, longest alias first.
    @staticmethod
    def _alias_regex(aliases: list[str]) -> re.Pattern[str]:
        alts = sorted({re.escape(a) for a in aliases}, key=len, reverse=True)
        return re.compile(r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w-])", re.IGNORECASE)

    # Compiled once and cached: one regex per product family, built from its name and aliases.
    @cached_property
    def _family_res(self) -> dict[str, re.Pattern[str]]:
        return {
            name: self._alias_regex([name, *spec.aliases]) for name, spec in self.families.items()
        }

    # Cached list of (family, regex) pairs for specific product models.
    @cached_property
    def _product_res(self) -> list[tuple[str, re.Pattern[str]]]:
        return [
            (name, re.compile(p)) for name, spec in self.families.items() for p in spec.patterns
        ]

    # Cached regex per hardware component name.
    @cached_property
    def _component_res(self) -> dict[str, re.Pattern[str]]:
        return {n: self._alias_regex([n, *a]) for n, a in self.components.items()}

    # Cached regex per software feature name.
    @cached_property
    def _feature_res(self) -> dict[str, re.Pattern[str]]:
        return {n: self._alias_regex([n, *a]) for n, a in self.features.items()}

    # Cached regex for bare macOS marketing names like "macOS Sonoma".
    @cached_property
    def _macos_name_re(self) -> re.Pattern[str]:
        names = "|".join(re.escape(n) for n in sorted(self.macos_names, key=len, reverse=True))
        # "macOS Ventura 13.5": the explicit number wins, so the name is only used when bare.
        return re.compile(rf"\bmacOS\s+({names})\b(?!\s+\d)")

    # Return which product families a text mentions; used for article product tags.
    def families_in(self, text: str) -> list[str]:
        """Product families mentioned in `text`, sorted."""
        return sorted(name for name, rx in self._family_res.items() if rx.search(text))

    # Return the specific product models found in a text, with their family.
    def products_in(self, text: str) -> list[tuple[str, str]]:
        """(family, canonical product name) pairs for specific models, sorted, deduplicated."""
        found: set[tuple[str, str]] = set()
        # Scan every model pattern and collect each match with whitespace normalized.
        for family, rx in self._product_res:
            for m in rx.finditer(text):
                found.add((family, " ".join(m.group(0).split())))
        return sorted(found)

    # Return the OS versions found in a text, turning macOS names into version numbers.
    def os_versions_in(self, text: str) -> list[str]:
        """Normalized OS versions, e.g. 'iOS 26.1', 'macOS 26' (from 'macOS Tahoe'), sorted."""
        found = {f"{p} {v}" for p, v in _OS_VERSION_RE.findall(text)}
        for name in self._macos_name_re.findall(text):
            found.add(f"macOS {self.macos_names[name]}")
        return sorted(found, key=_os_sort_key)

    # Return the known components mentioned in a text.
    def components_in(self, text: str) -> list[str]:
        return sorted(n for n, rx in self._component_res.items() if rx.search(text))

    # Return the known features mentioned in a text.
    def features_in(self, text: str) -> list[str]:
        return sorted(n for n, rx in self._feature_res.items() if rx.search(text))


# Split a version string into platform and numbers so versions can be compared and sorted.
def parse_os_version(version: str) -> tuple[str, int, int, int]:
    """'iOS 26.1' -> ('iOS', 26, 1, 0)."""
    platform, _, num = version.partition(" ")
    parts = [int(x) for x in num.split(".")] + [0, 0]
    return platform, parts[0], parts[1], parts[2]


# Sort key so OS versions sort by number, not alphabetically.
def _os_sort_key(version: str) -> tuple[str, int, int, int]:
    return parse_os_version(version)


# Read the ontology YAML and validate it into an Ontology object; callers load it once.
def load_ontology(path: Path = DEFAULT_ONTOLOGY) -> Ontology:
    with path.open(encoding="utf-8") as f:
        return Ontology.model_validate(yaml.safe_load(f))
