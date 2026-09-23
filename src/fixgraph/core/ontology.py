"""Seed ontology loader and rule-based matchers (products, OS versions, components, features)."""

import re
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from fixgraph.core.paths import CONFIGS_DIR

DEFAULT_ONTOLOGY = CONFIGS_DIR / "ontology.yaml"

_OS_VERSION_RE = re.compile(
    # Optional marketing name between platform and number: "macOS Ventura 13.5".
    r"\b(iOS|iPadOS|watchOS|macOS|tvOS|visionOS|audioOS)\s+(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\s+)?"
    r"(\d{1,2}(?:\.\d{1,2}){0,2})\b"
)


class FamilySpec(BaseModel):
    aliases: list[str] = Field(default_factory=lambda: list[str]())
    patterns: list[str] = Field(default_factory=lambda: list[str]())


class OSPlatform(BaseModel):
    family: str


class Dependency(BaseModel):
    child: str
    parent: str


class Ontology(BaseModel):
    families: dict[str, FamilySpec]
    target_families: list[str]
    os_platforms: dict[str, OSPlatform]
    macos_names: dict[str, str]  # marketing name -> version, e.g. Catalina -> 10.15
    dependencies: list[Dependency]
    components: dict[str, list[str]]
    features: dict[str, list[str]]

    @staticmethod
    def _alias_regex(aliases: list[str]) -> re.Pattern[str]:
        alts = sorted({re.escape(a) for a in aliases}, key=len, reverse=True)
        return re.compile(r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w-])", re.IGNORECASE)

    @cached_property
    def _family_res(self) -> dict[str, re.Pattern[str]]:
        return {
            name: self._alias_regex([name, *spec.aliases]) for name, spec in self.families.items()
        }

    @cached_property
    def _product_res(self) -> list[tuple[str, re.Pattern[str]]]:
        return [
            (name, re.compile(p)) for name, spec in self.families.items() for p in spec.patterns
        ]

    @cached_property
    def _component_res(self) -> dict[str, re.Pattern[str]]:
        return {n: self._alias_regex([n, *a]) for n, a in self.components.items()}

    @cached_property
    def _feature_res(self) -> dict[str, re.Pattern[str]]:
        return {n: self._alias_regex([n, *a]) for n, a in self.features.items()}

    @cached_property
    def _macos_name_re(self) -> re.Pattern[str]:
        names = "|".join(re.escape(n) for n in sorted(self.macos_names, key=len, reverse=True))
        # "macOS Ventura 13.5": the explicit number wins, so the name is only used when bare.
        return re.compile(rf"\bmacOS\s+({names})\b(?!\s+\d)")

    def families_in(self, text: str) -> list[str]:
        """Product families mentioned in `text`, sorted."""
        return sorted(name for name, rx in self._family_res.items() if rx.search(text))

    def products_in(self, text: str) -> list[tuple[str, str]]:
        """(family, canonical product name) pairs for specific models, sorted, deduplicated."""
        found: set[tuple[str, str]] = set()
        for family, rx in self._product_res:
            for m in rx.finditer(text):
                found.add((family, " ".join(m.group(0).split())))
        return sorted(found)

    def os_versions_in(self, text: str) -> list[str]:
        """Normalized OS versions, e.g. 'iOS 26.1', 'macOS 26' (from 'macOS Tahoe'), sorted."""
        found = {f"{p} {v}" for p, v in _OS_VERSION_RE.findall(text)}
        for name in self._macos_name_re.findall(text):
            found.add(f"macOS {self.macos_names[name]}")
        return sorted(found, key=_os_sort_key)

    def components_in(self, text: str) -> list[str]:
        return sorted(n for n, rx in self._component_res.items() if rx.search(text))

    def features_in(self, text: str) -> list[str]:
        return sorted(n for n, rx in self._feature_res.items() if rx.search(text))


def parse_os_version(version: str) -> tuple[str, int, int, int]:
    """'iOS 26.1' -> ('iOS', 26, 1, 0)."""
    platform, _, num = version.partition(" ")
    parts = [int(x) for x in num.split(".")] + [0, 0]
    return platform, parts[0], parts[1], parts[2]


def _os_sort_key(version: str) -> tuple[str, int, int, int]:
    return parse_os_version(version)


def load_ontology(path: Path = DEFAULT_ONTOLOGY) -> Ontology:
    with path.open(encoding="utf-8") as f:
        return Ontology.model_validate(yaml.safe_load(f))
