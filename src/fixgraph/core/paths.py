"""Filesystem locations. Always pathlib, never string concatenation (spec §5.8)."""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIGS_DIR: Path = REPO_ROOT / "configs"
DEFAULT_CONFIG_FILE: Path = CONFIGS_DIR / "base.yaml"


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw_html(self) -> Path:
        return self.root / "raw" / "html"

    @property
    def article_urls(self) -> Path:
        return self.root / "raw" / "article_urls.json"

    @property
    def corpus(self) -> Path:
        return self.root / "corpus"

    @property
    def articles(self) -> Path:
        return self.corpus / "articles.parquet"

    @property
    def chunks(self) -> Path:
        return self.corpus / "chunks.parquet"

    @property
    def extractions(self) -> Path:
        return self.root / "extractions"

    @property
    def kg(self) -> Path:
        return self.root / "kg"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def bench(self) -> Path:
        return self.root / "bench"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def results(self) -> Path:
        return self.root / "results"
