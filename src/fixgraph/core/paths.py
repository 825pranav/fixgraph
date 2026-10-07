"""Filesystem locations. Always pathlib, never string concatenation (spec §5.8).

REPO_ROOT, configs/ and `DataPaths` (every file under data/). Used by core.config (exposed to
the rest of the code as `settings.paths`) and core.ontology. No fixgraph imports.
"""

# Imports: only stdlib, so this module sits at the bottom of the dependency chain.
from dataclasses import dataclass
from pathlib import Path

# Repo root is found from this file's location, so code works no matter where it is launched from.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIGS_DIR: Path = REPO_ROOT / "configs"
DEFAULT_CONFIG_FILE: Path = CONFIGS_DIR / "base.yaml"


# Map of every data folder and file; each property returns a Path under the data root.
@dataclass(frozen=True)
class DataPaths:
    root: Path

    # Raw scraped HTML pages, one file per article (written by the scraper, read by the parser).
    @property
    def raw_html(self) -> Path:
        return self.root / "raw" / "html"

    # List of article URLs taken from Apple's sitemap.
    @property
    def article_urls(self) -> Path:
        return self.root / "raw" / "article_urls.json"

    # Corpus folder holding the parquet outputs of ingest.
    @property
    def corpus(self) -> Path:
        return self.root / "corpus"

    # Parsed Article records (output of parse + select).
    @property
    def articles(self) -> Path:
        return self.corpus / "articles.parquet"

    # Chunk records that feed both the search index and KG extraction.
    @property
    def chunks(self) -> Path:
        return self.corpus / "chunks.parquet"

    # Folder of raw KG extraction JSONL files from the LLM.
    @property
    def extractions(self) -> Path:
        return self.root / "extractions"

    # Built knowledge graph: nodes, edges and mentions parquet files.
    @property
    def kg(self) -> Path:
        return self.root / "kg"

    # Hand-labelled or drafted gold data used for evaluation.
    @property
    def gold(self) -> Path:
        return self.root / "gold"

    # Benchmark question sets read by the harness.
    @property
    def bench(self) -> Path:
        return self.root / "bench"

    # Qdrant store and BM25 stats used at query time.
    @property
    def index(self) -> Path:
        return self.root / "index"

    # Output folder for benchmark runs and reports.
    @property
    def results(self) -> Path:
        return self.root / "results"
