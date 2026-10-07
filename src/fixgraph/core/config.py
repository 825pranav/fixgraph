"""Settings: init kwargs > environment > .env > configs/base.yaml > defaults.

`load_settings()` is the single entry point: every CLI (cli.py, ingest/kg/gnn/bench cli.py) and
api/app.py call it; llm/factory.py reads the LLM section to pick a backend.
Uses: core.paths (repo root, default config file, DataPaths).
"""

# Imports: pydantic-settings does the layered config loading; paths gives repo root and data layout.
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from fixgraph.core.paths import DEFAULT_CONFIG_FILE, REPO_ROOT, DataPaths

# The three LLM backends the factory can build; "fake" is for tests and offline demo mode.
LLMBackend = Literal["ollama", "openai", "fake"]


# LLM section of the config: which model does which job (extract, answer, judge) and call options.
# Every LLM client and the SQLite reply cache read their settings from here.
class LLMSettings(BaseModel):
    backend: LLMBackend = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    extraction_model: str = "qwen3:4b"
    answer_model: str = "qwen3:4b"
    linking_model: str = "qwen3:4b"
    judge_model: str = "qwen3:8b"
    num_ctx: int = 8192
    temperature: float = 0.0
    think: bool = False
    keep_alive: str = "2m"
    timeout_s: float = 300.0
    cache_path: Path = Path("data/cache/llm_cache.sqlite")


# S1 rerank settings: which cross-encoder, how many candidates it rescores, optional blend.
class RetrievalSettings(BaseModel):
    """S1's rerank stage (DECISIONS.md D39-D41). Names come from retrieval.rerank.RERANKERS.
    Default = the benchmarked S1. The recall-oriented variant from the reranking study is
    `second_reranker: qwen` with `second_weight: 0.5` (about 2.8x the retrieval latency)."""

    reranker: str = "bge"
    rerank_top: int = 30
    second_reranker: str | None = None
    second_weight: float = 0.0
    first_stage_weight: float = 0.0


# Top-level settings object that the CLIs and the API load once and pass around.
class Settings(BaseSettings):
    # Read .env and configs/base.yaml; "__" lets an env var like LLM__ANSWER_MODEL set a nested key.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=DEFAULT_CONFIG_FILE,
        yaml_file_encoding="utf-8",
    )

    # Core fields: where data lives, the random seed, and the nested LLM and retrieval sections.
    data_dir: Path = Path("data")
    seed: int = 13
    llm: LLMSettings = LLMSettings()
    retrieval: RetrievalSettings = RetrievalSettings()

    # Set the priority order: code kwargs win, then env vars, then .env, then the YAML file.
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
        )

    # Turn a relative config path into an absolute one under the repo root.
    def resolve(self, path: Path) -> Path:
        """Resolve a configured relative path against the repo root."""
        return path if path.is_absolute() else REPO_ROOT / path

    # Give every caller one DataPaths object so file locations are defined in a single place.
    @property
    def paths(self) -> DataPaths:
        return DataPaths(self.resolve(self.data_dir))


# Single entry point used by every CLI and the API; overrides are mainly handy in tests.
def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
