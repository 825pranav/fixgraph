"""Settings: init kwargs > environment > .env > configs/base.yaml > defaults."""

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

LLMBackend = Literal["ollama", "openai", "fake"]


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        yaml_file=DEFAULT_CONFIG_FILE,
        yaml_file_encoding="utf-8",
    )

    data_dir: Path = Path("data")
    seed: int = 13
    llm: LLMSettings = LLMSettings()

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

    def resolve(self, path: Path) -> Path:
        """Resolve a configured relative path against the repo root."""
        return path if path.is_absolute() else REPO_ROOT / path

    @property
    def paths(self) -> DataPaths:
        return DataPaths(self.resolve(self.data_dir))


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
