"""Filesystem locations. Always pathlib, never string concatenation (spec §5.8)."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CONFIGS_DIR: Path = REPO_ROOT / "configs"
DEFAULT_CONFIG_FILE: Path = CONFIGS_DIR / "base.yaml"
