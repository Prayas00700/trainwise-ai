"""Central configuration: paths, seed and governance policies (policy-as-code)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
KB_PATH = ROOT_DIR / "data" / "kb" / "policies.json"
DEFAULT_ARTIFACT_DIR = Path(os.getenv("TRAINWISE_ARTIFACT_DIR", str(ROOT_DIR / "artifacts")))
SEED = int(os.getenv("TRAINWISE_SEED", "7"))


@lru_cache(maxsize=1)
def load_policies() -> dict:
    with open(CONFIG_DIR / "governance_policies.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def model_policy(name: str) -> dict:
    """Effective policy for one model: defaults overlaid with model-specific settings."""
    pol = load_policies()
    return {**pol.get("model_defaults", {}), **(pol.get("models", {}).get(name) or {})}
