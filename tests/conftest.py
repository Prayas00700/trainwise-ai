import os
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRAINWISE_SCALE", "small")

from trainwise.platform import Platform  # noqa: E402

TODAY = date(2026, 10, 15)


@pytest.fixture(scope="session")
def platform(tmp_path_factory):
    return Platform(artifact_dir=tmp_path_factory.mktemp("artifacts"), scale="small", fresh=True, today=TODAY)


@pytest.fixture()
def fresh_platform(tmp_path):
    return Platform(artifact_dir=tmp_path / "artifacts", scale="small", fresh=True, today=TODAY)
