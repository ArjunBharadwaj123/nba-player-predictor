"""Shared pytest fixtures/config for NFL + NBA tests."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NFL_DEV_MODE", "1")

MODELS_SAVED = ROOT / "nfl" / "models" / "saved"
SERVING = ROOT / "nfl" / "data" / "processed" / "features_serving.parquet"


def have_artifacts() -> bool:
    return (MODELS_SAVED / "feature_names.json").exists() and SERVING.exists()


requires_artifacts = pytest.mark.skipif(
    not have_artifacts(),
    reason="NFL model/serving artifacts not built (run `python -m nfl.pipeline.update`)",
)
