import pytest
import json
import numpy as np
from pathlib import Path

# Load metadata
with open("models/metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = metadata["threshold"]
FEATURES  = metadata["features"]

def test_metadata_has_required_keys():
    required = ["model", "threshold", "f1_score", "precision", "recall", "roc_auc"]
    for key in required:
        assert key in metadata

def test_threshold_in_valid_range():
    assert 0 < THRESHOLD < 1

def test_model_performance_acceptable():
    assert metadata["f1_score"] >= 0.70
    assert metadata["recall"]   >= 0.80
    assert metadata["roc_auc"]  >= 0.90

def test_feature_count():
    assert len(FEATURES) == 30
