from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kinase_state_comparator as comparator


def test_component_aliases():
    assert comparator.normalize_component("DFG loop") == "DFG"
    assert comparator.normalize_component("alphaC") == "alphaC"
    assert comparator.normalize_component("Lys-Glu") == "Lys-Glu"


def test_kabsch_superposition():
    reference = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)
    mobile = reference @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) + 5
    aligned = comparator.kabsch_to_reference(mobile, reference)
    assert np.allclose(aligned, reference, atol=1e-6)


def test_stratified_split_keeps_both_partitions():
    metadata = pd.DataFrame(
        {
            "kinase": ["A"] * 8 + ["B"] * 8,
            "label": [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4,
        }
    )
    train, validation = comparator.stratified_indices(metadata, 0.25, 42)
    assert len(train) == 12
    assert len(validation) == 4
    assert set(train).isdisjoint(validation)


def test_simple_yaml_fallback_schema():
    parsed = comparator.simple_yaml_load(
        'kinase: "EGFR"\ncomponents:\n  - DFG\ntraining:\n  epochs: 100\n'
    )
    assert parsed["kinase"] == "EGFR"
    assert parsed["components"] == ["DFG"]
    assert parsed["training"]["epochs"] == 100
