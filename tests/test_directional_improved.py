from __future__ import annotations

import sys
from pathlib import Path

import torch


TRAINING = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import flow_matching_cfg_directional_improved as improved


def test_directionality_cosine():
    source = torch.zeros(2, 3, 3)
    target = torch.ones(2, 3, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert torch.allclose(
        improved.displacement_cosine(target * 0.5, source, target, mask),
        torch.ones(2),
    )
    assert torch.allclose(
        improved.displacement_cosine(-target, source, target, mask),
        -torch.ones(2),
    )


def test_magnitude_error_penalizes_overdeformation():
    source = torch.zeros(1, 2, 3)
    target = torch.ones(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    exact = improved.magnitude_error(target, source, target, mask)
    excessive = improved.magnitude_error(target * 4, source, target, mask)
    assert float(exact) == 0.0
    assert float(excessive) > 0.0


def test_all_required_variants_exist():
    names = {spec.name for spec in improved.experiment_specs()}
    assert len(names) == 7
    assert "full_improved_model_15ep" in names
