from __future__ import annotations

import sys
from pathlib import Path

import torch

TRAINING = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import flow_matching_projected_progress as projected


def test_exact_interpolation_has_expected_progress():
    source = torch.zeros(1, 4, 3)
    target = torch.ones(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    progress, orthogonal, _, _ = projected.projection_components(
        0.5 * target, source, target, mask
    )
    assert torch.allclose(progress, torch.tensor([0.5]))
    assert torch.allclose(orthogonal, torch.zeros_like(orthogonal), atol=1e-7)


def test_orthogonal_motion_is_penalized():
    source = torch.zeros(1, 1, 3)
    target = torch.tensor([[[1.0, 0.0, 0.0]]])
    generated = torch.tensor([[[0.0, 1.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    progress, orthogonal, _, _ = projected.projection_components(
        generated, source, target, mask
    )
    assert float(progress) == 0.0
    assert float(orthogonal) > 0.0


def test_variants_are_independent():
    specs = projected.variant_specs()
    assert len(specs) == 4
    assert specs[0].lambda_dfg > specs[0].lambda_alphac
    assert specs[1].lambda_alphac > specs[1].lambda_dfg
