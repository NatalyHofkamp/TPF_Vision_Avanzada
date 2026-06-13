from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TRAINING = Path(__file__).resolve().parents[1] / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import classification_diagnostics as diagnostics


def test_canonicalization_is_centered_and_scaled():
    coords = np.arange(30, dtype=float).reshape(10, 3)
    result = diagnostics.canonicalize(coords)
    assert np.allclose(result.mean(0), 0, atol=1e-6)
    assert np.isclose(np.sqrt(np.mean(np.sum(result**2, axis=1))), 1)


def test_interpolation_has_fixed_size():
    coords = np.random.default_rng(1).normal(size=(17, 3))
    assert diagnostics.interpolate_coords(coords).shape == (64, 3)
