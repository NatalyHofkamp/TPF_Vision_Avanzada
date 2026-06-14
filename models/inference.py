"""Inference entry points backed by official FoldFlow reverse integration."""

from __future__ import annotations

from typing import Mapping

import torch

from .foldflow_backbone import FoldFlowBackbone


@torch.no_grad()
def generate_samples(
    model: FoldFlowBackbone,
    foldflow_features: Mapping[str, torch.Tensor],
    conditioning: Mapping[str, torch.Tensor],
    num_samples: int = 1,
    guidance_scale: float = 1.0,
    num_steps: int = 50,
    min_t: float = 0.01,
    noise_scale: float = 0.0,
):
    """Generate translation trajectories with the official FoldFlow matcher."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    outputs = []
    for _ in range(num_samples):
        outputs.append(
            model.sample_translation(
                foldflow_features=foldflow_features,
                conditioning=conditioning,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                min_t=min_t,
                noise_scale=noise_scale,
            )
        )
    return outputs
