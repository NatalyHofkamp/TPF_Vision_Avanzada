"""Mobility-guided EGNN for kinase conformational translation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MobilityEGNNConfig:
    input_dim: int
    hidden_dim: int = 128
    message_dim: int = 128
    num_layers: int = 4
    coord_update_scale: float = 0.01
    delta_scale: float = 0.5
    use_binary_mobile_mask: bool = True
    mobility_gate_temperature: float = 1.0
    gate_floor: float = 0.05


class MobilityEGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, message_dim: int, coord_update_scale: float):
        super().__init__()
        edge_in = hidden_dim * 2 + 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, message_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_in, message_dim),
            nn.SiLU(),
            nn.Linear(message_dim, 1),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.coord_update_scale = coord_update_scale

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nres = coords.shape[1]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(nres, device=coords.device, dtype=pair_mask.dtype).unsqueeze(0)
        pair_mask = pair_mask * (1.0 - eye)
        rel = coords[:, :, None, :] - coords[:, None, :, :]
        dist2 = rel.pow(2).sum(dim=-1, keepdim=True)
        hi = h[:, :, None, :].expand(-1, -1, nres, -1)
        hj = h[:, None, :, :].expand(-1, nres, -1, -1)
        edge_input = torch.cat([hi, hj, dist2], dim=-1)
        degree = pair_mask.sum(dim=2, keepdim=True).clamp_min(1.0)

        messages = self.edge_mlp(edge_input) * pair_mask.unsqueeze(-1)
        agg = messages.sum(dim=2) / degree
        h = self.node_norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))

        coord_weights = torch.tanh(self.coord_mlp(edge_input)) * pair_mask.unsqueeze(-1)
        coord_update = (coord_weights * rel).sum(dim=2) / degree
        coords = coords + self.coord_update_scale * coord_update
        coords = coords * mask.unsqueeze(-1)
        return h, coords


class MobilityGuidedEGNN(nn.Module):
    """Mobility-gated EGNN conditioned on ESM + mobility prior."""

    def __init__(self, config: MobilityEGNNConfig):
        super().__init__()
        self.config = config
        binary_dim = 1 if config.use_binary_mobile_mask else 0
        input_dim = config.input_dim + 2 + binary_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            [
                MobilityEGNNLayer(
                    hidden_dim=config.hidden_dim,
                    message_dim=config.message_dim,
                    coord_update_scale=config.coord_update_scale,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        self.motion_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 3),
        )
        self.mobile_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def _mobility_gate(
        self,
        mobility_prior: torch.Tensor,
        binary_mobile_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mobility_prior.ndim == 2:
            mobility_prior = mobility_prior.unsqueeze(-1)
        gate = mobility_prior.clamp(0.0, 1.0)
        if binary_mobile_mask is not None:
            if binary_mobile_mask.ndim == 2:
                binary_mobile_mask = binary_mobile_mask.unsqueeze(-1)
            gate = torch.maximum(gate, binary_mobile_mask.float())
        gate = torch.clamp(gate, min=self.config.gate_floor, max=1.0)
        if self.config.mobility_gate_temperature != 1.0:
            gate = gate.pow(1.0 / self.config.mobility_gate_temperature)
        return gate

    def forward(
        self,
        coords: torch.Tensor,
        esm: torch.Tensor,
        mobility_prior: torch.Tensor,
        mask: torch.Tensor,
        binary_mobile_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = self._mobility_gate(mobility_prior, binary_mobile_mask)
        if mobility_prior.ndim == 2:
            mobility_prior = mobility_prior.unsqueeze(-1)
        if binary_mobile_mask is not None and binary_mobile_mask.ndim == 2:
            binary_mobile_mask = binary_mobile_mask.unsqueeze(-1)
        node_features = [esm, mobility_prior, gate]
        if self.config.use_binary_mobile_mask:
            if binary_mobile_mask is None:
                binary_mobile_mask = torch.zeros_like(mobility_prior)
            node_features.append(binary_mobile_mask)
        h = self.input_proj(torch.cat(node_features, dim=-1))
        coords_work = coords
        for layer in self.layers:
            h, coords_work = layer(h, coords_work, mask)
        motion_raw = self.motion_head(h).squeeze(-1)
        pred_motion = torch.nn.functional.softplus(motion_raw) * gate.squeeze(-1) * self.config.delta_scale
        direction = self.direction_head(h)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pred_direction = direction * mask.unsqueeze(-1)
        pred_delta = pred_motion.unsqueeze(-1) * pred_direction
        pred_delta = pred_delta * mask.unsqueeze(-1)
        pred_coords = coords_work + pred_delta
        pred_mobile_logits = self.mobile_head(h).squeeze(-1)
        return pred_delta, pred_coords, pred_motion, pred_direction, pred_mobile_logits, gate.squeeze(-1)


def weighted_delta_loss(
    pred_delta: torch.Tensor,
    true_delta: torch.Tensor,
    mobility_prior: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    if mobility_prior.ndim == 2:
        mobility_prior = mobility_prior.unsqueeze(-1)
    weights = (1.0 + alpha * mobility_prior).float() * mask.unsqueeze(-1).float()
    return ((pred_delta - true_delta).pow(2) * weights).sum() / weights.sum().clamp_min(1.0) / 3.0


def weighted_motion_loss(
    pred_motion: torch.Tensor,
    true_motion: torch.Tensor,
    mobility_prior: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    if mobility_prior.ndim == 2:
        mobility_prior = mobility_prior.unsqueeze(-1)
    weights = (1.0 + alpha * mobility_prior).float() * mask.float()
    return ((pred_motion - true_motion).pow(2) * weights).sum() / weights.sum().clamp_min(1.0)


def direction_alignment_loss(
    pred_direction: torch.Tensor,
    true_direction: torch.Tensor,
    mobile_mask: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mobile_mask.float() * mask.float()
    if weights.sum() <= 0:
        return torch.zeros((), device=pred_direction.device)
    pred_direction = pred_direction / pred_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    true_direction = true_direction / true_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((pred_direction - true_direction).pow(2).sum(dim=-1) * weights).sum() / weights.sum().clamp_min(1.0)


def mobile_classification_loss(
    pred_mobile_logits: torch.Tensor,
    mobile_mask: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_mobile_logits, mobile_mask.float(), reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def local_pairwise_distance_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    mask: torch.Tensor,
    window_radius: int = 4,
    focus_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    nres = pred_coords.shape[1]
    idx = torch.arange(nres, device=pred_coords.device)
    local_mask = (idx[None, :] - idx[:, None]).abs() <= window_radius
    local_mask = local_mask & (~torch.eye(nres, dtype=torch.bool, device=pred_coords.device))
    pair_mask = mask[:, :, None] * mask[:, None, :]
    if focus_mask is not None:
        if focus_mask.ndim == 2:
            focus_mask = focus_mask.unsqueeze(-1)
        pair_mask = pair_mask * (focus_mask[:, :, None] * focus_mask[:, None, :])
    pair_mask = pair_mask * local_mask.unsqueeze(0).float()
    pred_dist = torch.cdist(pred_coords, pred_coords)
    target_dist = torch.cdist(target_coords, target_coords)
    denom = pair_mask.sum().clamp_min(1.0)
    return (((pred_dist - target_dist).pow(2)) * pair_mask).sum() / denom
