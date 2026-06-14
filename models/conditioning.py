"""Conditioning adapters for the official FoldFlow backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn

KINASES = (
    "EGFR",
    "BRAF",
    "ABL1",
    "FGFR1",
    "MET",
    "KIT",
    "ALK",
    "PIK3CA",
    "AKT1",
    "CDK4",
    "CDK6",
    "PDGFRA",
    "ERBB2",
)
KINASE_TO_ID = {kinase: index for index, kinase in enumerate(KINASES)}
STATE_TO_ID = {"inactive": 0, "active": 1}


class ESMConditionEncoder(nn.Module):
    """Project cached residue-level ESM2 embeddings to FoldFlow node width."""

    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 256,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, esm_embedding: torch.Tensor, residue_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if esm_embedding.ndim != 3 or esm_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected ESM embeddings [B, N, {self.input_dim}], "
                f"received {tuple(esm_embedding.shape)}"
            )
        output = self.network(esm_embedding.float())
        if residue_mask is not None:
            output = output * residue_mask[..., None].to(output.dtype)
        return output


class KinaseConditionEncoder(nn.Module):
    """Trainable embeddings for the 13 supported kinases."""

    def __init__(self, output_dim: int = 256, kinases: Sequence[str] = KINASES):
        super().__init__()
        if len(set(kinases)) != len(kinases):
            raise ValueError("Kinase vocabulary contains duplicate names")
        self.kinases = tuple(kinases)
        self.kinase_to_id = {
            kinase: index for index, kinase in enumerate(self.kinases)
        }
        self.output_dim = output_dim
        self.embedding = nn.Embedding(len(self.kinases), output_dim)

    def encode_names(self, kinase_names: Sequence[str], device=None) -> torch.Tensor:
        unknown = sorted(set(kinase_names) - set(self.kinase_to_id))
        if unknown:
            raise ValueError(f"Unsupported kinases: {unknown}")
        return torch.tensor(
            [self.kinase_to_id[name] for name in kinase_names],
            dtype=torch.long,
            device=device,
        )

    def forward(self, kinase_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(kinase_id.long())


class StateConditionEncoder(nn.Module):
    """Trainable active/inactive target-state embeddings."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(len(STATE_TO_ID), output_dim)

    def encode_names(self, states: Sequence[str], device=None) -> torch.Tensor:
        unknown = sorted(set(states) - set(STATE_TO_ID))
        if unknown:
            raise ValueError(f"Unsupported conformational states: {unknown}")
        return torch.tensor(
            [STATE_TO_ID[state] for state in states],
            dtype=torch.long,
            device=device,
        )

    def forward(
        self, target_state: torch.Tensor, drop_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        embedding = self.embedding(target_state.long())
        if drop_mask is not None:
            embedding = embedding * (~drop_mask.bool()).unsqueeze(-1).to(embedding.dtype)
        return embedding


@dataclass(frozen=True)
class ConditionDimensions:
    esm_input: int
    foldflow_node: int
    kinase: int
    state: int


class FoldFlowConditionEncoder(nn.Module):
    """Combine ESM, kinase, and optional target-state node conditions."""

    def __init__(
        self,
        esm_input_dim: int = 1280,
        node_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.esm = ESMConditionEncoder(
            input_dim=esm_input_dim,
            output_dim=node_dim,
            dropout=dropout,
        )
        self.kinase = KinaseConditionEncoder(output_dim=node_dim)
        self.state = StateConditionEncoder(output_dim=node_dim)
        self.output_norm = nn.LayerNorm(node_dim)
        self.dimensions = ConditionDimensions(
            esm_input=esm_input_dim,
            foldflow_node=node_dim,
            kinase=node_dim,
            state=node_dim,
        )

    def forward(
        self,
        conditioning: Mapping[str, torch.Tensor],
        residue_mask: torch.Tensor,
        drop_target_state: bool | torch.Tensor = False,
    ) -> torch.Tensor:
        required = {"esm_embedding", "kinase_id", "target_state"}
        missing = required - set(conditioning)
        if missing:
            raise ValueError(f"Conditioning is missing keys: {sorted(missing)}")
        esm = self.esm(conditioning["esm_embedding"], residue_mask)
        kinase = self.kinase(conditioning["kinase_id"])
        state_ids = conditioning["target_state"]
        if isinstance(drop_target_state, bool):
            drop_mask = torch.full(
                state_ids.shape,
                drop_target_state,
                dtype=torch.bool,
                device=state_ids.device,
            )
        else:
            drop_mask = drop_target_state.to(device=state_ids.device, dtype=torch.bool)
        state = self.state(state_ids, drop_mask=drop_mask)
        combined = esm + kinase[:, None, :] + state[:, None, :]
        return self.output_norm(combined) * residue_mask[..., None].to(combined.dtype)

    @staticmethod
    def sample_target_dropout(
        batch_size: int, probability: float, device=None
    ) -> torch.Tensor:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("CFG dropout probability must be in [0, 1]")
        return torch.rand(batch_size, device=device) < probability
