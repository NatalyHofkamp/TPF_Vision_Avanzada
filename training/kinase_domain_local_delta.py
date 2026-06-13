"""Kinase-domain alignment and local delta generation.

This module extracts comparable motif-anchored kinase domains, keeps exact
residue correspondence, and trains a masked Flow-Matching model whose output is
a local coordinate delta rather than a full unconstrained structure.
"""

from __future__ import annotations

import difflib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import plausible_inverse_state_generation as base

STATES = ("inactive", "active")
REGIONS = ("dfg", "alphac", "hrd", "activation_loop")


@dataclass
class Config:
    seed: int = 42
    epochs: int = 500
    patience: int = 30
    batch_size: int = 8
    num_points: int = 144
    hidden_dim: int = 160
    embedding_dim: int = 32
    num_layers: int = 4
    dropout: float = 0.10
    condition_dropout: float = 0.15
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    guidance_scale: float = 1.5
    guidance_sweep: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    integration_steps: int = 24
    validation_every: int = 5
    grad_clip: float = 1.0
    noise_std: float = 0.025
    max_pairs_per_kinase: int = 96
    min_domain_length: int = 80
    max_domain_length: int = 210
    min_alignment_coverage: float = 0.72
    min_sequence_identity: float = 0.80
    max_gap_fraction: float = 0.28
    ambiguous_rmsd: float = 1.0
    flow_weight: float = 1.0
    local_weight: float = 0.45
    global_weight: float = 0.30
    distance_weight: float = 0.15
    bond_weight: float = 0.20
    smoothness_weight: float = 0.08
    geometry_weight: float = 0.15
    num_workers: int = 0


@dataclass(frozen=True)
class DomainRecord:
    structure: base.Structure
    start: int
    end: int
    original_indices: tuple[int, ...]
    sequence: str
    coords: tuple[tuple[float, float, float], ...]
    residue_numbers: tuple[int, ...]
    amino_acids: tuple[str, ...]
    dfg_index: int | None
    hrd_index: int | None
    vaik_lys_index: int | None
    alphac_glu_index: int | None

    @property
    def pdb_id(self) -> str:
        return self.structure.pdb_id

    @property
    def kinase(self) -> str:
        return self.structure.kinase

    @property
    def state(self) -> str:
        return self.structure.state

    @property
    def split(self) -> str:
        return self.structure.split


def set_seed(seed: int) -> None:
    base.set_seed(seed)


def device_name() -> str:
    return base.device_name()


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def extract_domain(record: base.Structure, config: Config) -> tuple[DomainRecord | None, str]:
    anchors = {
        "dfg": record.dfg_index,
        "hrd": record.hrd_index,
        "lys": record.vaik_lys_index,
        "alphac": record.alphac_glu_index,
    }
    if any(value is None for value in anchors.values()):
        missing = ",".join(name for name, value in anchors.items() if value is None)
        return None, f"missing_motifs:{missing}"

    # Motif-anchored catalytic core. This deliberately removes long regulatory
    # regions from PIK3CA and preserves the same functional span in all kinases.
    start = min(
        int(record.alphac_glu_index) - 28,
        int(record.vaik_lys_index) - 42,
        int(record.hrd_index) - 52,
    )
    end = max(int(record.dfg_index) + 58, int(record.hrd_index) + 48)
    start, end = max(0, start), min(len(record.residues), end)
    length = end - start
    if length < config.min_domain_length:
        return None, "domain_too_short"
    if length > config.max_domain_length:
        center = (int(record.vaik_lys_index) + int(record.dfg_index)) // 2
        half = config.max_domain_length // 2
        start = max(0, center - half)
        end = min(len(record.residues), start + config.max_domain_length)
        start = max(0, end - config.max_domain_length)

    residues = record.residues[start:end]
    if not residues:
        return None, "empty_domain"

    def local(index: int | None) -> int | None:
        return index - start if index is not None and start <= index < end else None

    domain = DomainRecord(
        structure=record,
        start=start,
        end=end,
        original_indices=tuple(range(start, end)),
        sequence="".join(residue.aa for residue in residues),
        coords=tuple(residue.ca for residue in residues),
        residue_numbers=tuple(residue.number for residue in residues),
        amino_acids=tuple(residue.aa for residue in residues),
        dfg_index=local(record.dfg_index),
        hrd_index=local(record.hrd_index),
        vaik_lys_index=local(record.vaik_lys_index),
        alphac_glu_index=local(record.alphac_glu_index),
    )
    if any(
        value is None
        for value in (
            domain.dfg_index,
            domain.hrd_index,
            domain.vaik_lys_index,
            domain.alphac_glu_index,
        )
    ):
        return None, "motif_outside_domain"
    return domain, ""


def load_domains(
    root: Path, config: Config
) -> tuple[dict[str, list[DomainRecord]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, structure_audit, dataset_summary = base.load_structures(root)
    domains: dict[str, list[DomainRecord]] = {"train": [], "val": [], "test": []}
    rows = []
    for split, values in records.items():
        for record in values:
            domain, reason = extract_domain(record, config)
            rows.append(
                {
                    "split": split,
                    "pdb_id": record.pdb_id,
                    "kinase": record.kinase,
                    "state": record.state,
                    "original_length": len(record.residues),
                    "domain_length": len(domain.sequence) if domain else np.nan,
                    "domain_start_original_index": domain.start if domain else np.nan,
                    "domain_end_original_index": domain.end if domain else np.nan,
                    "kept_original_indices": (
                        "|".join(map(str, domain.original_indices)) if domain else ""
                    ),
                    "removed_residues": (
                        len(record.residues) - len(domain.sequence) if domain else len(record.residues)
                    ),
                    "dfg_found": domain is not None and domain.dfg_index is not None,
                    "alphac_found": domain is not None and domain.alphac_glu_index is not None,
                    "hrd_found": domain is not None and domain.hrd_index is not None,
                    "lys_glu_found": (
                        domain is not None
                        and domain.vaik_lys_index is not None
                        and domain.alphac_glu_index is not None
                    ),
                    "accepted": domain is not None,
                    "discard_reason": reason,
                }
            )
            if domain:
                domains[split].append(domain)
    return domains, pd.DataFrame(rows), structure_audit, dataset_summary


def region_indices(domain: DomainRecord, region: str) -> set[int]:
    if region == "dfg":
        return set(range(domain.dfg_index - 2, domain.dfg_index + 5))
    if region == "alphac":
        return set(range(domain.alphac_glu_index - 7, domain.alphac_glu_index + 8))
    if region == "hrd":
        return set(range(domain.hrd_index - 2, domain.hrd_index + 5))
    if region == "activation_loop":
        return set(range(domain.dfg_index, min(len(domain.sequence), domain.dfg_index + 25)))
    return set()


def align_domains(source: DomainRecord, target: DomainRecord) -> dict[str, Any]:
    matcher = difflib.SequenceMatcher(None, source.sequence, target.sequence, autojunk=False)
    pairs = [
        (block.a + offset, block.b + offset)
        for block in matcher.get_matching_blocks()
        for offset in range(block.size)
    ]
    source_indices = [left for left, _ in pairs]
    target_indices = [right for _, right in pairs]
    matches = sum(source.sequence[left] == target.sequence[right] for left, right in pairs)
    aligned = len(pairs)
    coverage = aligned / max(len(source.sequence), len(target.sequence), 1)
    identity = matches / max(aligned, 1)
    gap_fraction = 1 - coverage
    return {
        "source_indices": source_indices,
        "target_indices": target_indices,
        "coverage": coverage,
        "identity": identity,
        "gap_fraction": gap_fraction,
        "gaps": max(len(source.sequence), len(target.sequence)) - aligned,
    }


def kabsch(mobile: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return base.kabsch(mobile, reference)


def select_positions(
    target: DomainRecord,
    target_indices: Sequence[int],
    num_points: int,
) -> list[int]:
    target_to_aligned = {residue: position for position, residue in enumerate(target_indices)}
    mandatory_residues = set().union(*(region_indices(target, region) for region in REGIONS))
    mandatory_residues.update(
        index
        for index in (target.vaik_lys_index, target.alphac_glu_index)
        if index is not None
    )
    mandatory = sorted(
        target_to_aligned[index] for index in mandatory_residues if index in target_to_aligned
    )
    evenly_spaced = (
        torch.linspace(0, len(target_indices) - 1, min(num_points, len(target_indices)))
        .round()
        .long()
        .tolist()
    )
    selected = list(dict.fromkeys(mandatory + evenly_spaced))
    if len(selected) > num_points:
        fixed = mandatory[:num_points]
        selected = fixed + [
            value for value in selected if value not in set(fixed)
        ][: num_points - len(fixed)]
    if len(selected) < num_points:
        selected += [
            index for index in range(len(target_indices)) if index not in set(selected)
        ][: num_points - len(selected)]
    return sorted(selected[:num_points])


def prepare_pair(
    source: DomainRecord, target: DomainRecord, config: Config
) -> dict[str, Any] | None:
    alignment = align_domains(source, target)
    if (
        alignment["coverage"] < config.min_alignment_coverage
        or alignment["identity"] < config.min_sequence_identity
        or alignment["gap_fraction"] > config.max_gap_fraction
    ):
        return None
    source_indices = alignment["source_indices"]
    target_indices = alignment["target_indices"]
    selected = select_positions(target, target_indices, config.num_points)
    source_selected = [source_indices[index] for index in selected]
    target_selected = [target_indices[index] for index in selected]
    source_xyz = torch.tensor([source.coords[index] for index in source_selected], dtype=torch.float32)
    target_xyz = torch.tensor([target.coords[index] for index in target_selected], dtype=torch.float32)
    source_xyz -= source_xyz.mean(0, keepdim=True)
    target_xyz = kabsch(target_xyz, source_xyz)
    scale = source_xyz.square().sum(-1).mean().sqrt().clamp_min(1e-6)

    masks = {}
    for region in REGIONS:
        region_set = region_indices(target, region)
        masks[region] = torch.tensor([index in region_set for index in target_selected])
    functional = torch.stack(list(masks.values())).any(0)
    conserved = ~functional
    source_original = [source.original_indices[index] for index in source_selected]
    target_original = [target.original_indices[index] for index in target_selected]
    consecutive = torch.tensor(
        [
            source_original[i + 1] == source_original[i] + 1
            and target_original[i + 1] == target_original[i] + 1
            for i in range(len(source_original) - 1)
        ],
        dtype=torch.bool,
    )
    marker_masks = {}
    for name, index in {
        "lys": target.vaik_lys_index,
        "glu": target.alphac_glu_index,
        "dfg_anchor": target.dfg_index,
        "hrd_anchor": target.hrd_index,
    }.items():
        marker_masks[name] = torch.tensor([value == index for value in target_selected])

    length = len(source_xyz)
    padding = config.num_points - length
    valid = torch.ones(length, dtype=torch.bool)
    if padding:
        source_xyz = F.pad(source_xyz, (0, 0, 0, padding))
        target_xyz = F.pad(target_xyz, (0, 0, 0, padding))
        valid = F.pad(valid, (0, padding), value=False)
        consecutive = F.pad(consecutive, (0, max(0, config.num_points - 1 - len(consecutive))), value=False)
        masks = {name: F.pad(mask, (0, padding), value=False) for name, mask in masks.items()}
        marker_masks = {
            name: F.pad(mask, (0, padding), value=False) for name, mask in marker_masks.items()
        }
        source_original += [-1] * padding
        target_original += [-1] * padding
        source_selected += [-1] * padding
        target_selected += [-1] * padding

    return {
        "x0": source_xyz / scale,
        "x1": target_xyz / scale,
        "delta": (target_xyz - source_xyz) / scale,
        "scale": scale,
        "mask": valid,
        "consecutive_mask": consecutive[: config.num_points - 1],
        "functional_mask": functional if not padding else F.pad(functional, (0, padding), value=False),
        "conserved_mask": conserved if not padding else F.pad(conserved, (0, padding), value=False),
        **{f"{name}_mask": value for name, value in masks.items()},
        **{f"{name}_mask": value for name, value in marker_masks.items()},
        "source_original_indices": torch.tensor(source_original, dtype=torch.long),
        "target_original_indices": torch.tensor(target_original, dtype=torch.long),
        "source_domain_indices": torch.tensor(source_selected, dtype=torch.long),
        "target_domain_indices": torch.tensor(target_selected, dtype=torch.long),
        "coverage": alignment["coverage"],
        "identity": alignment["identity"],
        "gap_fraction": alignment["gap_fraction"],
        "gaps": alignment["gaps"],
    }


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    differences = coords[..., :, None, :] - coords[..., None, :, :]
    return torch.sqrt(differences.square().sum(-1) + 1e-12)


def masked_rmsd(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    squared = (left - right).square().sum(-1)
    return torch.sqrt((squared * mask).sum(-1) / mask.sum(-1).clamp_min(1))


def marked_distance(coords: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    rows = []
    for index in range(len(coords)):
        if left[index].any() and right[index].any():
            rows.append(
                torch.linalg.vector_norm(
                    coords[index][left[index]][0] - coords[index][right[index]][0]
                )
            )
        else:
            rows.append(coords.new_tensor(float("nan")))
    return torch.stack(rows)


def local_features(coords: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    rows = []
    for index in range(len(coords)):
        valid = batch["mask"][index].bool()
        xyz = coords[index][valid]
        dm = pairwise_distances(xyz)
        upper = dm[torch.triu_indices(len(xyz), len(xyz), 1, device=xyz.device).unbind()]
        values = [
            torch.sqrt(((xyz - xyz.mean(0)).square().sum(-1)).mean()),
            upper.mean(),
            upper.std(),
        ]
        for region in REGIONS:
            region_mask = batch[f"{region}_mask"][index][valid].bool()
            local = xyz[region_mask]
            if len(local) > 1:
                local_dm = pairwise_distances(local)
                local_upper = local_dm[
                    torch.triu_indices(len(local), len(local), 1, device=xyz.device).unbind()
                ]
                values.extend(
                    [
                        local_upper.mean(),
                        torch.sqrt(((local - local.mean(0)).square().sum(-1)).mean()),
                    ]
                )
            else:
                values.extend([xyz.new_tensor(0.0), xyz.new_tensor(0.0)])

        def distance(left: str, right: str) -> torch.Tensor:
            left_mask = batch[f"{left}_mask"][index][valid].bool()
            right_mask = batch[f"{right}_mask"][index][valid].bool()
            if left_mask.any() and right_mask.any():
                return torch.linalg.vector_norm(xyz[left_mask][0] - xyz[right_mask][0])
            return xyz.new_tensor(0.0)

        values.extend(
            [
                distance("hrd_anchor", "dfg_anchor"),
                distance("lys", "glu"),
                distance("lys", "dfg_anchor"),
            ]
        )
        rows.append(torch.stack(values))
    return torch.stack(rows)


def domain_feature_table(domains: Mapping[str, Sequence[DomainRecord]], config: Config):
    metadata, features = [], []
    for split, values in domains.items():
        for domain in values:
            prepared = prepare_pair(domain, domain, config)
            if prepared is None:
                continue
            batch = {
                key: value[None]
                for key, value in prepared.items()
                if torch.is_tensor(value) and key not in {"scale", "delta"}
            }
            feature = local_features(prepared["x0"][None], batch)[0].numpy()
            metadata.append(
                {
                    "split": split,
                    "pdb_id": domain.pdb_id,
                    "kinase": domain.kinase,
                    "state": domain.state,
                    "domain_length": len(domain.sequence),
                    "dfg_state": domain.structure.dfg_state,
                    "alphac_state": domain.structure.alphac_state,
                }
            )
            features.append(feature)
    return pd.DataFrame(metadata), np.stack(features)


def train_feature_classifier(metadata: pd.DataFrame, features: np.ndarray):
    train = metadata.split.eq("train").to_numpy()
    labels = metadata.state.eq("active").astype(int).to_numpy()
    scaler = StandardScaler().fit(features[train])
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=42
    )
    model.fit(scaler.transform(features[train]), labels[train])
    report = []
    for split in ("train", "val", "test"):
        selected = metadata.split.eq(split).to_numpy()
        probability = model.predict_proba(scaler.transform(features[selected]))[:, 1]
        prediction = probability >= 0.5
        y = labels[selected]
        report.append(
            {
                "split": split,
                "samples": int(selected.sum()),
                "balanced_accuracy": balanced_accuracy_score(y, prediction),
                "roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) == 2 else np.nan,
            }
        )
    importance = pd.DataFrame(
        {
            "feature": feature_names(),
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return (scaler, model), pd.DataFrame(report), importance


def feature_names() -> list[str]:
    names = ["radius_of_gyration", "mean_distance", "std_distance"]
    for region in REGIONS:
        names.extend([f"{region}_mean_distance", f"{region}_radius"])
    names.extend(["hrd_dfg_distance", "lys_glu_distance", "lys_dfg_distance"])
    return names


def feature_probability(classifier, features: np.ndarray) -> np.ndarray:
    scaler, model = classifier
    return model.predict_proba(scaler.transform(features))[:, 1]


def build_pairs(
    domains: Mapping[str, Sequence[DomainRecord]], config: Config
) -> tuple[pd.DataFrame, dict[str, list[tuple[DomainRecord, DomainRecord]]]]:
    rows = []
    selected: dict[str, list[tuple[DomainRecord, DomainRecord]]] = {
        "train": [], "val": [], "test": []
    }
    rng = random.Random(config.seed)
    for split, values in domains.items():
        grouped: dict[str, list[DomainRecord]] = defaultdict(list)
        for domain in values:
            grouped[domain.kinase].append(domain)
        for kinase, kinase_domains in sorted(grouped.items()):
            active = [value for value in kinase_domains if value.state == "active"]
            inactive = [value for value in kinase_domains if value.state == "inactive"]
            candidates = []
            for source_pool, target_pool in ((active, inactive), (inactive, active)):
                for source in source_pool:
                    for target in target_pool:
                        prepared = prepare_pair(source, target, config)
                        if prepared is None:
                            continue
                        valid = prepared["mask"]
                        source_xyz = prepared["x0"][valid]
                        target_xyz = prepared["x1"][valid]
                        global_rmsd = float(torch.sqrt(((source_xyz - target_xyz).square().sum(-1)).mean()))
                        local = {
                            region: float(
                                masked_rmsd(
                                    prepared["x0"][None],
                                    prepared["x1"][None],
                                    prepared[f"{region}_mask"][None],
                                )[0]
                            )
                            for region in REGIONS
                        }
                        quality = (
                            2 * prepared["coverage"]
                            + prepared["identity"]
                            - prepared["gap_fraction"]
                            - 0.05 * global_rmsd
                        )
                        candidates.append((quality, source, target, prepared, global_rmsd, local))
            candidates.sort(key=lambda value: value[0], reverse=True)
            # Preserve both directions while limiting combinatorial explosion.
            by_direction: dict[str, list[Any]] = defaultdict(list)
            for candidate in candidates:
                direction = f"{candidate[1].state}_to_{candidate[2].state}"
                by_direction[direction].append(candidate)
            chosen = []
            for direction_values in by_direction.values():
                rng.shuffle(direction_values)
                direction_values.sort(key=lambda value: value[0], reverse=True)
                chosen.extend(direction_values[: config.max_pairs_per_kinase])
            for quality, source, target, prepared, global_rmsd, local in chosen:
                selected[split].append((source, target))
                rows.append(
                    {
                        "split": split,
                        "source_pdb": source.pdb_id,
                        "weak_target_pdb": target.pdb_id,
                        "kinase": kinase,
                        "source_state": source.state,
                        "target_state": target.state,
                        "direction": f"{source.state}_to_{target.state}",
                        "alignment_coverage": prepared["coverage"],
                        "sequence_identity": prepared["identity"],
                        "gap_fraction": prepared["gap_fraction"],
                        "global_rmsd": global_rmsd * float(prepared["scale"]),
                        "dfg_rmsd": local["dfg"] * float(prepared["scale"]),
                        "alphac_rmsd": local["alphac"] * float(prepared["scale"]),
                        "activation_loop_rmsd": local["activation_loop"] * float(prepared["scale"]),
                        "quality_score": quality,
                        "ambiguous_below_1A": global_rmsd * float(prepared["scale"]) < config.ambiguous_rmsd,
                    }
                )
    return pd.DataFrame(rows), selected


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[tuple[DomainRecord, DomainRecord]],
        config: Config,
        kinase_vocab: Mapping[str, int],
    ):
        self.pairs = list(pairs)
        self.config = config
        self.kinase_vocab = kinase_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, target = self.pairs[index]
        prepared = prepare_pair(source, target, self.config)
        if prepared is None:
            raise RuntimeError(f"Invalid pair {source.pdb_id}->{target.pdb_id}")
        prepared.update(
            {
                "source_state": torch.tensor(STATES.index(source.state)),
                "target_state": torch.tensor(STATES.index(target.state)),
                "direction_id": torch.tensor(0 if source.state == "active" else 1),
                "kinase_id": torch.tensor(self.kinase_vocab.get(source.kinase, 0)),
                "target_dfg_id": torch.tensor(
                    {"in": 1, "out": 2, "out-like": 3}.get(target.structure.dfg_state, 0)
                ),
                "target_alphac_id": torch.tensor(
                    {"in": 1, "out": 2}.get(target.structure.alphac_state, 0)
                ),
                "source_pdb": source.pdb_id,
                "weak_target_pdb": target.pdb_id,
                "kinase": source.kinase,
                "source_state_name": source.state,
                "target_state_name": target.state,
                "direction": f"{source.state}_to_{target.state}",
            }
        )
        return prepared


class LocalDeltaFlow(nn.Module):
    def __init__(self, config: Config, n_kinases: int):
        super().__init__()
        self.config = config
        e, h = config.embedding_dim, config.hidden_dim
        self.target_state = nn.Embedding(3, e)
        self.kinase = nn.Embedding(n_kinases, e)
        self.direction = nn.Embedding(2, e)
        self.dfg = nn.Embedding(4, e)
        self.alphac = nn.Embedding(3, e)
        self.time = nn.Sequential(nn.Linear(3, e), nn.SiLU(), nn.Linear(e, e))
        self.region = nn.Embedding(16, e)
        self.input = nn.Linear(6 + 7 * e + 2, h)
        self.position = nn.Parameter(torch.randn(1, config.num_points, h) * 0.01)
        heads = next(value for value in (8, 4, 2, 1) if h % value == 0)
        layer = nn.TransformerEncoderLayer(
            h, heads, 4 * h, config.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, config.num_layers)
        self.output = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 3))

    def forward(
        self,
        delta_t: torch.Tensor,
        t: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        unconditional: bool = False,
        training_drop: bool = False,
    ) -> torch.Tensor:
        size = len(delta_t)
        if unconditional:
            keep = torch.zeros(size, 1, device=delta_t.device)
        elif training_drop:
            keep = (
                torch.rand(size, 1, device=delta_t.device) >= self.config.condition_dropout
            ).float()
        else:
            keep = torch.ones(size, 1, device=delta_t.device)
        time = t[:, None]
        time_embedding = self.time(
            torch.cat(
                [time, torch.sin(2 * math.pi * time), torch.cos(2 * math.pi * time)], -1
            )
        )
        context = torch.cat(
            [
                time_embedding,
                self.target_state(batch["target_state"]) * keep,
                self.kinase(batch["kinase_id"]) * keep,
                self.direction(batch["direction_id"]) * keep,
                self.dfg(batch["target_dfg_id"]) * keep,
                self.alphac(batch["target_alphac_id"]) * keep,
            ],
            -1,
        )
        role = sum(
            batch[f"{region}_mask"].long() << index
            for index, region in enumerate(REGIONS)
        )
        role_embedding = self.region(role) * keep[:, None]
        position = torch.linspace(0, 1, delta_t.shape[1], device=delta_t.device)
        position = torch.stack([position, torch.sin(2 * math.pi * position)], -1)
        inputs = torch.cat(
            [
                batch["x0"],
                delta_t,
                context[:, None].expand(-1, delta_t.shape[1], -1),
                role_embedding,
                position[None].expand(size, -1, -1),
            ],
            -1,
        )
        hidden = self.input(inputs)
        hidden = self.body(
            hidden + self.position[:, : delta_t.shape[1]],
            src_key_padding_mask=~batch["mask"],
        )
        velocity = self.output(hidden)
        # The model can move the conserved core slightly, but functional regions
        # receive full capacity while the rest is damped.
        mobility = 0.20 + 0.80 * batch["functional_mask"].float()
        return velocity * mobility[:, :, None] * batch["mask"][:, :, None]


def move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point() and value.dtype != torch.float32:
                value = value.float()
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def loss_components(
    model: LocalDeltaFlow,
    batch: Mapping[str, torch.Tensor],
    config: Config,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_delta = batch["delta"]
    t = torch.rand(len(target_delta), device=target_delta.device)
    noise = config.noise_std * torch.randn_like(target_delta)
    delta_t = t[:, None, None] * target_delta
    if training:
        delta_t = delta_t + noise * batch["functional_mask"][:, :, None]
    prediction = model(delta_t, t, batch, training_drop=training)
    mask = batch["mask"]
    functional = batch["functional_mask"] & mask
    conserved = batch["conserved_mask"] & mask
    denominator = mask.sum().clamp_min(1) * 3
    denoising = (((prediction - target_delta) ** 2) * mask[:, :, None]).sum() / denominator
    local = (
        ((prediction - target_delta) ** 2) * functional[:, :, None]
    ).sum() / (functional.sum().clamp_min(1) * 3)
    global_preservation = (
        prediction.square() * conserved[:, :, None]
    ).sum() / (conserved.sum().clamp_min(1) * 3)
    generated = batch["x0"] + prediction
    generated_dm = pairwise_distances(generated)
    target_dm = pairwise_distances(batch["x1"])
    pair_mask = mask[:, :, None] & mask[:, None, :]
    distance = (((generated_dm - target_dm) ** 2) * pair_mask).sum() / pair_mask.sum().clamp_min(1)
    generated_bonds = torch.linalg.vector_norm(generated[:, 1:] - generated[:, :-1], dim=-1)
    source_bonds = torch.linalg.vector_norm(batch["x0"][:, 1:] - batch["x0"][:, :-1], dim=-1)
    consecutive = batch["consecutive_mask"]
    bond = (
        ((generated_bonds - source_bonds) ** 2) * consecutive
    ).sum() / consecutive.sum().clamp_min(1)
    delta_step = prediction[:, 1:] - prediction[:, :-1]
    smoothness = (
        delta_step.square().sum(-1) * consecutive
    ).sum() / consecutive.sum().clamp_min(1)
    radius_generated = torch.sqrt(
        ((generated - generated.mean(1, keepdim=True)).square().sum(-1) * mask).sum(-1)
        / mask.sum(-1).clamp_min(1)
    )
    radius_source = torch.sqrt(
        ((batch["x0"] - batch["x0"].mean(1, keepdim=True)).square().sum(-1) * mask).sum(-1)
        / mask.sum(-1).clamp_min(1)
    )
    geometry = (radius_generated - radius_source).square().mean()
    total = (
        config.flow_weight * denoising
        + config.local_weight * local
        + config.global_weight * global_preservation
        + config.distance_weight * distance
        + config.bond_weight * bond
        + config.smoothness_weight * smoothness
        + config.geometry_weight * geometry
    )
    return total, {
        "denoising_loss": float(denoising.detach()),
        "local_transition_loss": float(local.detach()),
        "global_preservation_loss": float(global_preservation.detach()),
        "distance_matrix_loss": float(distance.detach()),
        "bond_validity_loss": float(bond.detach()),
        "smoothness_loss": float(smoothness.detach()),
        "geometry_penalty": float(geometry.detach()),
    }


@torch.no_grad()
def integrate(
    model: LocalDeltaFlow,
    batch: Mapping[str, torch.Tensor],
    guidance_scale: float,
    steps: int,
    save_times: Sequence[float] = (1.0,),
) -> dict[float, torch.Tensor]:
    delta = torch.zeros_like(batch["x0"])
    result = {0.0: batch["x0"].detach().cpu()}
    dt = 1.0 / steps
    for step in range(steps):
        t0_value, t1_value = step / steps, (step + 1) / steps
        t0 = torch.full((len(delta),), t0_value, device=delta.device)
        t1 = torch.full((len(delta),), t1_value, device=delta.device)

        def velocity(value, time):
            conditional = model(value, time, batch)
            if guidance_scale == 1:
                return conditional
            unconditional = model(value, time, batch, unconditional=True)
            return unconditional + guidance_scale * (conditional - unconditional)

        v0 = velocity(delta, t0)
        predictor = delta + dt * v0
        v1 = velocity(predictor, t1)
        delta = delta + 0.5 * dt * (v0 + v1)
        generated = batch["x0"] + delta
        for requested in save_times:
            if requested not in result and t1_value + 1e-9 >= requested:
                result[float(requested)] = generated.detach().cpu()
    result[1.0] = (batch["x0"] + delta).detach().cpu()
    return result


def corrected_bond_metrics(coords_angstrom: torch.Tensor, batch: Mapping[str, torch.Tensor]):
    distances = torch.linalg.vector_norm(coords_angstrom[:, 1:] - coords_angstrom[:, :-1], dim=-1)
    consecutive = batch["consecutive_mask"]
    denominator = consecutive.sum(-1).clamp_min(1)
    error = (distances - 3.8).abs()
    return {
        "consecutive_ca_mae": (error * consecutive).sum(-1) / denominator,
        "consecutive_ca_valid_fraction": (
            ((distances > 3.3) & (distances < 4.3)) * consecutive
        ).sum(-1) / denominator,
        "consecutive_ca_broken_fraction": (
            ((distances < 2.5) | (distances > 5.0)) * consecutive
        ).sum(-1) / denominator,
    }


def evaluate(
    coords: torch.Tensor,
    batch: Mapping[str, Any],
    classifier,
) -> pd.DataFrame:
    features = local_features(coords, batch).detach().cpu().numpy()
    source_features = local_features(batch["x0"], batch).detach().cpu().numpy()
    target_features = local_features(batch["x1"], batch).detach().cpu().numpy()
    active_probability = feature_probability(classifier, features)
    target_probability = np.where(
        np.asarray(batch["target_state_name"]) == "active",
        active_probability,
        1 - active_probability,
    )
    scale = batch["scale"]
    coords_angstrom = coords * scale[:, None, None]
    source_angstrom = batch["x0"] * scale[:, None, None]
    target_angstrom = batch["x1"] * scale[:, None, None]
    bonds = corrected_bond_metrics(coords_angstrom, batch)
    rows = []
    for index in range(len(coords)):
        valid = batch["mask"][index].bool()
        source = source_angstrom[index][valid]
        generated = coords_angstrom[index][valid]
        target = target_angstrom[index][valid]
        source_error = torch.linalg.vector_norm(generated - source, dim=-1)
        target_error = torch.linalg.vector_norm(generated - target, dim=-1)
        region_values = {}
        for region in REGIONS:
            region_mask = batch[f"{region}_mask"][index][valid].bool()
            region_values[f"{region}_source_generated_rmsd"] = (
                float(torch.sqrt(((generated[region_mask] - source[region_mask]).square().sum(-1)).mean()))
                if region_mask.any() else np.nan
            )
            region_values[f"{region}_generated_target_rmsd"] = (
                float(torch.sqrt(((generated[region_mask] - target[region_mask]).square().sum(-1)).mean()))
                if region_mask.any() else np.nan
            )
        source_distance = np.linalg.norm(source_features[index] - target_features[index])
        generated_distance = np.linalg.norm(features[index] - target_features[index])
        rows.append(
            {
                "source_pdb": batch["source_pdb"][index],
                "weak_target_pdb": batch["weak_target_pdb"][index],
                "kinase": batch["kinase"][index],
                "direction": batch["direction"][index],
                "target_probability": float(target_probability[index]),
                "classifier_correct": int(target_probability[index] >= 0.5),
                "rmsd_source_generated": float(
                    torch.sqrt(((generated - source).square().sum(-1)).mean())
                ),
                "rmsd_generated_weak_target": float(
                    torch.sqrt(((generated - target).square().sum(-1)).mean())
                ),
                "mean_residue_error_source": float(source_error.mean()),
                "mean_residue_error_target": float(target_error.mean()),
                "feature_progress_to_target": float(source_distance - generated_distance),
                "dfg_progress": float(
                    np.linalg.norm(source_features[index, 3:5] - target_features[index, 3:5])
                    - np.linalg.norm(features[index, 3:5] - target_features[index, 3:5])
                ),
                "alphac_progress": float(
                    np.linalg.norm(source_features[index, 5:7] - target_features[index, 5:7])
                    - np.linalg.norm(features[index, 5:7] - target_features[index, 5:7])
                ),
                "activation_loop_progress": float(
                    np.linalg.norm(source_features[index, 9:11] - target_features[index, 9:11])
                    - np.linalg.norm(features[index, 9:11] - target_features[index, 9:11])
                ),
                "global_conservation": float(
                    1 / (1 + torch.sqrt(((generated - source).square().sum(-1)).mean()))
                ),
                "consecutive_ca_mae": float(bonds["consecutive_ca_mae"][index]),
                "consecutive_ca_valid_fraction": float(
                    bonds["consecutive_ca_valid_fraction"][index]
                ),
                "consecutive_ca_broken_fraction": float(
                    bonds["consecutive_ca_broken_fraction"][index]
                ),
                **region_values,
            }
        )
    return pd.DataFrame(rows)


def baselines(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = batch["mask"][:, :, None]
    source, target = batch["x0"], batch["x1"]
    noise = source + 0.04 * torch.randn_like(source) * batch["functional_mask"][:, :, None]
    interpolation = source + 0.25 * (target - source)
    mean_delta = (target - source).mean(0, keepdim=True)
    target_mean = source + mean_delta.expand_as(source)
    return {
        "copy_source": source,
        "random_noise": noise * mask,
        "simple_interpolation": interpolation * mask,
        "weak_target_oracle": target * mask,
        "aligned_batch_target_mean": target_mean * mask,
    }


def independent_target_baselines(
    batch: Mapping[str, Any],
    domain_lookup: Mapping[str, DomainRecord],
    domains: Sequence[DomainRecord],
    config: Config,
) -> dict[str, torch.Tensor]:
    """Compute source-specific real-target baselines excluding the weak target."""
    nearest_rows, mean_rows = [], []
    for index in range(len(batch["x0"])):
        source = domain_lookup[batch["source_pdb"][index]]
        target_state = batch["target_state_name"][index]
        candidates = [
            domain
            for domain in domains
            if domain.kinase == source.kinase
            and domain.state == target_state
            and domain.pdb_id != batch["weak_target_pdb"][index]
        ]
        aligned = []
        for candidate in candidates:
            prepared = prepare_pair(source, candidate, config)
            if prepared is not None:
                aligned.append(prepared["x1"].to(batch["x0"].device))
        if aligned:
            stack = torch.stack(aligned)
            source_coords = batch["x0"][index][None]
            mask = batch["mask"][index][None]
            rmsd = masked_rmsd(
                stack,
                source_coords.expand_as(stack),
                mask.expand(len(stack), -1),
            )
            nearest_rows.append(stack[rmsd.argmin()])
            mean_rows.append(stack.mean(0))
        else:
            nearest_rows.append(batch["x1"][index])
            mean_rows.append(batch["x1"][index])
    mask = batch["mask"][:, :, None]
    return {
        "independent_nearest_real_target": torch.stack(nearest_rows) * mask,
        "aligned_target_state_mean": torch.stack(mean_rows) * mask,
    }


def add_mean_delta_baseline(
    frames: dict[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    mean_delta_by_kinase: Mapping[tuple[str, str], torch.Tensor],
) -> None:
    generated = []
    for index, (kinase, direction) in enumerate(zip(batch["kinase"], batch["direction"])):
        delta = mean_delta_by_kinase.get((kinase, direction))
        if delta is None:
            generated.append(batch["x0"][index])
        else:
            generated.append(batch["x0"][index] + delta.to(batch["x0"].device))
    frames["mean_delta_by_kinase"] = torch.stack(generated) * batch["mask"][:, :, None]


def mean_deltas(dataset: PairDataset) -> dict[tuple[str, str], torch.Tensor]:
    grouped: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for index in range(len(dataset)):
        item = dataset[index]
        grouped[(item["kinase"], item["direction"])].append(item["delta"])
    return {key: torch.stack(values).mean(0) for key, values in grouped.items()}


def save_config(config: Config, output: Path, device: str) -> None:
    json_dump(
        {
            **asdict(config),
            "device": device,
            "objective": "motif-anchored kinase-domain local delta generation",
            "generated_equation": "generated = source + predicted_delta",
            "exact_target_ground_truth": False,
        },
        output / "run_config.json",
    )
