"""Distributional inverse-state generation for kinase C-alpha traces.

The experiment intentionally treats active/inactive structures as samples from
state distributions, not as unique paired endpoints. Pairs are weak transport
examples selected within a kinase after sequence alignment.
"""

from __future__ import annotations

import difflib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
STATES = ("inactive", "active")
MOTIFS = ("dfg", "alphac", "hrd", "activation_loop")


@dataclass
class Config:
    seed: int = 42
    epochs: int = 500
    patience: int = 30
    batch_size: int = 8
    num_points: int = 192
    hidden_dim: int = 160
    embedding_dim: int = 32
    num_layers: int = 4
    dropout: float = 0.10
    condition_dropout: float = 0.15
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    noise_std: float = 0.015
    guidance_scale: float = 1.5
    guidance_sweep: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    integration_steps: int = 32
    max_sources_per_kinase: int = 128
    max_eval_sources_per_kinase: int = 64
    min_alignment_coverage: float = 0.70
    min_sequence_identity: float = 0.80
    geometry_weight: float = 0.20
    core_preservation_weight: float = 0.10
    target_profile_weight: float = 0.10
    validation_every: int = 5
    num_workers: int = 0


@dataclass(frozen=True)
class Residue:
    number: int
    insertion: str
    aa: str
    ca: tuple[float, float, float]


@dataclass(frozen=True)
class Structure:
    split: str
    pdb_id: str
    kinase: str
    state: str
    dfg_state: str
    alphac_state: str
    ligand_present: int
    resolution: float
    chain: str
    residues: tuple[Residue, ...]
    sequence: str
    dfg_index: int | None
    hrd_index: int | None
    vaik_lys_index: int | None
    alphac_glu_index: int | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def parse_pdb(path: Path) -> dict[str, list[Residue]]:
    atoms: dict[tuple[str, int, str, str], tuple[float, float, float]] = {}
    order: list[tuple[str, int, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            if line[16:17] not in {" ", "A"}:
                continue
            name3 = line[17:20].strip().upper()
            if name3 not in AA3_TO_1:
                continue
            try:
                chain = line[21:22].strip() or "_"
                number = int(line[22:26])
                insertion = line[26:27].strip()
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            key = (chain, number, insertion, name3)
            if key not in atoms:
                atoms[key] = xyz
                order.append(key)
    chains: dict[str, list[Residue]] = defaultdict(list)
    for chain, number, insertion, name3 in order:
        chains[chain].append(
            Residue(number, insertion, AA3_TO_1[name3], atoms[(chain, number, insertion, name3)])
        )
    return dict(chains)


def motif_indices(sequence: str) -> dict[str, int | None]:
    dfg = re.search("DFG", sequence)
    hrd = re.search("HRD", sequence)
    vaik = list(re.finditer(r"[VIL][A-Z][IVL]K", sequence))
    lys = vaik[0].start() + 3 if vaik else None
    alphac = None
    if lys is not None:
        candidates = [i for i in range(max(0, lys - 40), max(0, lys - 5)) if sequence[i] == "E"]
        alphac = candidates[-1] if candidates else None
    return {
        "dfg": dfg.start() if dfg else None,
        "hrd": hrd.start() if hrd else None,
        "vaik_lys": lys,
        "alphac_glu": alphac,
    }


def choose_chain(chains: Mapping[str, Sequence[Residue]]) -> str | None:
    candidates = [
        (chain, residues)
        for chain, residues in chains.items()
        if "DFG" in "".join(residue.aa for residue in residues)
    ]
    candidates = candidates or list(chains.items())
    return max(candidates, key=lambda item: len(item[1]))[0] if candidates else None


def load_structures(root: Path) -> tuple[dict[str, list[Structure]], pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(root / "data/metadata/kinase_labels.csv")
    metadata["pdb_id"] = metadata.pdb_id.astype(str).str.lower()
    metadata["conformational_state"] = metadata.conformational_state.astype(str).str.lower()
    conflicts = metadata.groupby("pdb_id").conformational_state.nunique()
    conflict_ids = set(conflicts[conflicts > 1].index)
    clean = metadata[~metadata.pdb_id.isin(conflict_ids)].copy()
    clean["_resolution"] = pd.to_numeric(clean.resolution, errors="coerce").fillna(np.inf)
    clean = clean.sort_values(["pdb_id", "_resolution"]).drop_duplicates("pdb_id")

    split_by_kinase: dict[str, str] = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(root / "data/splits" / f"{split}.csv")
        for kinase in frame.kinase_name.astype(str).unique():
            split_by_kinase[kinase] = split

    records: dict[str, list[Structure]] = {"train": [], "val": [], "test": []}
    audit: list[dict[str, Any]] = []
    for row in clean.itertuples(index=False):
        split = split_by_kinase.get(str(row.kinase_name))
        path = root / "data/raw/pdbs" / f"{row.pdb_id}.pdb"
        reason = ""
        structure = None
        if split is None:
            reason = "missing_split"
        elif not path.exists():
            reason = "missing_pdb"
        else:
            chains = parse_pdb(path)
            chain = choose_chain(chains)
            if chain is None:
                reason = "no_ca_chain"
            else:
                residues = tuple(chains[chain])
                sequence = "".join(residue.aa for residue in residues)
                motifs = motif_indices(sequence)
                if len(residues) < 40:
                    reason = "too_short"
                else:
                    structure = Structure(
                        split=split,
                        pdb_id=row.pdb_id,
                        kinase=str(row.kinase_name),
                        state=str(row.conformational_state),
                        dfg_state=str(row.dfg_state).lower(),
                        alphac_state=str(row.alphac_state).lower(),
                        ligand_present=int(row.ligand_present),
                        resolution=float(row.resolution),
                        chain=chain,
                        residues=residues,
                        sequence=sequence,
                        dfg_index=motifs["dfg"],
                        hrd_index=motifs["hrd"],
                        vaik_lys_index=motifs["vaik_lys"],
                        alphac_glu_index=motifs["alphac_glu"],
                    )
                    records[split].append(structure)
        audit.append(
            {
                "pdb_id": row.pdb_id,
                "kinase": row.kinase_name,
                "state": row.conformational_state,
                "split": split,
                "valid": structure is not None,
                "reason": reason,
                "residues": len(structure.residues) if structure else np.nan,
                "dfg_found": structure.dfg_index is not None if structure else False,
                "hrd_found": structure.hrd_index is not None if structure else False,
                "vaik_lys_found": structure.vaik_lys_index is not None if structure else False,
                "alphac_glu_found": structure.alphac_glu_index is not None if structure else False,
            }
        )
    summary = pd.DataFrame(
        [{
            "raw_records": len(metadata),
            "raw_unique_pdb": metadata.pdb_id.nunique(),
            "conflicting_pdb_removed": len(conflict_ids),
            "valid_unique_structures": sum(map(len, records.values())),
            "active": sum(r.state == "active" for values in records.values() for r in values),
            "inactive": sum(r.state == "inactive" for values in records.values() for r in values),
            "kinases": len({r.kinase for values in records.values() for r in values}),
        }]
    )
    return records, pd.DataFrame(audit), summary


def align_sequences(source: Structure, target: Structure) -> tuple[list[int], list[int], float, float]:
    matcher = difflib.SequenceMatcher(None, source.sequence, target.sequence, autojunk=False)
    pairs = [
        (block.a + offset, block.b + offset)
        for block in matcher.get_matching_blocks()
        for offset in range(block.size)
    ]
    source_indices = [left for left, _ in pairs]
    target_indices = [right for _, right in pairs]
    matches = sum(source.sequence[left] == target.sequence[right] for left, right in pairs)
    coverage = len(pairs) / max(len(source.sequence), len(target.sequence), 1)
    identity = matches / max(len(pairs), 1)
    return source_indices, target_indices, coverage, identity


def kabsch(mobile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mobile = mobile - mobile.mean(0, keepdim=True)
    target = target - target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(mobile.T @ target)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return mobile @ rotation


def _region_indices(record: Structure, motif: str) -> set[int]:
    if motif == "dfg" and record.dfg_index is not None:
        return set(range(record.dfg_index - 2, record.dfg_index + 5))
    if motif == "alphac" and record.alphac_glu_index is not None:
        return set(range(record.alphac_glu_index - 7, record.alphac_glu_index + 8))
    if motif == "hrd" and record.hrd_index is not None:
        return set(range(record.hrd_index - 2, record.hrd_index + 5))
    if motif == "activation_loop" and record.dfg_index is not None:
        return set(range(record.dfg_index, record.dfg_index + 25))
    return set()


def prepare_pair(source: Structure, target: Structure, num_points: int) -> dict[str, Any] | None:
    source_indices, target_indices, coverage, identity = align_sequences(source, target)
    if not source_indices:
        return None
    source_xyz = torch.tensor([source.residues[i].ca for i in source_indices], dtype=torch.float32)
    target_xyz = torch.tensor([target.residues[i].ca for i in target_indices], dtype=torch.float32)
    source_xyz -= source_xyz.mean(0, keepdim=True)
    target_xyz = kabsch(target_xyz, source_xyz)
    target_to_aligned = {value: index for index, value in enumerate(target_indices)}
    mandatory_target = set().union(*(_region_indices(target, motif) for motif in MOTIFS))
    for index in (target.vaik_lys_index, target.alphac_glu_index):
        if index is not None:
            mandatory_target.add(index)
    mandatory = sorted(target_to_aligned[i] for i in mandatory_target if i in target_to_aligned)
    even = torch.linspace(0, len(source_xyz) - 1, min(num_points, len(source_xyz))).round().long().tolist()
    selected = list(dict.fromkeys(mandatory + even))
    if len(selected) > num_points:
        mandatory = mandatory[:num_points]
        selected = mandatory + [i for i in selected if i not in set(mandatory)][: num_points - len(mandatory)]
    if len(selected) < num_points:
        selected += [i for i in range(len(source_xyz)) if i not in set(selected)][: num_points - len(selected)]
    selected = sorted(selected[:num_points])
    source_xyz, target_xyz = source_xyz[selected], target_xyz[selected]
    selected_target_indices = [target_indices[i] for i in selected]
    masks = {}
    for motif in MOTIFS:
        region = _region_indices(target, motif)
        masks[motif] = torch.tensor([index in region for index in selected_target_indices])
    marker_masks = {}
    for name, residue_index in {
        "lys": target.vaik_lys_index,
        "glu": target.alphac_glu_index,
        "dfg_anchor": target.dfg_index,
        "hrd_anchor": target.hrd_index,
    }.items():
        marker_masks[name] = torch.tensor([index == residue_index for index in selected_target_indices])
    valid = torch.ones(len(source_xyz), dtype=torch.bool)
    padding = num_points - len(source_xyz)
    if padding:
        source_xyz = F.pad(source_xyz, (0, 0, 0, padding))
        target_xyz = F.pad(target_xyz, (0, 0, 0, padding))
        valid = F.pad(valid, (0, padding), value=False)
        masks = {name: F.pad(mask, (0, padding), value=False) for name, mask in masks.items()}
        marker_masks = {
            name: F.pad(mask, (0, padding), value=False) for name, mask in marker_masks.items()
        }
    scale = source_xyz[valid].square().sum(-1).mean().sqrt().clamp_min(1e-6)
    return {
        "x0": source_xyz / scale,
        "x1": target_xyz / scale,
        "scale": scale,
        "mask": valid,
        **{f"{name}_mask": value for name, value in masks.items()},
        **{f"{name}_mask": value for name, value in marker_masks.items()},
        "coverage": coverage,
        "identity": identity,
    }


def geometry_features(coords: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Differentiable local/global features for real and generated structures."""
    mask = batch["mask"].bool()
    rows = []
    for i in range(len(coords)):
        xyz = coords[i][mask[i]]
        dm = pairwise_distances(xyz)
        upper = dm[torch.triu_indices(len(xyz), len(xyz), 1, device=xyz.device).unbind()]
        center = xyz.mean(0)
        values = [
            torch.sqrt(((xyz - center).square().sum(-1)).mean()),
            upper.mean(),
            upper.std(),
            (upper < 0.45).float().mean(),
            (upper < 0.65).float().mean(),
        ]
        for motif in MOTIFS:
            local_mask = batch[f"{motif}_mask"][i][mask[i]].bool()
            local = xyz[local_mask]
            if len(local) > 1:
                local_dm = pairwise_distances(local)
                local_upper = local_dm[
                    torch.triu_indices(len(local), len(local), 1, device=xyz.device).unbind()
                ]
                values.extend([
                    local_upper.mean(),
                    torch.sqrt(((local - local.mean(0)).square().sum(-1)).mean()),
                ])
            else:
                values.extend([xyz.new_tensor(0.0), xyz.new_tensor(0.0)])

        def marked_distance(left: str, right: str) -> torch.Tensor:
            left_mask = batch[f"{left}_mask"][i][mask[i]].bool()
            right_mask = batch[f"{right}_mask"][i][mask[i]].bool()
            if left_mask.any() and right_mask.any():
                return torch.linalg.vector_norm(xyz[left_mask][0] - xyz[right_mask][0])
            return xyz.new_tensor(0.0)

        values.extend([
            marked_distance("lys", "glu"),
            marked_distance("hrd_anchor", "dfg_anchor"),
        ])
        rows.append(torch.stack(values))
    return torch.stack(rows)


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    differences = coords[:, None, :] - coords[None, :, :]
    return torch.sqrt(differences.square().sum(-1) + 1e-12)


def bond_validity(coords_angstrom: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    distances = torch.linalg.vector_norm(coords_angstrom[:, 1:] - coords_angstrom[:, :-1], dim=-1)
    deviation = (distances - 3.8).abs()
    denom = valid_pairs.sum(-1).clamp_min(1)
    mean_deviation = (deviation * valid_pairs).sum(-1) / denom
    valid_fraction = (((distances > 3.3) & (distances < 4.3)) * valid_pairs).sum(-1) / denom
    broken_fraction = (((distances < 2.5) | (distances > 5.0)) * valid_pairs).sum(-1) / denom
    return {
        "ca_bond_mean_abs_error": mean_deviation,
        "ca_bond_valid_fraction": valid_fraction,
        "broken_geometry_fraction": broken_fraction,
    }


class WeakDistributionDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Structure],
        config: Config,
        kinase_vocab: Mapping[str, int],
        profile_lookup: Mapping[tuple[str, str], np.ndarray],
        global_profiles: Mapping[str, np.ndarray],
        training: bool,
    ):
        by_kinase_state: dict[tuple[str, str], list[Structure]] = defaultdict(list)
        for record in records:
            by_kinase_state[(record.kinase, record.state)].append(record)
        self.sources = [
            record for record in records
            if by_kinase_state[(record.kinase, STATES[1 - STATES.index(record.state)])]
        ]
        if training:
            limited = []
            for _, values in _group(self.sources, lambda record: record.kinase):
                limited.extend(values[: config.max_sources_per_kinase])
            self.sources = limited
        self.pools = by_kinase_state
        self.config = config
        self.kinase_vocab = kinase_vocab
        self.profile_lookup = profile_lookup
        self.global_profiles = global_profiles
        self.epoch = 0
        self.training = training

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.sources[index]
        target_state = STATES[1 - STATES.index(source.state)]
        pool = self.pools[(source.kinase, target_state)]
        rng = random.Random(self.config.seed + 100003 * self.epoch + index)
        candidates = list(pool)
        rng.shuffle(candidates)
        prepared = None
        target = candidates[0]
        for candidate in candidates:
            item = prepare_pair(source, candidate, self.config.num_points)
            if item and item["coverage"] >= self.config.min_alignment_coverage and item["identity"] >= self.config.min_sequence_identity:
                prepared, target = item, candidate
                break
        if prepared is None:
            prepared = prepare_pair(source, target, self.config.num_points)
        if prepared is None:
            raise RuntimeError(f"Could not align {source.pdb_id} to an opposite-state structure")
        profile = self.profile_lookup.get((source.kinase, target_state), self.global_profiles[target_state])
        prepared.update(
            {
                "source_state": torch.tensor(STATES.index(source.state)),
                "target_state": torch.tensor(STATES.index(target_state)),
                "kinase": torch.tensor(self.kinase_vocab.get(source.kinase, 0)),
                "target_profile": torch.tensor(profile, dtype=torch.float32),
                "source_pdb": source.pdb_id,
                "weak_target_pdb": target.pdb_id,
                "kinase_name": source.kinase,
                "source_state_name": source.state,
                "target_state_name": target_state,
                "direction": f"{source.state}_to_{target_state}",
                "source_dfg_state": source.dfg_state,
                "source_alphac_state": source.alphac_state,
                "target_dfg_state": target.dfg_state,
                "target_alphac_state": target.alphac_state,
            }
        )
        return prepared


def _group(values: Iterable[Any], key):
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for value in values:
        grouped[key(value)].append(value)
    return grouped.items()


class FoldFlowCFG(nn.Module):
    def __init__(self, config: Config, n_kinases: int, feature_dim: int):
        super().__init__()
        self.config = config
        e, h = config.embedding_dim, config.hidden_dim
        self.source_state = nn.Embedding(3, e)
        self.target_state = nn.Embedding(3, e)
        self.kinase = nn.Embedding(n_kinases, e)
        self.profile = nn.Sequential(nn.Linear(feature_dim, e), nn.SiLU(), nn.Linear(e, e))
        self.time = nn.Sequential(nn.Linear(3, e), nn.SiLU(), nn.Linear(e, e))
        self.role = nn.Embedding(16, e)
        self.input = nn.Linear(3 + 6 * e + 2, h)
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
        x: torch.Tensor,
        t: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        unconditional: bool = False,
        training_drop: bool = False,
    ) -> torch.Tensor:
        size = len(x)
        if unconditional:
            keep = torch.zeros(size, 1, device=x.device)
        elif training_drop:
            keep = (torch.rand(size, 1, device=x.device) >= self.config.condition_dropout).float()
        else:
            keep = torch.ones(size, 1, device=x.device)
        time = t[:, None]
        time = self.time(torch.cat([time, torch.sin(2 * math.pi * time), torch.cos(2 * math.pi * time)], -1))
        context = torch.cat(
            [
                time,
                self.source_state(batch["source_state"]) * keep,
                self.target_state(batch["target_state"]) * keep,
                self.kinase(batch["kinase"]) * keep,
                self.profile(batch["target_profile"]) * keep,
            ],
            -1,
        )
        roles = sum(
            batch[f"{motif}_mask"].long() << index for index, motif in enumerate(MOTIFS)
        )
        role_embedding = self.role(roles) * keep[:, None]
        position = torch.linspace(0, 1, x.shape[1], device=x.device)
        position = torch.stack([position, torch.sin(2 * math.pi * position)], -1)
        hidden = self.input(
            torch.cat(
                [
                    x,
                    context[:, None].expand(-1, x.shape[1], -1),
                    role_embedding,
                    position[None].expand(size, -1, -1),
                ],
                -1,
            )
        )
        hidden = self.body(hidden + self.position[:, : x.shape[1]], src_key_padding_mask=~batch["mask"])
        return self.output(hidden) * batch["mask"][:, :, None]


def move_batch(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if value.is_floating_point() and value.dtype != torch.float32:
            value = value.float()
        moved[key] = value.to(device)
    return moved


def flow_objective(
    model: FoldFlowCFG,
    batch: Mapping[str, torch.Tensor],
    config: Config,
    training: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    x0, x1, mask = batch["x0"], batch["x1"], batch["mask"]
    t = torch.rand(len(x0), device=x0.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    if training:
        xt = xt + config.noise_std * torch.randn_like(xt) * mask[:, :, None]
    velocity = x1 - x0
    prediction = model(xt, t, batch, training_drop=training)
    denominator = mask.sum().clamp_min(1) * 3
    flow = (((prediction - velocity) ** 2) * mask[:, :, None]).sum() / denominator
    endpoint = xt + (1 - t[:, None, None]) * prediction
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    pred_bonds = torch.linalg.vector_norm(endpoint[:, 1:] - endpoint[:, :-1], dim=-1)
    target_bonds = torch.linalg.vector_norm(x1[:, 1:] - x1[:, :-1], dim=-1)
    geometry = (((pred_bonds - target_bonds) ** 2) * valid_pairs).sum() / valid_pairs.sum().clamp_min(1)
    functional = torch.stack([batch[f"{name}_mask"] for name in MOTIFS]).any(0)
    core = mask & ~functional
    core_pairs = core[:, :, None] & core[:, None, :]
    endpoint_dm = torch.sqrt(
        (endpoint[:, :, None, :] - endpoint[:, None, :, :]).square().sum(-1) + 1e-12
    )
    source_dm = torch.sqrt(
        (x0[:, :, None, :] - x0[:, None, :, :]).square().sum(-1) + 1e-12
    )
    core_preservation = (((endpoint_dm - source_dm) ** 2) * core_pairs).sum() / core_pairs.sum().clamp_min(1)
    features = geometry_features(endpoint, batch)
    profile = F.mse_loss(features, batch["target_profile"])
    loss = (
        flow
        + config.geometry_weight * geometry
        + config.core_preservation_weight * core_preservation
        + config.target_profile_weight * profile
    )
    return loss, {
        "flow_loss": float(flow.detach()),
        "geometry_loss": float(geometry.detach()),
        "core_preservation_loss": float(core_preservation.detach()),
        "target_profile_loss": float(profile.detach()),
    }


@torch.no_grad()
def integrate(
    model: FoldFlowCFG,
    batch: Mapping[str, torch.Tensor],
    guidance_scale: float,
    steps: int,
    save_times: Sequence[float] = (1.0,),
) -> dict[float, torch.Tensor]:
    x = batch["x0"].clone()
    result = {0.0: x.detach().cpu()}
    dt = 1.0 / steps
    for step in range(steps):
        t0_value, t1_value = step / steps, (step + 1) / steps
        t0 = torch.full((len(x),), t0_value, device=x.device)
        t1 = torch.full((len(x),), t1_value, device=x.device)

        def velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            conditional = model(value, time, batch)
            if guidance_scale == 1:
                return conditional
            unconditional = model(value, time, batch, unconditional=True)
            return unconditional + guidance_scale * (conditional - unconditional)

        v0 = velocity(x, t0)
        predictor = x + dt * v0
        v1 = velocity(predictor, t1)
        x = x + 0.5 * dt * (v0 + v1)
        x = x * batch["mask"][:, :, None]
        for requested in save_times:
            if requested not in result and t1_value + 1e-9 >= requested:
                result[float(requested)] = x.detach().cpu()
    result[1.0] = x.detach().cpu()
    return result


def build_real_feature_bank(
    records: Mapping[str, Sequence[Structure]],
    config: Config,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    rows, features = [], []
    banks: dict[str, dict[str, list[np.ndarray]]] = {
        split: {"active": [], "inactive": []} for split in records
    }
    for split, values in records.items():
        for record in values:
            prepared = prepare_pair(record, record, config.num_points)
            if prepared is None:
                continue
            batch = {
                key: value[None]
                for key, value in prepared.items()
                if torch.is_tensor(value) and key != "scale"
            }
            feature = geometry_features(prepared["x0"][None], batch)[0].numpy()
            rows.append(
                {
                    "split": split,
                    "pdb_id": record.pdb_id,
                    "kinase": record.kinase,
                    "state": record.state,
                    "dfg_state": record.dfg_state,
                    "alphac_state": record.alphac_state,
                }
            )
            features.append(feature)
            banks[split][record.state].append(feature)
    bank_arrays = {
        split: {
            state: np.stack(values) if values else np.empty((0, len(features[0])))
            for state, values in state_values.items()
        }
        for split, state_values in banks.items()
    }
    return pd.DataFrame(rows), np.stack(features), bank_arrays


def train_frozen_classifier(
    metadata: pd.DataFrame,
    features: np.ndarray,
) -> tuple[Any, pd.DataFrame]:
    train = metadata.split.eq("train").to_numpy()
    labels = metadata.state.eq("active").astype(int).to_numpy()
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
    )
    classifier.fit(features[train], labels[train])
    rows = []
    for split in ("train", "val", "test"):
        selected = metadata.split.eq(split).to_numpy()
        if not selected.any():
            continue
        probability = classifier.predict_proba(features[selected])[:, 1]
        prediction = probability >= 0.5
        y = labels[selected]
        rows.append(
            {
                "split": split,
                "samples": int(selected.sum()),
                "balanced_accuracy": balanced_accuracy_score(y, prediction),
                "roc_auc": roc_auc_score(y, probability) if len(np.unique(y)) == 2 else np.nan,
            }
        )
    return classifier, pd.DataFrame(rows)


def distribution_profiles(
    metadata: pd.DataFrame,
    features: np.ndarray,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    train = metadata[metadata.split.eq("train")].copy()
    train["_index"] = train.index
    by_kinase = {
        (kinase, state): np.median(features[group._index], axis=0)
        for (kinase, state), group in train.groupby(["kinase", "state"])
    }
    global_profiles = {
        state: np.median(features[train.loc[train.state.eq(state), "_index"]], axis=0)
        for state in STATES
    }
    return by_kinase, global_profiles


def nearest_feature_distance(
    generated: np.ndarray,
    bank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(bank) == 0:
        return np.full(len(generated), np.nan), np.full(len(generated), -1)
    distances = np.linalg.norm(generated[:, None] - bank[None], axis=-1)
    indices = distances.argmin(1)
    return distances[np.arange(len(generated)), indices], indices


def evaluate_coordinates(
    coords: torch.Tensor,
    batch: Mapping[str, Any],
    classifier: Any,
    target_bank: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    normalized_features = geometry_features(coords, batch).detach().cpu().numpy()
    source_features = geometry_features(batch["x0"], batch).detach().cpu().numpy()
    target_profiles = batch["target_profile"].detach().cpu().numpy()
    probability_active = classifier.predict_proba(normalized_features)[:, 1]
    target_probability = np.where(
        np.asarray(batch["target_state_name"]) == "active",
        probability_active,
        1 - probability_active,
    )
    coords_angstrom = coords * batch["scale"][:, None, None]
    source_angstrom = batch["x0"] * batch["scale"][:, None, None]
    mask = batch["mask"]
    validity = bond_validity(coords_angstrom, mask)
    rows = []
    for i in range(len(coords)):
        valid = mask[i].bool()
        generated = coords_angstrom[i][valid]
        source = source_angstrom[i][valid]
        weak_target = batch["x1"][i][valid] * batch["scale"][i]
        source_residue_error = torch.linalg.vector_norm(generated - source, dim=-1)
        target_residue_error = torch.linalg.vector_norm(generated - weak_target, dim=-1)
        global_rmsd = torch.sqrt(((generated - source).square().sum(-1)).mean())
        local_values = {}
        for motif in MOTIFS:
            local_mask = batch[f"{motif}_mask"][i][valid].bool()
            local_values[f"{motif}_source_generated_rmsd"] = (
                float(torch.sqrt(((generated[local_mask] - source[local_mask]).square().sum(-1)).mean()))
                if local_mask.any() else np.nan
            )
        bank = target_bank[batch["target_state_name"][i]]
        neighbor_distance, _ = nearest_feature_distance(normalized_features[i : i + 1], bank)
        scale = float(batch["scale"][i])
        dfg_target_progress = (
            np.linalg.norm(source_features[i, 5:7] - target_profiles[i, 5:7])
            - np.linalg.norm(normalized_features[i, 5:7] - target_profiles[i, 5:7])
        )
        alphac_target_progress = (
            np.linalg.norm(source_features[i, 7:9] - target_profiles[i, 7:9])
            - np.linalg.norm(normalized_features[i, 7:9] - target_profiles[i, 7:9])
        )
        rows.append(
            {
                "source_pdb": batch["source_pdb"][i],
                "weak_target_pdb": batch["weak_target_pdb"][i],
                "kinase": batch["kinase_name"][i],
                "direction": batch["direction"][i],
                "source_dfg_state": batch["source_dfg_state"][i],
                "source_alphac_state": batch["source_alphac_state"][i],
                "target_probability": float(target_probability[i]),
                "classifier_correct": int(target_probability[i] >= 0.5),
                "global_source_generated_rmsd": float(global_rmsd),
                "mean_residue_displacement": float(source_residue_error.mean()),
                "rmse_residue_displacement": float(
                    torch.sqrt(source_residue_error.square().mean())
                ),
                "weak_target_residue_mae": float(target_residue_error.mean()),
                "weak_target_residue_rmse": float(
                    torch.sqrt(target_residue_error.square().mean())
                ),
                "ca_bond_mean_abs_error": float(validity["ca_bond_mean_abs_error"][i]),
                "ca_bond_valid_fraction": float(validity["ca_bond_valid_fraction"][i]),
                "broken_geometry_fraction": float(validity["broken_geometry_fraction"][i]),
                "radius_of_gyration": float(normalized_features[i, 0] * batch["scale"][i]),
                "target_neighbor_feature_distance": float(neighbor_distance[0]),
                "dfg_target_progress": float(dfg_target_progress),
                "alphac_target_progress": float(alphac_target_progress),
                "lys_glu_distance_change": float(
                    (normalized_features[i, 13] - source_features[i, 13]) * scale
                ),
                "hrd_dfg_distance_change": float(
                    (normalized_features[i, 14] - source_features[i, 14]) * scale
                ),
                **local_values,
            }
        )
    return pd.DataFrame(rows)


def make_baselines(
    batch: Mapping[str, torch.Tensor],
    target_bank_coords: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    source, target, mask = batch["x0"], batch["x1"], batch["mask"][:, :, None]
    noise = source + 0.05 * torch.randn_like(source) * mask
    interpolation = (0.75 * source + 0.25 * target) * mask
    mean_target = target.mean(0, keepdim=True).expand_as(target) * mask
    return {
        "copy_source": source,
        "random_noise": noise,
        "nearest_target_state": target if target_bank_coords is None else target_bank_coords,
        "target_state_mean": mean_target,
        "simple_interpolation": interpolation,
    }


def summarize_test(details: pd.DataFrame) -> pd.DataFrame:
    return (
        details.groupby(["method", "direction"], dropna=False)
        .agg(
            n_samples=("source_pdb", "size"),
            target_state_probability=("target_probability", "mean"),
            frozen_classifier_accuracy=("classifier_correct", "mean"),
            dfg_change=("dfg_target_progress", "mean"),
            alphac_change=("alphac_target_progress", "mean"),
            hrd_dfg_change=("hrd_dfg_distance_change", "mean"),
            lys_glu_change=("lys_glu_distance_change", "mean"),
            activation_loop_change=("activation_loop_source_generated_rmsd", "mean"),
            rmsd_source_generated=("global_source_generated_rmsd", "mean"),
            mean_residue_error=("mean_residue_displacement", "mean"),
            residue_rmse=("rmse_residue_displacement", "mean"),
            weak_target_residue_mae=("weak_target_residue_mae", "mean"),
            weak_target_residue_rmse=("weak_target_residue_rmse", "mean"),
            geometric_validity=("ca_bond_valid_fraction", "mean"),
            broken_geometry=("broken_geometry_fraction", "mean"),
            nearest_real_target_distance=("target_neighbor_feature_distance", "mean"),
        )
        .reset_index()
    )


def automatic_conclusion(summary: pd.DataFrame) -> str:
    model = summary[summary.method.eq("foldflow_cfg")].set_index("direction")
    copy = summary[summary.method.eq("copy_source")].set_index("direction")
    noise = summary[summary.method.eq("random_noise")].set_index("direction")
    lines = ["# Conclusión automática", ""]
    passed = []
    for direction, row in model.iterrows():
        copy_row = copy.loc[direction]
        noise_row = noise.loc[direction]
        state_gain = row.target_state_probability - copy_row.target_state_probability
        distribution_gain = copy_row.nearest_real_target_distance - row.nearest_real_target_distance
        geometry_ok = row.geometric_validity >= 0.85 and row.broken_geometry <= 0.05
        beats_noise = row.nearest_real_target_distance < noise_row.nearest_real_target_distance
        residue_error_ok = row.weak_target_residue_mae < noise_row.weak_target_residue_mae
        local_change = row.activation_loop_change > 0.25 or row.dfg_change > 0.20 or row.alphac_change > 0.20
        success = (
            state_gain >= 0.10
            and distribution_gain > 0
            and geometry_ok
            and beats_noise
            and residue_error_ok
            and local_change
        )
        passed.append(success)
        lines.extend(
            [
                f"## {direction}",
                f"- Ganancia de probabilidad del estado objetivo vs copiar source: {state_gain:.3f}.",
                f"- Mejora de distancia a la distribución objetivo: {distribution_gain:.3f}.",
                f"- Validez geométrica: {row.geometric_validity:.3f}; geometría rota: {row.broken_geometry:.3f}.",
                f"- MAE por residuo vs referencia débil: {row.weak_target_residue_mae:.3f}; "
                f"mejor que ruido: {'sí' if residue_error_ok else 'no'}.",
                f"- Cambio local funcional suficiente: {'sí' if local_change else 'no'}.",
                f"- Diagnóstico: {'evidencia favorable' if success else 'evidencia insuficiente'}.",
                "",
            ]
        )
    if passed and all(passed):
        verdict = (
            "El modelo cumple los criterios operativos de plausibilidad en ambas direcciones. "
            "Esto apoya generación de hipótesis conformacionales, no reconstrucción de un PDB único."
        )
    else:
        verdict = (
            "El modelo no cumple todos los criterios de plausibilidad y superioridad frente a baselines. "
            "No debe declararse exitoso solo por la predicción del clasificador congelado."
        )
    lines.extend(["## Veredicto", verdict])
    return "\n".join(lines) + "\n"


def save_config(config: Config, output: Path, device: str) -> None:
    json_dump(
        {
            **asdict(config),
            "device": device,
            "objective": "plausible inverse-state distribution generation",
            "representation": "sequence-aligned motif-preserving C-alpha trace",
            "exact_target_reconstruction": False,
        },
        output / "run_config.json",
    )
