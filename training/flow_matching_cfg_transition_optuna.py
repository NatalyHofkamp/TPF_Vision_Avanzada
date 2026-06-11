#!/usr/bin/env python3
"""Flow Matching + classifier-free guidance for kinase transitions.

This is an isolated experiment: it only reads the existing KLIFS artifacts and
writes into ``results/flow_matching_cfg_transition_optuna`` (or ``--output``).

Examples
--------
python training/flow_matching_cfg_transition_optuna.py audit
python training/flow_matching_cfg_transition_optuna.py all --trials 20 --epochs 80
python training/flow_matching_cfg_transition_optuna.py smoke
python training/flow_matching_cfg_transition_optuna.py bidirectional --epochs 80
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kinase_flow_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kinase_flow_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset


LOG = logging.getLogger("kinase_flow")
SPLITS = ("train", "val", "test")
NULL_TOKEN = "<NULL>"
UNK_TOKEN = "<UNK>"


@dataclass
class Config:
    seed: int = 42
    num_points: int = 256
    max_pairs_per_kinase: int = 256
    learning_rate: float = 3e-4
    batch_size: int = 16
    hidden_dim: int = 192
    embedding_dim: int = 32
    num_layers: int = 4
    dropout: float = 0.1
    condition_dropout: float = 0.15
    guidance_scale: float = 2.0
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    integration_steps: int = 32
    architecture: str = "transformer"
    epochs: int = 80
    patience: int = 12
    min_delta: float = 1e-5
    use_kinase: bool = True
    use_dfg: bool = True
    use_alphac: bool = True
    noise_std: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0


@dataclass(frozen=True)
class Pair:
    split: str
    kinase: str
    inactive_pdb: str
    active_pdb: str
    inactive_dfg: str
    active_dfg: str
    inactive_alphac: str
    active_alphac: str
    inactive_length: int
    active_length: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, default=str)


def configure_plotting(output: Path) -> None:
    cache = output / ".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def pyplot(output: Path):
    configure_plotting(output)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return plt, sns


def discover_project_root(start: Path) -> Path:
    """Find data artifacts instead of assuming the current working directory."""
    candidates = [start.resolve(), *start.resolve().parents, Path(__file__).resolve().parents[1]]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        required = [
            candidate / "data" / "metadata" / "kinase_labels.csv",
            candidate / "data" / "processed",
            candidate / "data" / "splits" / "train.csv",
            candidate / "data" / "splits" / "val.csv",
            candidate / "data" / "splits" / "test.csv",
        ]
        if all(path.exists() for path in required):
            return candidate
    raise FileNotFoundError(
        "Could not locate data/metadata/kinase_labels.csv, data/processed and "
        "data/splits/{train,val,test}.csv from the current path."
    )


class DataRepository:
    def __init__(self, root: Path):
        self.root = root
        self.metadata_path = root / "data" / "metadata" / "kinase_labels.csv"
        self.splits_dir = root / "data" / "splits"
        self.processed_dir = root / "data" / "processed"
        self._shape_cache: Dict[str, int] = {}

    def tensor_dir(self, pdb_id: str) -> Path:
        value = str(pdb_id)
        candidates = [
            self.processed_dir / value.lower(),
            self.processed_dir / value.upper(),
            self.processed_dir / value,
        ]
        for path in candidates:
            if path.is_dir():
                return path
        return candidates[0]

    def load_coords(self, pdb_id: str) -> torch.Tensor:
        path = self.tensor_dir(pdb_id) / "ca_coords.pt"
        coords = torch.load(path, map_location="cpu", weights_only=False).float()
        if coords.ndim != 2 or coords.shape[1] != 3 or not torch.isfinite(coords).all():
            raise ValueError(f"Invalid C-alpha coordinates at {path}: {tuple(coords.shape)}")
        return coords

    def length(self, pdb_id: str) -> int:
        key = str(pdb_id).lower()
        if key not in self._shape_cache:
            self._shape_cache[key] = int(self.load_coords(key).shape[0])
        return self._shape_cache[key]

    def load_splits(self) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for split in SPLITS:
            frame = pd.read_csv(self.splits_dir / f"{split}.csv")
            required = {
                "pdb_id",
                "kinase_name",
                "conformational_state",
                "dfg_state",
                "alphac_state",
            }
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{split}.csv is missing columns: {sorted(missing)}")
            result[split] = frame
        return result

    def audit(self) -> Dict[str, Any]:
        splits = self.load_splits()
        tensor_dirs = sorted(path for path in self.processed_dir.iterdir() if path.is_dir())
        lengths: List[int] = []
        invalid: List[Dict[str, Any]] = []
        missing_metadata = 0
        for directory in tensor_dirs:
            coords_path = directory / "ca_coords.pt"
            dist_path = directory / "distance_matrix.pt"
            meta_path = directory / "metadata.pt"
            if not meta_path.exists():
                missing_metadata += 1
            if not coords_path.exists() or not dist_path.exists():
                invalid.append({"pdb_id": directory.name, "reason": "missing tensor"})
                continue
            try:
                coords = torch.load(coords_path, map_location="cpu", weights_only=False)
                dist = torch.load(dist_path, map_location="cpu", weights_only=False)
                valid = (
                    coords.ndim == 2
                    and coords.shape[1] == 3
                    and dist.shape == (coords.shape[0], coords.shape[0])
                    and torch.isfinite(coords).all()
                    and torch.isfinite(dist).all()
                )
                if not valid:
                    invalid.append(
                        {
                            "pdb_id": directory.name,
                            "coords_shape": list(coords.shape),
                            "distance_shape": list(dist.shape),
                        }
                    )
                else:
                    lengths.append(int(coords.shape[0]))
            except Exception as exc:
                invalid.append({"pdb_id": directory.name, "reason": repr(exc)})

        split_report: Dict[str, Any] = {}
        split_pdbs: Dict[str, set[str]] = {}
        split_kinases: Dict[str, set[str]] = {}
        for name, frame in splits.items():
            ids = frame["pdb_id"].astype(str).str.lower()
            split_pdbs[name] = set(ids)
            split_kinases[name] = set(frame["kinase_name"].astype(str))
            states = frame["conformational_state"].astype(str).str.lower()
            dedup = frame.assign(_pdb=ids).drop_duplicates("_pdb")
            mixed = dedup.groupby("kinase_name")["conformational_state"].agg(
                lambda values: {"active", "inactive"} <= set(map(str.lower, map(str, values)))
            )
            split_report[name] = {
                "rows": len(frame),
                "unique_pdb": len(set(ids)),
                "duplicate_rows_by_pdb": int(ids.duplicated().sum()),
                "kinases": sorted(split_kinases[name]),
                "state_counts": states.value_counts(dropna=False).to_dict(),
                "kinases_with_both_states": int(mixed.sum()),
            }

        overlaps: Dict[str, Any] = {}
        for index, left in enumerate(SPLITS):
            for right in SPLITS[index + 1 :]:
                overlaps[f"{left}_{right}"] = {
                    "pdb_overlap": sorted(split_pdbs[left] & split_pdbs[right]),
                    "kinase_overlap": sorted(split_kinases[left] & split_kinases[right]),
                }
        return {
            "project_root": str(self.root),
            "metadata_csv": str(self.metadata_path),
            "splits_dir": str(self.splits_dir),
            "processed_dir": str(self.processed_dir),
            "metadata_rows": len(pd.read_csv(self.metadata_path)),
            "processed_directories": len(tensor_dirs),
            "valid_tensor_sets": len(lengths),
            "invalid_tensor_sets": invalid,
            "missing_metadata_pt": missing_metadata,
            "length_statistics": {
                "min": min(lengths) if lengths else None,
                "median": float(np.median(lengths)) if lengths else None,
                "max": max(lengths) if lengths else None,
                "unique_lengths": len(set(lengths)),
                "most_common": Counter(lengths).most_common(10),
            },
            "splits": split_report,
            "cross_split_overlaps": overlaps,
        }


def write_audit_report(repo: DataRepository, output: Path) -> Dict[str, Any]:
    report = repo.audit()
    output.mkdir(parents=True, exist_ok=True)
    json_dump(report, output / "data_structure_report.json")
    lines = [
        "# Data structure report",
        "",
        f"- Project root: `{report['project_root']}`",
        f"- Metadata: `{report['metadata_csv']}` ({report['metadata_rows']} rows)",
        f"- Processed tensor directories: {report['processed_directories']}",
        f"- Valid tensor sets: {report['valid_tensor_sets']}",
        f"- Invalid tensor sets: {len(report['invalid_tensor_sets'])}",
        f"- Missing metadata.pt: {report['missing_metadata_pt']}",
        f"- C-alpha lengths: {report['length_statistics']}",
        "",
        "## Splits",
        "",
    ]
    for split, info in report["splits"].items():
        lines.append(
            f"- {split}: {info['rows']} rows, {info['unique_pdb']} unique PDBs, "
            f"{info['duplicate_rows_by_pdb']} duplicate rows, "
            f"{len(info['kinases'])} kinases, states={info['state_counts']}"
        )
    lines.extend(
        [
            "",
            "## Leakage checks",
            "",
            f"```json\n{json.dumps(report['cross_split_overlaps'], indent=2)}\n```",
            "",
            "Variable lengths are handled by deterministic index-space linear resampling. "
            "The active structure is centered and Kabsch-aligned to the inactive structure. "
            "This is an approximation because the processed tensors do not contain residue IDs.",
        ]
    )
    (output / "data_structure_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return UNK_TOKEN
    text = str(value).strip()
    return text if text else UNK_TOKEN


def deduplicate_split(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_pdb_key"] = result["pdb_id"].astype(str).str.lower()
    if "resolution" in result:
        result["_resolution_sort"] = pd.to_numeric(result["resolution"], errors="coerce").fillna(np.inf)
        result = result.sort_values(["_pdb_key", "_resolution_sort"])
    return result.drop_duplicates("_pdb_key", keep="first").drop(
        columns=["_pdb_key", "_resolution_sort"], errors="ignore"
    )


def build_pairs(
    repo: DataRepository, max_pairs_per_kinase: int, seed: int
) -> Tuple[Dict[str, List[Pair]], pd.DataFrame]:
    """Pair active/inactive structures within kinase and split.

    A balanced deterministic cycle is used instead of the full Cartesian
    product, preventing large kinases from dominating while exposing every
    structure where possible.
    """
    rng = random.Random(seed)
    all_pairs: Dict[str, List[Pair]] = {}
    rows: List[Dict[str, Any]] = []
    for split, raw in repo.load_splits().items():
        frame = deduplicate_split(raw)
        frame["conformational_state"] = frame["conformational_state"].astype(str).str.lower()
        split_pairs: List[Pair] = []
        for kinase, group in frame.groupby("kinase_name", sort=True):
            inactive = group[group["conformational_state"] == "inactive"].to_dict("records")
            active = group[group["conformational_state"] == "active"].to_dict("records")
            rng.shuffle(inactive)
            rng.shuffle(active)
            possible = len(inactive) * len(active)
            target = min(max(len(inactive), len(active)), possible, max_pairs_per_kinase)
            generated: set[Tuple[str, str]] = set()
            cursor = 0
            while len(generated) < target and cursor < possible * 2 + 1:
                left = inactive[cursor % len(inactive)] if inactive else None
                right = active[(cursor + cursor // max(1, len(inactive))) % len(active)] if active else None
                cursor += 1
                if left is None or right is None:
                    break
                key = (str(left["pdb_id"]).lower(), str(right["pdb_id"]).lower())
                if key in generated:
                    continue
                generated.add(key)
                try:
                    pair = Pair(
                        split=split,
                        kinase=str(kinase),
                        inactive_pdb=str(left["pdb_id"]),
                        active_pdb=str(right["pdb_id"]),
                        inactive_dfg=normalize_value(left["dfg_state"]),
                        active_dfg=normalize_value(right["dfg_state"]),
                        inactive_alphac=normalize_value(left["alphac_state"]),
                        active_alphac=normalize_value(right["alphac_state"]),
                        inactive_length=repo.length(str(left["pdb_id"])),
                        active_length=repo.length(str(right["pdb_id"])),
                    )
                except (FileNotFoundError, ValueError) as exc:
                    LOG.warning("Skipping invalid pair %s: %s", key, exc)
                    continue
                split_pairs.append(pair)
            rows.append(
                {
                    "split": split,
                    "kinase": kinase,
                    "inactive_structures": len(inactive),
                    "active_structures": len(active),
                    "cartesian_pairs": possible,
                    "selected_pairs": len(generated),
                    "strategy": "balanced_cycle" if possible else "no_pair_available",
                }
            )
        all_pairs[split] = split_pairs
    summary = pd.DataFrame(rows)
    return all_pairs, summary


def make_vocab(train_pairs: Sequence[Pair]) -> Dict[str, Dict[str, int]]:
    def mapping(values: Iterable[str]) -> Dict[str, int]:
        vocab = {NULL_TOKEN: 0, UNK_TOKEN: 1}
        for value in sorted(set(values)):
            if value not in vocab:
                vocab[value] = len(vocab)
        return vocab

    return {
        "state": mapping(["inactive", "active"]),
        "kinase": mapping(pair.kinase for pair in train_pairs),
        "dfg": mapping(
            value for pair in train_pairs for value in (pair.inactive_dfg, pair.active_dfg)
        ),
        "alphac": mapping(
            value for pair in train_pairs for value in (pair.inactive_alphac, pair.active_alphac)
        ),
    }


def vocab_id(vocab: Mapping[str, int], value: str) -> int:
    return vocab.get(value, vocab[UNK_TOKEN])


def resample_coords(coords: torch.Tensor, num_points: int) -> torch.Tensor:
    if coords.shape[0] == num_points:
        return coords.clone()
    values = coords.T.unsqueeze(0)
    return F.interpolate(values, size=num_points, mode="linear", align_corners=True).squeeze(0).T


def center(coords: torch.Tensor) -> torch.Tensor:
    return coords - coords.mean(dim=0, keepdim=True)


def kabsch_align(mobile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mobile_centered = center(mobile)
    target_centered = center(target)
    covariance = mobile_centered.T @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return mobile_centered @ rotation


class PairDataset(Dataset):
    def __init__(
        self,
        repo: DataRepository,
        pairs: Sequence[Pair],
        vocab: Mapping[str, Mapping[str, int]],
        num_points: int,
        direction: str = "activation",
    ):
        self.repo = repo
        self.pairs = list(pairs)
        self.vocab = vocab
        self.num_points = num_points
        if direction not in {"activation", "deactivation"}:
            raise ValueError(f"Unknown transition direction: {direction}")
        self.direction = direction

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        inactive = center(resample_coords(self.repo.load_coords(pair.inactive_pdb), self.num_points))
        active = resample_coords(self.repo.load_coords(pair.active_pdb), self.num_points)
        active = kabsch_align(active, inactive)
        scale = inactive.square().sum(dim=-1).mean().sqrt().clamp_min(1e-6)
        inactive = inactive / scale
        active = active / scale
        if self.direction == "activation":
            x0, x1 = inactive, active
            source_state, target_state = "inactive", "active"
            target_dfg, target_alphac = pair.active_dfg, pair.active_alphac
        else:
            x0, x1 = active, inactive
            source_state, target_state = "active", "inactive"
            target_dfg, target_alphac = pair.inactive_dfg, pair.inactive_alphac
        return {
            "x0": x0,
            "x1": x1,
            "inactive": inactive,
            "active": active,
            "scale": scale,
            "state": torch.tensor(vocab_id(self.vocab["state"], target_state)),
            "kinase": torch.tensor(vocab_id(self.vocab["kinase"], pair.kinase)),
            "dfg": torch.tensor(vocab_id(self.vocab["dfg"], target_dfg)),
            "alphac": torch.tensor(vocab_id(self.vocab["alphac"], target_alphac)),
            "kinase_name": pair.kinase,
            "inactive_pdb": pair.inactive_pdb,
            "active_pdb": pair.active_pdb,
            "source_state": source_state,
            "target_state": target_state,
            "direction": self.direction,
        }


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FlowField(nn.Module):
    def __init__(self, config: Config, vocab: Mapping[str, Mapping[str, int]]):
        super().__init__()
        self.config = config
        embed = config.embedding_dim
        self.state_embedding = nn.Embedding(len(vocab["state"]), embed)
        self.kinase_embedding = nn.Embedding(len(vocab["kinase"]), embed)
        self.dfg_embedding = nn.Embedding(len(vocab["dfg"]), embed)
        self.alphac_embedding = nn.Embedding(len(vocab["alphac"]), embed)
        self.time_net = nn.Sequential(nn.Linear(3, embed), nn.SiLU(), nn.Linear(embed, embed))
        condition_dim = embed * (
            2 + int(config.use_kinase) + int(config.use_dfg) + int(config.use_alphac)
        )
        self.input_projection = nn.Linear(3 + condition_dim + 2, config.hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.num_points, config.hidden_dim) * 0.01
        )
        if config.architecture == "transformer":
            heads = next(
                head for head in (8, 4, 2, 1) if config.hidden_dim % head == 0
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.body: nn.Module = nn.TransformerEncoder(layer, config.num_layers)
        elif config.architecture == "conv":
            blocks = []
            for _ in range(config.num_layers):
                blocks.extend(
                    [
                        nn.Conv1d(config.hidden_dim, config.hidden_dim, 5, padding=2),
                        nn.SiLU(),
                        nn.Dropout(config.dropout),
                    ]
                )
            self.body = nn.Sequential(*blocks)
        elif config.architecture == "mlp":
            self.body = nn.Sequential(
                *[ResidualMLP(config.hidden_dim, config.dropout) for _ in range(config.num_layers)]
            )
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")
        self.output = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, 3))

    def condition(
        self, batch: Mapping[str, torch.Tensor], force_unconditional: bool, training_drop: bool
    ) -> torch.Tensor:
        keys = ["state"]
        if self.config.use_kinase:
            keys.append("kinase")
        if self.config.use_dfg:
            keys.append("dfg")
        if self.config.use_alphac:
            keys.append("alphac")
        embeddings = []
        batch_size = batch["state"].shape[0]
        if force_unconditional:
            keep = torch.zeros(batch_size, 1, device=batch["state"].device)
        elif training_drop and self.config.condition_dropout > 0:
            keep = (
                torch.rand(batch_size, 1, device=batch["state"].device)
                >= self.config.condition_dropout
            ).float()
        else:
            keep = torch.ones(batch_size, 1, device=batch["state"].device)
        tables = {
            "state": self.state_embedding,
            "kinase": self.kinase_embedding,
            "dfg": self.dfg_embedding,
            "alphac": self.alphac_embedding,
        }
        for key in keys:
            embeddings.append(tables[key](batch[key]) * keep)
        return torch.cat(embeddings, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        force_unconditional: bool = False,
        training_drop: bool = False,
    ) -> torch.Tensor:
        t = t.reshape(-1, 1)
        time_features = torch.cat(
            [t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1
        )
        time_embedding = self.time_net(time_features)
        condition = self.condition(batch, force_unconditional, training_drop)
        context = torch.cat([time_embedding, condition], dim=-1)
        context = context[:, None, :].expand(-1, x.shape[1], -1)
        positions = torch.linspace(0, 1, x.shape[1], device=x.device)
        positions = torch.stack([positions, torch.sin(2 * math.pi * positions)], dim=-1)
        positions = positions[None].expand(x.shape[0], -1, -1)
        hidden = self.input_projection(torch.cat([x, context, positions], dim=-1))
        hidden = hidden + self.position_embedding[:, : x.shape[1]]
        if self.config.architecture == "conv":
            hidden = self.body(hidden.transpose(1, 2)).transpose(1, 2)
        else:
            hidden = self.body(hidden)
        return self.output(hidden)


def move_batch(batch: Mapping[str, Any], device: str) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def flow_loss(model: FlowField, batch: Mapping[str, Any], training: bool) -> torch.Tensor:
    x0, x1 = batch["x0"], batch["x1"]
    t = torch.rand(x0.shape[0], device=x0.device)
    interpolation = t[:, None, None]
    xt = (1 - interpolation) * x0 + interpolation * x1
    if training and model.config.noise_std > 0:
        xt = xt + torch.randn_like(xt) * model.config.noise_std
    target_velocity = x1 - x0
    predicted = model(xt, t, batch, training_drop=training)
    return F.mse_loss(predicted, target_velocity)


@torch.no_grad()
def guided_velocity(
    model: FlowField,
    x: torch.Tensor,
    t: torch.Tensor,
    batch: Mapping[str, Any],
    guidance_scale: float,
) -> torch.Tensor:
    conditional = model(x, t, batch)
    if guidance_scale == 1.0:
        return conditional
    unconditional = model(x, t, batch, force_unconditional=True)
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.no_grad()
def integrate(
    model: FlowField,
    batch: Mapping[str, Any],
    steps: int,
    guidance_scale: float,
    save_times: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[float, torch.Tensor]:
    """Heun integration of dx/dt = v_theta(x,t,c)."""
    x = batch["x0"].clone()
    result = {0.0: x.detach().cpu()}
    requested = sorted(set(float(value) for value in save_times))
    dt = 1.0 / steps
    for step in range(steps):
        t0_value = step / steps
        t1_value = (step + 1) / steps
        t0 = torch.full((x.shape[0],), t0_value, device=x.device)
        t1 = torch.full((x.shape[0],), t1_value, device=x.device)
        velocity0 = guided_velocity(model, x, t0, batch, guidance_scale)
        predictor = x + dt * velocity0
        velocity1 = guided_velocity(model, predictor, t1, batch, guidance_scale)
        x = x + 0.5 * dt * (velocity0 + velocity1)
        for requested_time in requested:
            if requested_time not in result and t1_value + 1e-9 >= requested_time:
                result[requested_time] = x.detach().cpu()
    result[1.0] = x.detach().cpu()
    return result


def loader(
    dataset: Dataset, batch_size: int, shuffle: bool, config: Config
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_loss(model: FlowField, data: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for raw in data:
            batch = move_batch(raw, device)
            losses.append(float(flow_loss(model, batch, training=False)))
    return float(np.mean(losses)) if losses else float("inf")


def masked_rmsd(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return ((left - right).square().sum(dim=-1).mean(dim=-1)).sqrt()


def distance_error(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_dist = torch.cdist(left, left)
    right_dist = torch.cdist(right, right)
    return (left_dist - right_dist).abs().mean(dim=(1, 2))


@torch.no_grad()
def quick_rmsd(model: FlowField, data: DataLoader, device: str, max_batches: int = 2) -> float:
    model.eval()
    values = []
    for index, raw in enumerate(data):
        if index >= max_batches:
            break
        batch = move_batch(raw, device)
        generated = integrate(
            model,
            batch,
            model.config.integration_steps,
            model.config.guidance_scale,
            (1.0,),
        )[1.0]
        values.extend(masked_rmsd(generated, raw["x1"]).tolist())
    return float(np.mean(values)) if values else float("inf")


def make_scheduler(
    optimizer: torch.optim.Optimizer, name: str, epochs: int
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )
    if name == "none":
        return None
    raise ValueError(name)


def train_model(
    config: Config,
    repo: DataRepository,
    pairs: Mapping[str, Sequence[Pair]],
    vocab: Mapping[str, Mapping[str, int]],
    output: Path,
    run_name: str,
    trial: Any = None,
    direction: str = "activation",
) -> Tuple[FlowField, pd.DataFrame, float]:
    set_seed(config.seed)
    run_dir = output / "models" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data = PairDataset(
        repo, pairs["train"], vocab, config.num_points, direction=direction
    )
    val_data = PairDataset(
        repo, pairs["val"], vocab, config.num_points, direction=direction
    )
    if not train_data or not val_data:
        raise RuntimeError("Training and validation both require at least one active/inactive pair.")
    train_loader = loader(train_data, config.batch_size, True, config)
    val_loader = loader(val_data, config.batch_size, False, config)
    device = device_name()
    model = FlowField(config, vocab).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = make_scheduler(optimizer, config.scheduler, config.epochs)
    best = float("inf")
    stale = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for raw in train_loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_loss(model, batch, training=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(model, val_loader, device)
        epoch_rmsd = quick_rmsd(model, val_loader, device) if epoch == 1 or epoch % 5 == 0 else np.nan
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_rmsd": epoch_rmsd,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        LOG.info(
            "%s epoch=%d train=%.6f val=%.6f rmsd=%s",
            run_name,
            epoch,
            train_loss,
            val_loss,
            f"{epoch_rmsd:.4f}" if np.isfinite(epoch_rmsd) else "-",
        )
        checkpoint = {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "vocab": vocab,
            "epoch": epoch,
            "validation_loss": val_loss,
            "direction": direction,
        }
        torch.save(checkpoint, run_dir / "last_model.pt")
        if val_loss < best - config.min_delta:
            best = val_loss
            stale = 0
            torch.save(checkpoint, run_dir / "best_model.pt")
        else:
            stale += 1
        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()
        if stale >= config.patience:
            LOG.info("%s early stopping at epoch %d", run_name, epoch)
            break
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(run_dir / "training_history.csv", index=False)
    json_dump(asdict(config), run_dir / "config.json")
    best_checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    return model, history_frame, best


def load_model(checkpoint_path: Path, device: str) -> Tuple[FlowField, Config, Dict[str, Dict[str, int]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = Config(**checkpoint["config"])
    vocab = checkpoint["vocab"]
    model = FlowField(config, vocab).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, vocab


def train_plots(history: pd.DataFrame, output: Path, name: str) -> None:
    plt, _ = pyplot(output)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set(title="Flow matching loss", xlabel="Epoch", ylabel="MSE")
    axes[0].legend()
    observed = history.dropna(subset=["validation_rmsd"])
    axes[1].plot(observed["epoch"], observed["validation_rmsd"], marker="o")
    axes[1].set(title="Validation RMSD", xlabel="Epoch", ylabel="Normalized RMSD")
    figure.tight_layout()
    figure.savefig(output / "plots" / f"{name}_training.png", dpi=180)
    plt.close(figure)


def evaluate_model(
    model: FlowField,
    config: Config,
    dataset: PairDataset,
    output: Path,
    name: str,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    device = device_name()
    data = loader(dataset, config.batch_size, False, config)
    rows: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    seen = 0
    model.eval()
    for raw in data:
        batch = move_batch(raw, device)
        paths = integrate(
            model,
            batch,
            config.integration_steps,
            config.guidance_scale,
            (0.0, 0.25, 0.5, 0.75, 1.0),
        )
        generated = paths[1.0]
        x0 = raw["x0"]
        x1 = raw["x1"]
        inactive = raw["inactive"]
        active = raw["active"]
        scales = raw["scale"]
        generated_to_target = masked_rmsd(generated, x1) * scales
        initial_to_target = masked_rmsd(x0, x1) * scales
        generated_to_source = masked_rmsd(generated, x0) * scales
        generated_to_active = masked_rmsd(generated, active) * scales
        generated_to_inactive = masked_rmsd(generated, inactive) * scales
        dist_error = distance_error(generated, x1) * scales
        path_stack = torch.stack([paths[t] for t in sorted(paths)], dim=1)
        increments = path_stack[:, 1:] - path_stack[:, :-1]
        path_length = increments.square().sum(dim=-1).mean(dim=-1).sqrt().sum(dim=-1) * scales
        diversity = path_stack.std(dim=1).square().sum(dim=-1).mean(dim=-1).sqrt() * scales
        for index in range(generated.shape[0]):
            baseline = float(initial_to_target[index])
            final = float(generated_to_target[index])
            target_state = raw["target_state"][index]
            source_state = raw["source_state"][index]
            rows.append(
                {
                    "model": name,
                    "direction": raw["direction"][index],
                    "source_state": source_state,
                    "target_state": target_state,
                    "kinase": raw["kinase_name"][index],
                    "inactive_pdb": raw["inactive_pdb"][index],
                    "active_pdb": raw["active_pdb"][index],
                    "rmsd_generated_to_target": final,
                    "rmsd_initial_to_target": baseline,
                    "rmsd_generated_to_source": float(generated_to_source[index]),
                    "rmsd_generated_to_active": float(generated_to_active[index]),
                    "rmsd_generated_to_inactive": float(generated_to_inactive[index]),
                    "distance_matrix_mae": float(dist_error[index]),
                    "conformational_diversity": float(diversity[index]),
                    "path_length": float(path_length[index]),
                    "relative_improvement": (baseline - final) / max(baseline, 1e-8),
                    "closer_to_target_than_initial": final < baseline,
                    "closer_to_target_than_source": final
                    < float(generated_to_source[index]),
                    "closer_to_active": float(generated_to_active[index])
                    < float(generated_to_inactive[index]),
                    "closer_to_inactive": float(generated_to_inactive[index])
                    < float(generated_to_active[index]),
                }
            )
            if len(trajectories) < 20:
                trajectories.append(
                    {
                        "kinase": raw["kinase_name"][index],
                        "inactive_pdb": raw["inactive_pdb"][index],
                        "active_pdb": raw["active_pdb"][index],
                        "source_state": source_state,
                        "target_state": target_state,
                        "direction": raw["direction"][index],
                        "x0": x0[index].numpy(),
                        "x1": x1[index].numpy(),
                        "path": {time: value[index].numpy() for time, value in paths.items()},
                        "scale": float(scales[index]),
                    }
                )
            seen += 1
            if max_samples and seen >= max_samples:
                break
        if max_samples and seen >= max_samples:
            break
    frame = pd.DataFrame(rows)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / f"{name}_per_pair.csv", index=False)
    per_kinase = (
        frame.groupby("kinase")
        .agg(
            pairs=("rmsd_generated_to_target", "size"),
            rmsd=("rmsd_generated_to_target", "mean"),
            initial_rmsd=("rmsd_initial_to_target", "mean"),
            distance_error=("distance_matrix_mae", "mean"),
            diversity=("conformational_diversity", "mean"),
            improvement=("relative_improvement", "mean"),
            closer_to_target_rate=("closer_to_target_than_initial", "mean"),
            destination_crossing_rate=("closer_to_target_than_source", "mean"),
            closer_to_active_rate=("closer_to_active", "mean"),
            closer_to_inactive_rate=("closer_to_inactive", "mean"),
        )
        .reset_index()
    )
    per_kinase.to_csv(metrics_dir / f"{name}_per_kinase.csv", index=False)
    return frame, trajectories


def linear_baseline(dataset: PairDataset, output: Path) -> pd.DataFrame:
    """Oracle path baseline; endpoint uses the known target structure."""
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        x0, x1, scale = sample["x0"], sample["x1"], float(sample["scale"])
        path = torch.stack([(1 - t) * x0 + t * x1 for t in (0, 0.25, 0.5, 0.75, 1)])
        baseline = float(masked_rmsd(x0[None], x1[None])[0] * scale)
        rows.append(
            {
                "model": "linear_interpolation_oracle",
                "direction": sample["direction"],
                "source_state": sample["source_state"],
                "target_state": sample["target_state"],
                "kinase": sample["kinase_name"],
                "inactive_pdb": sample["inactive_pdb"],
                "active_pdb": sample["active_pdb"],
                "rmsd_generated_to_target": 0.0,
                "rmsd_initial_to_target": baseline,
                "rmsd_generated_to_source": baseline,
                "rmsd_generated_to_active": 0.0
                if sample["target_state"] == "active"
                else baseline,
                "rmsd_generated_to_inactive": 0.0
                if sample["target_state"] == "inactive"
                else baseline,
                "distance_matrix_mae": 0.0,
                "conformational_diversity": float(
                    path.std(dim=0).square().sum(dim=-1).mean().sqrt() * scale
                ),
                "path_length": baseline,
                "relative_improvement": 1.0,
                "closer_to_target_than_initial": True,
                "closer_to_target_than_source": True,
                "closer_to_active": sample["target_state"] == "active",
                "closer_to_inactive": sample["target_state"] == "inactive",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics" / "linear_interpolation_oracle_per_pair.csv", index=False)
    return frame


def evaluation_plots(frame: pd.DataFrame, trajectories: Sequence[Dict[str, Any]], output: Path, name: str) -> None:
    if frame.empty:
        return
    plt, sns = pyplot(output)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    target_state = str(frame["target_state"].iloc[0])
    sns.histplot(frame["rmsd_generated_to_target"], kde=True, ax=axes[0])
    axes[0].set_title(f"Generated-to-{target_state} RMSD")
    sns.boxplot(data=frame, x="kinase", y="rmsd_generated_to_target", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("RMSD by kinase")
    axes[2].scatter(
        frame["rmsd_initial_to_target"],
        frame["rmsd_generated_to_target"],
        alpha=0.7,
    )
    limit = max(frame["rmsd_initial_to_target"].max(), frame["rmsd_generated_to_target"].max())
    axes[2].plot([0, limit], [0, limit], "--", color="black")
    axes[2].set(
        xlabel=f"Initial-to-{target_state} RMSD",
        ylabel=f"Generated-to-{target_state} RMSD",
    )
    figure.tight_layout()
    figure.savefig(plots / f"{name}_metrics.png", dpi=180)
    plt.close(figure)
    if trajectories:
        plot_trajectory_projection(trajectories, output, name)
        plot_structural_trajectory(trajectories[0], output, name)


def trajectory_features(coords: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    indices = np.triu_indices(coords.shape[0], k=1)
    return distances[indices]


def plot_trajectory_projection(
    trajectories: Sequence[Dict[str, Any]], output: Path, name: str
) -> None:
    records: List[Dict[str, Any]] = []
    features: List[np.ndarray] = []
    for trajectory_index, item in enumerate(trajectories):
        features.extend([trajectory_features(item["x0"]), trajectory_features(item["x1"])])
        records.extend(
            [
                {
                    "kind": f"real_{item['source_state']}",
                    "time": 0.0,
                    "trajectory": trajectory_index,
                },
                {
                    "kind": f"real_{item['target_state']}",
                    "time": 1.0,
                    "trajectory": trajectory_index,
                },
            ]
        )
        for time in sorted(item["path"]):
            features.append(trajectory_features(item["path"][time]))
            records.append(
                {"kind": "generated", "time": float(time), "trajectory": trajectory_index}
            )
    matrix = np.stack(features)
    projection = PCA(n_components=2).fit_transform(matrix)
    metadata = pd.DataFrame(records)
    metadata["pc1"] = projection[:, 0]
    metadata["pc2"] = projection[:, 1]
    metadata.to_csv(output / "metrics" / f"{name}_pca_projection.csv", index=False)
    plt, sns = pyplot(output)
    figure, axis = plt.subplots(figsize=(9, 7))
    real = metadata[metadata["kind"] != "generated"]
    sns.scatterplot(data=real, x="pc1", y="pc2", hue="kind", style="kind", s=90, ax=axis)
    generated = metadata[metadata["kind"] == "generated"]
    for _, group in generated.groupby("trajectory"):
        group = group.sort_values("time")
        axis.plot(group["pc1"], group["pc2"], "-o", alpha=0.65, color="tab:purple")
        for left, right in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            axis.annotate(
                "",
                xy=(right.pc1, right.pc2),
                xytext=(left.pc1, left.pc2),
                arrowprops={"arrowstyle": "->", "color": "tab:purple", "alpha": 0.5},
            )
    source_state = trajectories[0]["source_state"]
    target_state = trajectories[0]["target_state"]
    axis.set_title(
        f"{source_state.title()} -> intermediate -> {target_state} conformational field (PCA)"
    )
    figure.tight_layout()
    figure.savefig(output / "plots" / f"{name}_conformational_field_pca.png", dpi=200)
    plt.close(figure)

    if len(matrix) >= 10:
        perplexity = min(30, max(2, len(matrix) // 4), len(matrix) - 1)
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(matrix)
        metadata["tsne1"], metadata["tsne2"] = tsne[:, 0], tsne[:, 1]
        figure, axis = plt.subplots(figsize=(9, 7))
        sns.scatterplot(
            data=metadata,
            x="tsne1",
            y="tsne2",
            hue="kind",
            style="kind",
            size="time",
            ax=axis,
        )
        axis.set_title("Conformational field (t-SNE)")
        figure.tight_layout()
        figure.savefig(output / "plots" / f"{name}_conformational_field_tsne.png", dpi=200)
        plt.close(figure)
    try:
        import umap

        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(matrix)
        metadata["umap1"], metadata["umap2"] = embedding[:, 0], embedding[:, 1]
        figure, axis = plt.subplots(figsize=(9, 7))
        sns.scatterplot(
            data=metadata, x="umap1", y="umap2", hue="kind", style="kind", size="time", ax=axis
        )
        axis.set_title("Conformational field (UMAP)")
        figure.tight_layout()
        figure.savefig(output / "plots" / f"{name}_conformational_field_umap.png", dpi=200)
        plt.close(figure)
    except Exception as exc:
        LOG.warning("UMAP unavailable (%s); PCA and t-SNE were generated.", exc)
    metadata.to_csv(output / "metrics" / f"{name}_manifold_projection.csv", index=False)


def plot_structural_trajectory(item: Dict[str, Any], output: Path, name: str) -> None:
    plt, _ = pyplot(output)
    times = sorted(item["path"])
    figure = plt.figure(figsize=(4 * len(times), 4))
    for index, time in enumerate(times, start=1):
        axis = figure.add_subplot(1, len(times), index, projection="3d")
        coords = item["path"][time] * item["scale"]
        axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=1.2)
        axis.set_title(f"t={time:.2f}")
        axis.set_axis_off()
    figure.suptitle(
        f"{item['kinase']}: {item['source_state']} -> {item['target_state']} "
        f"({item['active_pdb']} / {item['inactive_pdb']})"
    )
    figure.tight_layout()
    figure.savefig(output / "plots" / f"{name}_structural_trajectory.png", dpi=180)
    plt.close(figure)


def automatic_interpretation(frame: pd.DataFrame, output: Path, name: str) -> None:
    target_state = str(frame["target_state"].iloc[0])
    source_state = str(frame["source_state"].iloc[0])
    kinase = (
        frame.groupby("kinase")
        .agg(
            rmsd=("rmsd_generated_to_target", "mean"),
            improvement=("relative_improvement", "mean"),
            closer_rate=("closer_to_target_than_source", "mean"),
            pairs=("kinase", "size"),
        )
        .sort_values("rmsd")
    )
    best = kinase.head(3).index.tolist()
    worst = kinase.tail(3).index.tolist()
    mean_improvement = frame["relative_improvement"].mean()
    closer_rate = frame["closer_to_active"].mean()
    target_rate = frame["closer_to_target_than_initial"].mean()
    crossing_rate = frame["closer_to_target_than_source"].mean()
    coherent = mean_improvement > 0 and crossing_rate > 0.5
    text = f"""# Automatic biological interpretation: {name}

- Direction: {source_state} -> {target_state}.
- The learned endpoint improves over the starting structure in {100 * target_rate:.1f}% of pairs.
- It ends closer to the target than to the source in {100 * crossing_rate:.1f}% of pairs.
- It ends closer to the active state in {100 * closer_rate:.1f}% of pairs.
- Mean relative RMSD improvement is {100 * mean_improvement:.1f}%.
- Best kinases by generated-to-target RMSD: {', '.join(best) or 'none'}.
- Worst kinases by generated-to-target RMSD: {', '.join(worst) or 'none'}.

## Conclusion

The transition is {'consistent with a learned target-directed field' if coherent else 'not yet consistently target-directed'} under these structural metrics.
Intermediate states are geometrically continuous, but biological plausibility
cannot be established from C-alpha RMSD and distance matrices alone.

## Limitations and required validation

- Residues are matched by normalized chain index because residue identifiers and
  sequence alignments are absent from the processed tensors.
- Pairing crystal structures does not provide experimentally observed transition paths;
  intermediate-state supervision is unavailable.
- Ligand, mutation, missing-residue, crystallographic and protonation effects are not modeled.
- The linear baseline is an oracle geometric reference because it uses the real active endpoint.
- Validate Ramachandran geometry after all-atom reconstruction, steric clashes,
  conserved kinase motifs (DFG, HRD and alphaC-Glu/Lys), molecular-dynamics
  stability, free-energy profiles and agreement with experimental observables.

This approach can prioritize hypotheses about cancer-kinase activation, but it
does not by itself demonstrate a biological mechanism or a physically populated
transition pathway.
"""
    (output / f"biological_interpretation_{name}.md").write_text(text, encoding="utf-8")


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for optimization. Install with `python -m pip install optuna`."
        ) from exc
    return optuna


def optimize(
    base: Config,
    repo: DataRepository,
    pairs: Mapping[str, Sequence[Pair]],
    vocab: Mapping[str, Mapping[str, int]],
    output: Path,
    trials: int,
) -> Dict[str, Any]:
    optuna = require_optuna()
    storage = f"sqlite:///{(output / 'optuna_study.db').resolve()}"
    study = optuna.create_study(
        study_name="kinase_flow_matching_cfg",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base.seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    def objective(trial: Any) -> float:
        values = asdict(base)
        values.update(
            {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16, 32]),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 192, 256]),
                "embedding_dim": trial.suggest_categorical("embedding_dim", [16, 32, 64]),
                "num_layers": trial.suggest_int("num_layers", 2, 6),
                "dropout": trial.suggest_float("dropout", 0.0, 0.35),
                "guidance_scale": trial.suggest_float("guidance_scale", 0.5, 5.0),
                "condition_dropout": trial.suggest_float("condition_dropout", 0.05, 0.35),
                "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
                "scheduler": trial.suggest_categorical("scheduler", ["cosine", "plateau", "none"]),
                "integration_steps": trial.suggest_categorical(
                    "integration_steps", [8, 16, 32, 64]
                ),
                "architecture": trial.suggest_categorical(
                    "architecture", ["mlp", "conv", "transformer"]
                ),
            }
        )
        config = Config(**values)
        model, _, best = train_model(
            config, repo, pairs, vocab, output / "optuna_trials", f"trial_{trial.number}", trial
        )
        validation_data = PairDataset(repo, pairs["val"], vocab, config.num_points)
        validation_loader = loader(validation_data, config.batch_size, False, config)
        validation_rmsd = quick_rmsd(model, validation_loader, device_name(), max_batches=2)
        objective_value = best + 0.1 * validation_rmsd
        trial.set_user_attr("best_validation_loss", best)
        trial.set_user_attr("validation_rmsd", validation_rmsd)
        return objective_value

    study.optimize(objective, n_trials=trials, gc_after_trial=True)
    best = {"value": study.best_value, "params": study.best_params, "trial": study.best_trial.number}
    json_dump(best, output / "best_hyperparameters.json")
    best_trial_checkpoint = (
        output
        / "optuna_trials"
        / "models"
        / f"trial_{study.best_trial.number}"
        / "best_model.pt"
    )
    if best_trial_checkpoint.exists():
        shutil.copy2(best_trial_checkpoint, output / "best_optuna_checkpoint.pt")
    study.trials_dataframe().to_csv(output / "optuna_trials.csv", index=False)
    try:
        importance = optuna.importance.get_param_importances(study)
        json_dump(importance, output / "optuna_parameter_importance.json")
    except Exception as exc:
        LOG.warning("Could not compute parameter importance: %s", exc)
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )

    plot_functions = {
        "optimization_history": plot_optimization_history,
        "parameter_importance": plot_param_importances,
        "parallel_coordinates": plot_parallel_coordinate,
        "slice": plot_slice,
    }
    plots = output / "optuna_plots"
    plots.mkdir(parents=True, exist_ok=True)
    for name, plot_function in plot_functions.items():
        try:
            plot_function(study).write_html(plots / f"{name}.html")
        except Exception as exc:
            LOG.warning("Could not generate Optuna plot %s: %s", name, exc)
    return best


def config_with(base: Config, **changes: Any) -> Config:
    values = asdict(base)
    values.update(changes)
    return Config(**values)


def train_and_evaluate(
    config: Config,
    repo: DataRepository,
    pairs: Mapping[str, Sequence[Pair]],
    vocab: Mapping[str, Mapping[str, int]],
    output: Path,
    name: str,
    max_eval_samples: Optional[int] = None,
    direction: str = "activation",
    checkpoint_path: Optional[Path] = None,
) -> pd.DataFrame:
    if checkpoint_path is None:
        model, history, _ = train_model(
            config, repo, pairs, vocab, output, name, direction=direction
        )
        train_plots(history, output, name)
    else:
        model, config, checkpoint_vocab = load_model(checkpoint_path, device_name())
        vocab = checkpoint_vocab
    test_dataset = PairDataset(
        repo, pairs["test"], vocab, config.num_points, direction=direction
    )
    frame, trajectories = evaluate_model(
        model, config, test_dataset, output, name, max_eval_samples
    )
    evaluation_plots(frame, trajectories, output, name)
    automatic_interpretation(frame, output, name)
    return frame


def comparison(
    base: Config,
    repo: DataRepository,
    pairs: Mapping[str, Sequence[Pair]],
    vocab: Mapping[str, Mapping[str, int]],
    output: Path,
    optimized_params: Optional[Mapping[str, Any]],
    max_eval_samples: Optional[int],
) -> pd.DataFrame:
    test_dataset = PairDataset(repo, pairs["test"], vocab, base.num_points)
    frames = [linear_baseline(test_dataset, output)]
    variants = [
        ("flow_unconditional", config_with(base, use_kinase=False, use_dfg=False, use_alphac=False, condition_dropout=1.0, guidance_scale=0.0)),
        ("flow_conditioned_no_cfg", config_with(base, condition_dropout=0.0, guidance_scale=1.0)),
        ("flow_cfg", base),
    ]
    if optimized_params:
        variants.append(("flow_cfg_optuna", config_with(base, **dict(optimized_params))))
    for name, config in variants:
        frames.append(
            train_and_evaluate(
                config, repo, pairs, vocab, output, name, max_eval_samples
            )
        )
    all_metrics = pd.concat(frames, ignore_index=True)
    summary = (
        all_metrics.groupby("model")
        .agg(
            pairs=("rmsd_generated_to_target", "size"),
            rmsd_mean=("rmsd_generated_to_target", "mean"),
            rmsd_std=("rmsd_generated_to_target", "std"),
            distance_error=("distance_matrix_mae", "mean"),
            diversity=("conformational_diversity", "mean"),
            path_length=("path_length", "mean"),
            improvement=("relative_improvement", "mean"),
            closer_to_target_rate=("closer_to_target_than_initial", "mean"),
            destination_crossing_rate=("closer_to_target_than_source", "mean"),
        )
        .reset_index()
        .sort_values("rmsd_mean")
    )
    summary.to_csv(output / "comparison_table.csv", index=False)
    return summary


def bidirectional_summary(
    activation: pd.DataFrame,
    deactivation: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    combined = pd.concat([activation, deactivation], ignore_index=True)
    summary = (
        combined.groupby("direction")
        .agg(
            pairs=("rmsd_generated_to_target", "size"),
            final_rmsd=("rmsd_generated_to_target", "mean"),
            final_rmsd_std=("rmsd_generated_to_target", "std"),
            initial_target_rmsd=("rmsd_initial_to_target", "mean"),
            relative_improvement=("relative_improvement", "mean"),
            percent_closer_to_target=("closer_to_target_than_initial", "mean"),
            percent_crosses_destination=("closer_to_target_than_source", "mean"),
            distance_to_active=("rmsd_generated_to_active", "mean"),
            distance_to_inactive=("rmsd_generated_to_inactive", "mean"),
            path_length=("path_length", "mean"),
            distance_matrix_mae=("distance_matrix_mae", "mean"),
        )
        .reset_index()
    )
    labels = {
        "activation": "Inactiva -> Activa",
        "deactivation": "Activa -> Inactiva",
    }
    summary.insert(0, "Dirección", summary["direction"].map(labels))
    summary["% más cerca del target"] = 100 * summary["percent_closer_to_target"]
    summary["% cruza al estado destino"] = 100 * summary["percent_crosses_destination"]
    final_table = summary[
        [
            "Dirección",
            "final_rmsd",
            "relative_improvement",
            "% más cerca del target",
            "% cruza al estado destino",
        ]
    ].rename(
        columns={
            "final_rmsd": "RMSD final",
            "relative_improvement": "Mejora relativa",
        }
    )
    summary.to_csv(output / "bidirectional_detailed_comparison.csv", index=False)
    final_table.to_csv(output / "bidirectional_comparison_table.csv", index=False)

    per_kinase = (
        combined.groupby(["kinase", "direction"])
        .agg(
            pairs=("rmsd_generated_to_target", "size"),
            final_rmsd=("rmsd_generated_to_target", "mean"),
            relative_improvement=("relative_improvement", "mean"),
            closer_to_target_rate=("closer_to_target_than_initial", "mean"),
            destination_crossing_rate=("closer_to_target_than_source", "mean"),
        )
        .reset_index()
    )
    per_kinase.to_csv(output / "metrics" / "bidirectional_per_kinase.csv", index=False)
    write_bidirectional_interpretation(summary, per_kinase, output)
    plot_bidirectional_comparison(summary, per_kinase, output)
    return final_table


def write_bidirectional_interpretation(
    summary: pd.DataFrame, per_kinase: pd.DataFrame, output: Path
) -> None:
    indexed = summary.set_index("direction")
    activation = indexed.loc["activation"]
    deactivation = indexed.loc["deactivation"]
    both_fail = (
        activation["relative_improvement"] <= 0
        and deactivation["relative_improvement"] <= 0
        and activation["percent_crosses_destination"] <= 0.5
        and deactivation["percent_crosses_destination"] <= 0.5
    )
    if both_fail:
        learned_better = (
            "ninguna; ambas fallan el criterio de mejora y cruce al destino"
        )
        relative_winner = (
            "desactivación"
            if deactivation["relative_improvement"] > activation["relative_improvement"]
            else "activación"
        )
    elif deactivation["relative_improvement"] > activation["relative_improvement"]:
        learned_better = "desactivación (activa -> inactiva)"
        relative_winner = "desactivación"
    elif deactivation["relative_improvement"] < activation["relative_improvement"]:
        learned_better = "activación (inactiva -> activa)"
        relative_winner = "activación"
    else:
        learned_better = "ninguna dirección de forma diferenciable"
        relative_winner = "ninguna"
    deactivation_stable = (
        deactivation["final_rmsd_std"] < activation["final_rmsd_std"]
        and deactivation["path_length"] <= activation["path_length"]
    )
    deactivation_reaches_inactive = (
        deactivation["distance_to_inactive"] < deactivation["distance_to_active"]
    )
    kinase_pivot = per_kinase.pivot(
        index="kinase", columns="direction", values="relative_improvement"
    )
    inverse_better = []
    if {"activation", "deactivation"} <= set(kinase_pivot.columns):
        inverse_better = kinase_pivot[
            kinase_pivot["deactivation"] > kinase_pivot["activation"]
        ].index.tolist()
    text = f"""# Comparación automática de direcciones

## Respuesta breve

- El modelo aprende mejor: **{learned_better}**, usando mejora relativa como criterio.
- La dirección con resultado numérico menos desfavorable es: **{relative_winner}**.
- La transición activa -> inactiva es más estable: **{'sí' if deactivation_stable else 'no'}**.
- La generación inversa termina más cerca de la inactiva que de la activa:
  **{'sí' if deactivation_reaches_inactive else 'no'}**.
- Kinasas donde el camino inverso mejora más que el directo:
  **{', '.join(inverse_better) if inverse_better else 'ninguna en los resultados disponibles'}**.

## Activación: inactiva -> activa

- RMSD final al target: {activation['final_rmsd']:.3f} A.
- Mejora relativa: {100 * activation['relative_improvement']:.2f}%.
- Pares que mejoran respecto del inicio: {100 * activation['percent_closer_to_target']:.1f}%.
- Pares que terminan más cerca del target que del source:
  {100 * activation['percent_crosses_destination']:.1f}%.

## Desactivación: activa -> inactiva

- RMSD final al target: {deactivation['final_rmsd']:.3f} A.
- Mejora relativa: {100 * deactivation['relative_improvement']:.2f}%.
- Pares que mejoran respecto del inicio: {100 * deactivation['percent_closer_to_target']:.1f}%.
- Pares que terminan más cerca del target que del source:
  {100 * deactivation['percent_crosses_destination']:.1f}%.
- Distancia final a la activa: {deactivation['distance_to_active']:.3f} A.
- Distancia final a la inactiva: {deactivation['distance_to_inactive']:.3f} A.

## Interpretación biológica

Una asimetría entre direcciones puede reflejar diferencias del dataset, la
heterogeneidad de las conformaciones activas e inactivas o un sesgo del modelo;
no demuestra por sí sola que la activación o desactivación biológica sea más
fácil. Si la desactivación resulta más estable y cruza al destino con mayor
frecuencia, puede sugerir que el conjunto de estados inactivos es un atractor
estructural más compacto bajo esta representación. Esta hipótesis requiere
validación por motivos funcionales, reconstrucción all-atom, dinámica molecular
y perfiles de energía libre.
"""
    (output / "bidirectional_biological_interpretation.md").write_text(
        text, encoding="utf-8"
    )


def plot_bidirectional_comparison(
    summary: pd.DataFrame, per_kinase: pd.DataFrame, output: Path
) -> None:
    plt, sns = pyplot(output)
    labels = {
        "activation": "Inactiva -> Activa",
        "deactivation": "Activa -> Inactiva",
    }
    plot_data = summary.copy()
    plot_data["label"] = plot_data["direction"].map(labels)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.barplot(data=plot_data, x="label", y="final_rmsd", ax=axes[0])
    axes[0].set(title="RMSD final al target", xlabel="", ylabel="RMSD (A)")
    sns.barplot(data=plot_data, x="label", y="relative_improvement", ax=axes[1])
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set(title="Mejora relativa", xlabel="", ylabel="Fracción")
    melted = plot_data.melt(
        id_vars=["label"],
        value_vars=["percent_closer_to_target", "percent_crosses_destination"],
        var_name="criterion",
        value_name="rate",
    )
    melted["rate"] *= 100
    sns.barplot(data=melted, x="label", y="rate", hue="criterion", ax=axes[2])
    axes[2].set(title="Éxito por criterio", xlabel="", ylabel="Pares (%)", ylim=(0, 100))
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(output / "plots" / "bidirectional_comparison.png", dpi=200)
    plt.close(figure)

    if not per_kinase.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        kinase_plot = per_kinase.copy()
        kinase_plot["Dirección"] = kinase_plot["direction"].map(labels)
        sns.barplot(
            data=kinase_plot,
            x="kinase",
            y="relative_improvement",
            hue="Dirección",
            ax=axis,
        )
        axis.axhline(0, linestyle="--", color="black", linewidth=1)
        axis.set(title="Mejora relativa por kinasa", xlabel="Kinasa", ylabel="Mejora")
        figure.tight_layout()
        figure.savefig(output / "plots" / "bidirectional_per_kinase.png", dpi=200)
        plt.close(figure)


def run_bidirectional_experiment(
    config: Config,
    repo: DataRepository,
    pairs: Mapping[str, Sequence[Pair]],
    vocab: Mapping[str, Mapping[str, int]],
    output: Path,
    max_eval_samples: Optional[int],
    activation_checkpoint: Optional[Path] = None,
    deactivation_checkpoint: Optional[Path] = None,
) -> pd.DataFrame:
    activation = train_and_evaluate(
        config,
        repo,
        pairs,
        vocab,
        output,
        "bidirectional_activation",
        max_eval_samples,
        direction="activation",
        checkpoint_path=activation_checkpoint,
    )
    deactivation = train_and_evaluate(
        config,
        repo,
        pairs,
        vocab,
        output,
        "bidirectional_deactivation",
        max_eval_samples,
        direction="deactivation",
        checkpoint_path=deactivation_checkpoint,
    )
    return bidirectional_summary(activation, deactivation, output)


def prepare(args: argparse.Namespace) -> Tuple[Config, DataRepository, Dict[str, List[Pair]], Dict[str, Dict[str, int]], Path]:
    root = Path(args.root).resolve() if args.root else discover_project_root(Path.cwd())
    output = Path(args.output).resolve() if args.output else root / "results" / "flow_matching_cfg_transition_optuna"
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "metrics").mkdir(exist_ok=True)
    repo = DataRepository(root)
    write_audit_report(repo, output)
    config = Config(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        num_points=args.num_points,
        max_pairs_per_kinase=args.max_pairs_per_kinase,
        num_workers=args.num_workers,
    )
    pairs, pair_summary = build_pairs(repo, config.max_pairs_per_kinase, config.seed)
    pair_summary.to_csv(output / "pairing_report.csv", index=False)
    json_dump({split: len(values) for split, values in pairs.items()}, output / "pair_counts.json")
    for split in SPLITS:
        if not pairs[split]:
            raise RuntimeError(f"No inactive-active pairs could be built for split '{split}'.")
    vocab = make_vocab(pairs["train"])
    json_dump(vocab, output / "vocab.json")
    return config, repo, pairs, vocab, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "audit",
            "optimize",
            "train",
            "evaluate",
            "compare",
            "bidirectional",
            "all",
            "smoke",
            "smoke-bidirectional",
        ],
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--activation-checkpoint", type=str, default=None)
    parser.add_argument("--deactivation-checkpoint", type=str, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--max-pairs-per-kinase", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout
    )
    args = parse_args()
    if args.command in {"smoke", "smoke-bidirectional"}:
        args.epochs = min(args.epochs, 1)
        args.patience = 1
        args.num_points = min(args.num_points, 64)
        args.max_pairs_per_kinase = min(args.max_pairs_per_kinase, 8)
        args.max_eval_samples = min(args.max_eval_samples or 4, 4)
    config, repo, pairs, vocab, output = prepare(args)
    LOG.info("root=%s output=%s device=%s pairs=%s", repo.root, output, device_name(), {key: len(value) for key, value in pairs.items()})
    if args.command == "audit":
        return 0
    if args.command == "optimize":
        optimize(config, repo, pairs, vocab, output, args.trials)
        return 0
    if args.command == "train":
        train_and_evaluate(
            config, repo, pairs, vocab, output, "flow_cfg", args.max_eval_samples
        )
        return 0
    if args.command == "evaluate":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for evaluate")
        model, loaded_config, loaded_vocab = load_model(Path(args.checkpoint), device_name())
        dataset = PairDataset(repo, pairs["test"], loaded_vocab, loaded_config.num_points)
        frame, trajectories = evaluate_model(
            model, loaded_config, dataset, output, "checkpoint", args.max_eval_samples
        )
        evaluation_plots(frame, trajectories, output, "checkpoint")
        automatic_interpretation(frame, output, "checkpoint")
        return 0
    if args.command in {"bidirectional", "smoke-bidirectional"}:
        saved_best = output / "best_hyperparameters.json"
        if saved_best.exists() and args.command == "bidirectional":
            saved = json.loads(saved_best.read_text(encoding="utf-8"))
            config = config_with(config, **saved["params"])
        if args.command == "smoke-bidirectional":
            config = config_with(
                config,
                hidden_dim=64,
                embedding_dim=16,
                num_layers=2,
                architecture="mlp",
                batch_size=4,
                integration_steps=4,
            )
        activation_checkpoint = (
            Path(args.activation_checkpoint).resolve()
            if args.activation_checkpoint
            else None
        )
        deactivation_checkpoint = (
            Path(args.deactivation_checkpoint).resolve()
            if args.deactivation_checkpoint
            else None
        )
        summary = run_bidirectional_experiment(
            config,
            repo,
            pairs,
            vocab,
            output,
            args.max_eval_samples,
            activation_checkpoint,
            deactivation_checkpoint,
        )
        LOG.info("Bidirectional comparison:\n%s", summary.to_string(index=False))
        return 0
    best: Optional[Dict[str, Any]] = None
    if args.command == "all":
        best = optimize(config, repo, pairs, vocab, output, args.trials)
    elif args.command == "compare":
        saved_best = output / "best_hyperparameters.json"
        if saved_best.exists():
            best = json.loads(saved_best.read_text(encoding="utf-8"))
    optimized_params = best["params"] if best else None
    if args.command == "smoke":
        frame = train_and_evaluate(
            config_with(
                config,
                hidden_dim=64,
                embedding_dim=16,
                num_layers=2,
                architecture="mlp",
                batch_size=4,
                integration_steps=4,
            ),
            repo,
            pairs,
            vocab,
            output,
            "smoke_flow_cfg",
            args.max_eval_samples,
        )
        LOG.info("Smoke metrics: %s", frame.mean(numeric_only=True).to_dict())
        return 0
    summary = comparison(
        config, repo, pairs, vocab, output, optimized_params, args.max_eval_samples
    )
    LOG.info("Comparison:\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
