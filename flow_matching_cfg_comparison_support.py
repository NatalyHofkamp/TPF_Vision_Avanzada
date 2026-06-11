"""Flow Matching + CFG comparison experiment for kinase conformations.

This module is designed to be executed from a notebook wrapper and to run
gracefully even when the full dataset is not present in the workspace.
When the KLIFS metadata, splits, and processed tensors are available, it:

- reuses the same train/val/test splits,
- trains a binary baseline and a Flow Matching + CFG model,
- evaluates them with shared metrics,
- analyzes inactive -> active transitions,
- writes plots and a summary report under results/flow_matching_cfg_comparison/.
"""

from __future__ import annotations

import os
import json
import math
import random
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
DATA_DIR = ROOT / "data"
METADATA_PATH = DATA_DIR / "metadata" / "kinase_labels.csv"
SPLIT_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results" / "flow_matching_cfg_comparison"
PLOTS_DIR = RESULTS_DIR / "plots"
SUMMARY_PATH = RESULTS_DIR / "summary.md"

SEED = 42
BATCH_SIZE = 16
EPOCHS_BASELINE = 3
EPOCHS_CFG = 3
LEARNING_RATE = 1e-3
CONDITION_DROPOUT_PROB = 0.1
GUIDANCE_SCALES = [0.0, 1.0, 2.0, 3.0, 5.0]
SAMPLE_STEPS = 20
MAX_GENERATION_SAMPLES = 64
CONTACT_CUTOFF = 8.0
KLIFS_DOWNLOAD_LIMIT = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
LOGGER = logging.getLogger(__name__)


def plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return plt, sns


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def data_ready() -> bool:
    required = [
        METADATA_PATH,
        SPLIT_DIR / "train.csv",
        SPLIT_DIR / "val.csv",
        SPLIT_DIR / "test.csv",
        PROCESSED_DIR,
    ]
    if not all(file_exists(p) or p.is_dir() for p in required):
        return False
    if not any(PROCESSED_DIR.glob("*")):
        return False
    return True


def _extract_ca_coordinates_from_pdb(pdb_path: Path) -> np.ndarray:
    coords: List[List[float]] = []
    if not pdb_path.exists():
        return np.empty((0, 3), dtype=np.float32)
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coords.append([x, y, z])
    return np.asarray(coords, dtype=np.float32)


class SimpleProteinPreprocessor:
    """Minimal PDB-to-tensor preprocessor that does not require BioPython."""

    def __init__(self, metadata_csv: Path, output_dir: Path):
        self.metadata_csv = Path(metadata_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, save_tensors: bool = True):
        df = pd.read_csv(self.metadata_csv)
        items = []
        for _, row in df.iterrows():
            pdb_path = Path(row["filepath"])
            coords = _extract_ca_coordinates_from_pdb(pdb_path)
            if coords.size == 0:
                continue
            dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).astype(np.float32)
            item = {
                "pdb_id": str(row["pdb_id"]),
                "kinase_name": str(row["kinase_name"]),
                "ca_coords": torch.from_numpy(coords),
                "distance_matrix": torch.from_numpy(dist),
                "sequence_length": int(coords.shape[0]),
                "resolution": float(row["resolution"]) if pd.notna(row["resolution"]) else 0.0,
                "label": int(str(row["conformational_state"]) == "active"),
            }
            items.append(item)
            if save_tensors:
                tensor_dir = self.output_dir / str(row["pdb_id"]).lower()
                tensor_dir.mkdir(parents=True, exist_ok=True)
                torch.save(item["ca_coords"], tensor_dir / "ca_coords.pt")
                torch.save(item["distance_matrix"], tensor_dir / "distance_matrix.pt")
                torch.save(
                    {
                        "label": item["label"],
                        "sequence_length": item["sequence_length"],
                        "resolution": item["resolution"],
                        "kinase_name": item["kinase_name"],
                    },
                    tensor_dir / "metadata.pt",
                )
        return items


def prepare_data_from_api(notes: List[str]) -> None:
    """Download and preprocess the KLIFS dataset if local artifacts are missing."""
    from scripts.download_klifs_dataset import KLIFSDownloader

    try:
        from scripts.preprocess_dataset import ProteinPreprocessor as ScriptProteinPreprocessor
        preprocessor_cls = ScriptProteinPreprocessor
    except Exception:
        preprocessor_cls = SimpleProteinPreprocessor

    notes.append("Local KLIFS artifacts were missing, so the notebook will fetch the dataset from the KLIFS API and preprocess it.")
    LOGGER.info("Preparing KLIFS dataset from API")
    downloader = KLIFSDownloader(
        output_dir=DATA_DIR / "raw" / "pdbs",
        metadata_dir=DATA_DIR / "metadata",
    )
    LOGGER.info("Downloading metadata and structures from KLIFS")
    metadata_df = downloader.download_all()
    LOGGER.info("Saving metadata and generating split files")
    downloader.save_metadata(metadata_df, "kinase_labels.csv")
    downloader.create_train_val_test_splits(metadata_df)

    LOGGER.info("Preprocessing downloaded structures into tensors")
    preprocessor = preprocessor_cls(
        metadata_csv=METADATA_PATH,
        output_dir=PROCESSED_DIR,
    )
    preprocessor.run(save_tensors=True)
    LOGGER.info("KLIFS preprocessing completed")


def load_metadata_and_splits() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    metadata_df = pd.read_csv(METADATA_PATH)
    splits = {
        split: pd.read_csv(SPLIT_DIR / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    return metadata_df, splits


def build_vocab(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    def make_map(values: Iterable[str]) -> Dict[str, int]:
        unique = sorted({str(v).strip() for v in values if pd.notna(v)})
        mapping = {"<NULL>": 0}
        for i, value in enumerate(unique, start=1):
            mapping[value] = i
        return mapping

    return {
        "kinase": make_map(df["kinase_name"].astype(str)),
        "state": make_map(df["conformational_state"].astype(str)),
        "dfg": make_map(df["dfg_state"].astype(str)),
        "alphac": make_map(df["alphac_state"].astype(str)),
    }


def safe_index(mapping: Dict[str, int], value: object) -> int:
    return mapping.get(str(value).strip(), 0)


def kabsch_align(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align x onto y using Kabsch. x, y: (N, 3)."""
    x_center = x.mean(axis=0, keepdims=True)
    y_center = y.mean(axis=0, keepdims=True)
    x0 = x - x_center
    y0 = y - y_center
    h = x0.T @ y0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    x_aligned = x0 @ r + y_center
    return x_aligned, y


def rmsd_aligned(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape or x.size == 0:
        return float("nan")
    x_aligned, y_ref = kabsch_align(x, y)
    return float(np.sqrt(np.mean(np.sum((x_aligned - y_ref) ** 2, axis=-1))))


def contact_map(coords: np.ndarray, cutoff: float = CONTACT_CUTOFF) -> np.ndarray:
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return (d < cutoff).astype(np.float32)


def pairwise_rmsd(coords_list: Sequence[np.ndarray]) -> float:
    if len(coords_list) < 2:
        return 0.0
    vals = []
    for i in range(len(coords_list)):
        for j in range(i + 1, len(coords_list)):
            a = coords_list[i]
            b = coords_list[j]
            n = min(len(a), len(b))
            vals.append(rmsd_aligned(a[:n], b[:n]))
    return float(np.nanmean(vals)) if vals else 0.0


def approx_contacts_similarity(a: np.ndarray, b: np.ndarray, cutoff: float = CONTACT_CUTOFF) -> float:
    n = min(len(a), len(b))
    ca = contact_map(a[:n], cutoff=cutoff)
    cb = contact_map(b[:n], cutoff=cutoff)
    upper = np.triu_indices(n, k=1)
    aa = ca[upper]
    bb = cb[upper]
    intersection = float(np.sum((aa > 0.5) & (bb > 0.5)))
    union = float(np.sum((aa > 0.5) | (bb > 0.5)))
    return intersection / union if union > 0 else 1.0


def structure_summary_features(coords: np.ndarray, dist: np.ndarray) -> np.ndarray:
    finite = np.isfinite(dist) & (dist > 0)
    d = dist[finite]
    if d.size == 0:
        d = np.array([0.0], dtype=np.float32)
    centered = coords - coords.mean(axis=0, keepdims=True)
    rg = np.sqrt(np.mean(np.sum(centered**2, axis=-1)))
    contact_density = float(np.mean(d < CONTACT_CUTOFF))
    return np.array(
        [
            coords.shape[0],
            float(rg),
            float(d.mean()),
            float(d.std()),
            float(d.min()),
            float(d.max()),
            contact_density,
        ],
        dtype=np.float32,
    )


class ProteinTensorDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, processed_dir: Path, vocab: Dict[str, Dict[str, int]]):
        self.split_df = split_df.reset_index(drop=True)
        self.processed_dir = processed_dir
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.split_df.iloc[idx]
        tensor_dir = self.processed_dir / str(row["pdb_id"]).lower()
        if not tensor_dir.exists():
            tensor_dir = self.processed_dir / str(row["pdb_id"])
        coords = torch.load(tensor_dir / "ca_coords.pt", map_location="cpu").float()
        dist = torch.load(tensor_dir / "distance_matrix.pt", map_location="cpu").float()
        meta = torch.load(tensor_dir / "metadata.pt", map_location="cpu")
        return {
            "pdb_id": row["pdb_id"],
            "kinase_name": str(row["kinase_name"]),
            "conformational_state": str(row["conformational_state"]),
            "dfg_state": str(row["dfg_state"]),
            "alphac_state": str(row["alphac_state"]),
            "label": int(row["conformational_state"] == "active"),
            "coords": coords,
            "dist": dist,
            "metadata": meta,
            "kinase_idx": safe_index(self.vocab["kinase"], row["kinase_name"]),
            "state_idx": safe_index(self.vocab["state"], row["conformational_state"]),
            "dfg_idx": safe_index(self.vocab["dfg"], row["dfg_state"]),
            "alphac_idx": safe_index(self.vocab["alphac"], row["alphac_state"]),
        }


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    coords = [item["coords"] for item in batch]
    dists = [item["dist"] for item in batch]
    lengths = torch.tensor([c.shape[0] for c in coords], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded_coords = torch.zeros(len(batch), max_len, 3, dtype=coords[0].dtype)
    padded_dists = torch.zeros(len(batch), max_len, max_len, dtype=dists[0].dtype)
    for i, (coord, dist) in enumerate(zip(coords, dists)):
        n = coord.shape[0]
        padded_coords[i, :n, :] = coord
        padded_dists[i, :n, :n] = dist

    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    return {
        "pdb_id": [item["pdb_id"] for item in batch],
        "kinase_name": [item["kinase_name"] for item in batch],
        "conformational_state": [item["conformational_state"] for item in batch],
        "dfg_state": [item["dfg_state"] for item in batch],
        "alphac_state": [item["alphac_state"] for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "kinase_idx": torch.tensor([item["kinase_idx"] for item in batch], dtype=torch.long),
        "state_idx": torch.tensor([item["state_idx"] for item in batch], dtype=torch.long),
        "dfg_idx": torch.tensor([item["dfg_idx"] for item in batch], dtype=torch.long),
        "alphac_idx": torch.tensor([item["alphac_idx"] for item in batch], dtype=torch.long),
        "coords": padded_coords,
        "dist": padded_dists,
        "lengths": lengths,
        "mask": mask,
    }


class ConditionEncoder(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], embed_dim: int = 64, cond_dim: int = 128):
        super().__init__()
        self.kinase_embedding = nn.Embedding(vocab_sizes["kinase"], embed_dim, padding_idx=0)
        self.state_embedding = nn.Embedding(vocab_sizes["state"], embed_dim, padding_idx=0)
        self.dfg_embedding = nn.Embedding(vocab_sizes["dfg"], embed_dim, padding_idx=0)
        self.alphac_embedding = nn.Embedding(vocab_sizes["alphac"], embed_dim, padding_idx=0)
        self.null_embedding = nn.Parameter(torch.zeros(embed_dim * 4))
        self.project = nn.Sequential(
            nn.Linear(embed_dim * 4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        dropout_prob: float = 0.0,
        force_null: bool = False,
    ) -> torch.Tensor:
        bsz = batch["kinase_idx"].shape[0]
        device = batch["kinase_idx"].device
        if force_null:
            raw = self.null_embedding[None, :].expand(bsz, -1).to(device)
            return self.project(raw)

        kinase = batch["kinase_idx"].clone()
        state = batch["state_idx"].clone()
        dfg = batch["dfg_idx"].clone()
        alphac = batch["alphac_idx"].clone()

        if self.training and dropout_prob > 0:
            drop_mask = torch.rand(bsz, device=device) < dropout_prob
            kinase[drop_mask] = 0
            state[drop_mask] = 0
            dfg[drop_mask] = 0
            alphac[drop_mask] = 0

        emb = torch.cat(
            [
                self.kinase_embedding(kinase),
                self.state_embedding(state),
                self.dfg_embedding(dfg),
                self.alphac_embedding(alphac),
            ],
            dim=-1,
        )
        return self.project(emb)


class BinaryConditionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, cond_dim: int = 128):
        super().__init__()
        self.kinase_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.state_embedding = nn.Embedding(3, embed_dim, padding_idx=0)
        self.project = nn.Sequential(
            nn.Linear(embed_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.LayerNorm(cond_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor], dropout_prob: float = 0.0, force_null: bool = False) -> torch.Tensor:
        kinase = batch["kinase_idx"].clone()
        state = batch["state_idx"].clone()
        if force_null:
            kinase.zero_()
            state.zero_()
        elif self.training and dropout_prob > 0:
            drop_mask = torch.rand_like(kinase.float()) < dropout_prob
            kinase[drop_mask] = 0
            state[drop_mask] = 0
        emb = torch.cat([self.kinase_embedding(kinase), self.state_embedding(state)], dim=-1)
        return self.project(emb)


class FlowMatchingModel(nn.Module):
    def __init__(self, cond_dim: int = 128, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_embed = nn.Linear(3, hidden_dim)
        self.cond_embed = nn.Linear(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(depth)
            ]
        )
        self.out = nn.Linear(hidden_dim, 3)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition_embedding: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        if t.ndim == 2:
            t = t[:, :, None]
        h = self.coord_embed(x_t) + self.time_embed(t.squeeze(-1))[:, None, :] + self.cond_embed(condition_embedding)[:, None, :]
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask[..., None].float()
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0) * pred.shape[-1]
    return diff.sum() / denom


@torch.no_grad()
def euler_sample(
    model: FlowMatchingModel,
    encoder: nn.Module,
    condition_batch: Dict[str, torch.Tensor],
    num_steps: int = SAMPLE_STEPS,
    guidance_scale: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    model.eval()
    encoder.eval()
    device = device or next(model.parameters()).device
    bsz = condition_batch["kinase_idx"].shape[0]
    max_len = int(condition_batch["mask"].shape[1])
    x = torch.randn(bsz, max_len, 3, device=device)
    mask = condition_batch["mask"].to(device)
    cond = {k: v.to(device) for k, v in condition_batch.items() if torch.is_tensor(v)}
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((bsz, 1), step / num_steps, device=device)
        if guidance_scale == 0.0:
            cond_emb = encoder(cond, force_null=False)
            v = model(x, t, cond_emb)
        else:
            cond_emb = encoder(cond, force_null=False)
            null_emb = encoder(cond, force_null=True)
            v_uncond = model(x, t, null_emb)
            v_cond = model(x, t, cond_emb)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        x = x + dt * v
        x = x * mask[..., None].float()
    return x


def train_flow_matching(
    model: FlowMatchingModel,
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    dropout_prob: float,
    device: torch.device,
    use_cfg: bool,
) -> Dict[str, List[float]]:
    model.to(device)
    encoder.to(device)
    model.train()
    encoder.train()
    params = list(model.parameters()) + list(encoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        LOGGER.info("Starting epoch %d/%d", len(history["train_loss"]) + 1, epochs)
        model.train()
        encoder.train()
        train_losses = []
        for batch in train_loader:
            coords = batch["coords"].to(device)
            mask = batch["mask"].to(device)
            cond_batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            x1 = coords
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], 1, device=device)
            xt = (1.0 - t[:, None, :]) * x0 + t[:, None, :] * x1
            target_v = x1 - x0
            cond_emb = encoder(cond_batch, dropout_prob=dropout_prob if use_cfg else 0.0)
            pred_v = model(xt, t, cond_emb)
            loss = masked_mse(pred_v, target_v, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        history["train_loss"].append(float(np.mean(train_losses)) if train_losses else float("nan"))

        model.eval()
        encoder.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                coords = batch["coords"].to(device)
                mask = batch["mask"].to(device)
                cond_batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                x1 = coords
                x0 = torch.randn_like(x1)
                t = torch.rand(x1.shape[0], 1, device=device)
                xt = (1.0 - t[:, None, :]) * x0 + t[:, None, :] * x1
                target_v = x1 - x0
                cond_emb = encoder(cond_batch, dropout_prob=0.0)
                pred_v = model(xt, t, cond_emb)
                val_losses.append(float(masked_mse(pred_v, target_v, mask).item()))
        history["val_loss"].append(float(np.mean(val_losses)) if val_losses else float("nan"))
        LOGGER.info(
            "Epoch %d finished - train_loss=%.6f val_loss=%.6f",
            len(history["train_loss"]),
            history["train_loss"][-1],
            history["val_loss"][-1],
        )

    return history


def collect_feature_matrix(dataset: ProteinTensorDataset) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    rows: List[Dict[str, object]] = []
    for i in range(len(dataset)):
        item = dataset[i]
        coords = item["coords"].numpy()
        dist = item["dist"].numpy()
        features.append(structure_summary_features(coords, dist))
        labels.append(int(item["label"]))
        rows.append(item)
    return np.vstack(features), np.asarray(labels), rows


def train_state_probe(train_x: np.ndarray, train_y: np.ndarray) -> object:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(train_x, train_y)
    return clf


def probe_metrics(clf, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = clf.predict_proba(features)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
    }


def generate_eval_set(
    model: FlowMatchingModel,
    encoder: nn.Module,
    split_df: pd.DataFrame,
    dataset: ProteinTensorDataset,
    guidance_scale: float,
    device: torch.device,
    max_samples: int = MAX_GENERATION_SAMPLES,
) -> Dict[str, object]:
    indices = np.linspace(0, len(dataset) - 1, min(len(dataset), max_samples), dtype=int) if len(dataset) else []
    generated = []
    desired_labels = []
    target_refs = []
    rows = []
    for idx in indices:
        item = dataset[int(idx)]
        batch = {
            "kinase_idx": torch.tensor([item["kinase_idx"]], dtype=torch.long),
            "state_idx": torch.tensor([item["state_idx"]], dtype=torch.long),
            "dfg_idx": torch.tensor([item["dfg_idx"]], dtype=torch.long),
            "alphac_idx": torch.tensor([item["alphac_idx"]], dtype=torch.long),
            "mask": torch.ones(1, item["coords"].shape[0], dtype=torch.bool),
        }
        sampled = euler_sample(model, encoder, batch, guidance_scale=guidance_scale, device=device)[0].cpu().numpy()
        generated.append(sampled[: item["coords"].shape[0]])
        desired_labels.append(item["label"])
        rows.append(item)
        target_refs.append(item["coords"].numpy())
    return {
        "generated": generated,
        "desired_labels": np.asarray(desired_labels),
        "reference_coords": target_refs,
        "rows": rows,
    }


def generation_metrics(
    generated: Sequence[np.ndarray],
    reference_coords: Sequence[np.ndarray],
    labels: np.ndarray,
    clf,
) -> Dict[str, float]:
    feats = []
    rmsds = []
    csim = []
    for gen, ref in zip(generated, reference_coords):
        n = min(len(gen), len(ref))
        gen = gen[:n]
        ref = ref[:n]
        dist_gen = np.linalg.norm(gen[:, None, :] - gen[None, :, :], axis=-1)
        dist_ref = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=-1)
        feats.append(structure_summary_features(gen, dist_gen))
        rmsds.append(rmsd_aligned(gen, ref))
        csim.append(approx_contacts_similarity(gen, ref))
    feat_arr = np.vstack(feats) if feats else np.zeros((0, 7), dtype=np.float32)
    metric_block = probe_metrics(clf, feat_arr, labels) if len(feat_arr) else {}
    probs = clf.predict_proba(feat_arr)[:, 1] if len(feat_arr) else np.array([])
    preds = (probs >= 0.5).astype(int) if len(probs) else np.array([])
    metric_block.update(
        {
            "rmsd": float(np.nanmean(rmsds)) if rmsds else float("nan"),
            "diversity": float(pairwise_rmsd(list(generated))) if len(generated) > 1 else 0.0,
            "contact_similarity": float(np.nanmean(csim)) if csim else float("nan"),
            "mean_matrix_distance": float(np.nanmean([np.linalg.norm((np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1) - np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)), ord="fro") / max(1, g.shape[0] ** 2) for g, r in zip(generated, reference_coords)])) if generated else float("nan"),
            "stability": float(np.nanstd(rmsds)) if rmsds else float("nan"),
        }
    )
    return metric_block


def plot_confusion(ax, y_true: np.ndarray, y_prob: np.ndarray, title: str) -> None:
    _, sns = plotting()
    pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


def save_curves(curves: Dict[str, Dict[str, np.ndarray]], path_prefix: Path) -> None:
    plt, sns = plotting()
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, vals in curves.items():
        ax.plot(vals["fpr"], vals["tpr"], label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_prefix / "roc_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, vals in curves.items():
        ax.plot(vals["recall"], vals["precision"], label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_prefix / "pr_curves.png", dpi=200)
    plt.close(fig)


def plot_metric_comparison(metric_rows: pd.DataFrame) -> None:
    plt, sns = plotting()
    fig, ax = plt.subplots(figsize=(10, 6))
    melted = metric_rows.melt(id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    sns.barplot(data=melted, x="variable", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Classification metrics comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metric_rows, x="model", y="rmsd", ax=ax)
    ax.set_title("RMSD by model")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rmsd_by_model.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=metric_rows, x="model", y="diversity", ax=ax)
    ax.set_title("Diversity by model")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "diversity_by_model.png", dpi=200)
    plt.close(fig)


def plot_transition_analysis(
    inactive: np.ndarray,
    active: np.ndarray,
    intermediates: Sequence[np.ndarray],
    title_suffix: str = "",
) -> None:
    plt, sns = plotting()
    rows = []
    for i, coords in enumerate([inactive, *intermediates, active]):
        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        rows.append({"step": i, "rg": np.sqrt(np.mean(np.sum((coords - coords.mean(axis=0)) ** 2, axis=-1))), "mean_dist": float(np.mean(dist[dist > 0]))})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["step"], df["rg"], marker="o", label="Radius of gyration")
    ax.plot(df["step"], df["mean_dist"], marker="o", label="Mean pairwise distance")
    ax.set_title(f"Inactive -> Active transition {title_suffix}".strip())
    ax.set_xlabel("Transition step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "transition_path.png", dpi=200)
    plt.close(fig)


def choose_transition_pairs(splits: Dict[str, pd.DataFrame]) -> List[Tuple[pd.Series, pd.Series]]:
    df = pd.concat(splits.values(), ignore_index=True)
    pairs = []
    for kinase, grp in df.groupby("kinase_name"):
        inactive = grp[grp["conformational_state"] == "inactive"]
        active = grp[grp["conformational_state"] == "active"]
        if len(inactive) and len(active):
            pairs.append((inactive.iloc[0], active.iloc[0]))
    return pairs


def write_summary(metric_rows: pd.DataFrame, notes: List[str]) -> None:
    lines = [
        "# Flow Matching + CFG comparison",
        "",
        "## Notes",
    ]
    lines.extend([f"- {note}" for note in notes])
    lines.extend(
        [
            "",
            "## Final comparison table",
            "",
            "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | RMSD | Diversity |",
            "| ------ | -------- | --------- | ------ | -- | ------- | ------ | ---- | --------- |",
        ]
    )
    if len(metric_rows):
        for _, row in metric_rows.iterrows():
            lines.append(
                f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} | {row['rmsd']:.4f} | {row['diversity']:.4f} |"
            )
    else:
        lines.append("| _No executed results yet_ | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The current workspace may not contain the processed tensors or split files required for end-to-end training.",
            "- Metrics for generative comparison rely on a downstream probe trained only on the training split.",
            "- The notebook is designed to be rerun on the full dataset without changing splits or labels.",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_experiment() -> Dict[str, object]:
    set_seed()
    ensure_dirs()

    notes: List[str] = []
    LOGGER.info("Starting flow matching + CFG comparison experiment")
    if not data_ready():
        try:
            prepare_data_from_api(notes)
        except Exception as exc:
            notes.append(
                f"Automatic KLIFS API preparation failed: {exc!r}. The notebook will still write a scaffold summary, but no training/evaluation was run."
            )
            write_summary(pd.DataFrame(), notes)
            return {"status": "skipped", "notes": notes}

    if not data_ready():
        notes.append(
            "The dataset artifacts are still unavailable after the API preparation step, so the notebook writes a reproducible scaffold and summary template instead of training models."
        )
        write_summary(pd.DataFrame(), notes)
        return {"status": "skipped", "notes": notes}

    metadata_df, splits = load_metadata_and_splits()
    LOGGER.info("Loaded metadata and splits")
    combined_df = pd.concat(splits.values(), ignore_index=True)
    vocab = build_vocab(combined_df)
    vocab_sizes = {k: len(v) for k, v in vocab.items()}

    train_ds = ProteinTensorDataset(splits["train"], PROCESSED_DIR, vocab)
    val_ds = ProteinTensorDataset(splits["val"], PROCESSED_DIR, vocab)
    test_ds = ProteinTensorDataset(splits["test"], PROCESSED_DIR, vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    train_x, train_y, _ = collect_feature_matrix(train_ds)
    val_x, val_y, _ = collect_feature_matrix(val_ds)
    test_x, test_y, test_rows = collect_feature_matrix(test_ds)
    LOGGER.info("Trained downstream probe features extracted")
    probe = train_state_probe(train_x, train_y)
    probe_val = probe_metrics(probe, val_x, val_y)
    probe_test = probe_metrics(probe, test_x, test_y)
    notes.append(f"Downstream probe validation ROC-AUC: {probe_val['roc_auc']:.4f}; test ROC-AUC: {probe_test['roc_auc']:.4f}.")
    LOGGER.info("Downstream probe ready - val ROC-AUC=%.4f test ROC-AUC=%.4f", probe_val["roc_auc"], probe_test["roc_auc"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    baseline_encoder = BinaryConditionEncoder(vocab_size=vocab_sizes["kinase"], embed_dim=64, cond_dim=128)
    baseline_model = FlowMatchingModel(cond_dim=128, hidden_dim=256, depth=3)
    LOGGER.info("Training baseline flow matching model")
    baseline_history = train_flow_matching(
        baseline_model,
        baseline_encoder,
        train_loader,
        val_loader,
        epochs=EPOCHS_BASELINE,
        lr=LEARNING_RATE,
        dropout_prob=0.0,
        device=device,
        use_cfg=False,
    )

    cfg_encoder = ConditionEncoder(vocab_sizes=vocab_sizes, embed_dim=64, cond_dim=128)
    cfg_model = FlowMatchingModel(cond_dim=128, hidden_dim=256, depth=3)
    LOGGER.info("Training Flow Matching + CFG model")
    cfg_history = train_flow_matching(
        cfg_model,
        cfg_encoder,
        train_loader,
        val_loader,
        epochs=EPOCHS_CFG,
        lr=LEARNING_RATE,
        dropout_prob=CONDITION_DROPOUT_PROB,
        device=device,
        use_cfg=True,
    )

    model_specs = [
        ("Existing baseline", baseline_model, baseline_encoder, 0.0),
        ("Flow Matching no CFG", cfg_model, cfg_encoder, 0.0),
    ]
    for scale in GUIDANCE_SCALES[1:]:
        model_specs.append((f"Flow Matching + CFG {scale:g}", cfg_model, cfg_encoder, float(scale)))

    metrics_rows = []
    roc_curves = {}
    pr_curves = {}
    for name, model, encoder, scale in model_specs:
        LOGGER.info("Evaluating %s (guidance_scale=%.1f)", name, scale)
        eval_pack = generate_eval_set(model, encoder, test_ds.split_df if hasattr(test_ds, "split_df") else splits["test"], test_ds, scale, device)
        gen_metrics = generation_metrics(
            eval_pack["generated"],
            eval_pack["reference_coords"],
            eval_pack["desired_labels"],
            probe,
        )
        row = {"model": name, **gen_metrics}
        metrics_rows.append(row)
        if len(eval_pack["generated"]):
            feats = np.vstack([
                structure_summary_features(gen, np.linalg.norm(gen[:, None, :] - gen[None, :, :], axis=-1))
                for gen in eval_pack["generated"]
            ])
            probs = probe.predict_proba(feats)[:, 1]
            y_true = eval_pack["desired_labels"]
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, probs)
                precision, recall, _ = precision_recall_curve(y_true, probs)
            else:
                fpr, tpr, precision, recall = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0])
            roc_curves[name] = {"fpr": fpr, "tpr": tpr}
            pr_curves[name] = {"precision": precision, "recall": recall}

    metric_df = pd.DataFrame(metrics_rows)
    LOGGER.info("Generated comparison metrics for %d models", len(metric_df))

    if len(roc_curves):
        save_curves({k: {**v, **pr_curves.get(k, {})} for k, v in roc_curves.items()}, PLOTS_DIR)

    plot_metric_comparison(metric_df)

    plot_transition_analysis(
        train_ds[0]["coords"].numpy(),
        train_ds[min(1, len(train_ds) - 1)]["coords"].numpy(),
        [train_ds[min(i, len(train_ds) - 1)]["coords"].numpy() for i in range(2, min(6, len(train_ds)))],
    )

    plt, sns = plotting()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    axes[0].plot(baseline_history["train_loss"], label="train")
    axes[0].plot(baseline_history["val_loss"], label="val")
    axes[0].set_title("Baseline loss")
    axes[0].legend()
    axes[1].plot(cfg_history["train_loss"], label="train")
    axes[1].plot(cfg_history["val_loss"], label="val")
    axes[1].set_title("Flow Matching + CFG loss")
    axes[1].legend()
    sns.heatmap(confusion_matrix(test_y, probe.predict(test_x)), annot=True, fmt="d", cbar=False, ax=axes[2])
    axes[2].set_title("Probe confusion matrix on test")
    axes[3].hist(test_x[:, 1], bins=20, alpha=0.7, label="test structures")
    axes[3].set_title("Feature distribution proxy")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "loss_and_proxies.png", dpi=200)
    plt.close(fig)

    metric_df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    write_summary(metric_df, notes)
    LOGGER.info("Results written to %s", RESULTS_DIR)

    return {
        "status": "completed",
        "probe_val": probe_val,
        "probe_test": probe_test,
        "metrics": metric_df,
        "notes": notes,
    }


def main() -> Dict[str, object]:
    result = run_experiment()
    print(json.dumps({"status": result.get("status", "unknown"), "notes": result.get("notes", [])}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
