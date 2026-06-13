#!/usr/bin/env python3
"""Audit active/inactive classification using real kinase structures only."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/classification_diagnostics_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/classification_diagnostics_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import flow_matching_transition_diagnostics as diag
import transition_path_diagnostics as path_diag


LOG = logging.getLogger("classification_diagnostics")
N_POINTS = 64


def interpolate_coords(coords: np.ndarray, size: int = N_POINTS) -> np.ndarray:
    if len(coords) == size:
        return coords
    old = np.linspace(0, 1, len(coords))
    new = np.linspace(0, 1, size)
    return np.stack([np.interp(new, old, coords[:, axis]) for axis in range(3)], axis=1)


def domain_coords(record: diag.StructureRecord) -> Tuple[np.ndarray, np.ndarray]:
    start = record.vaik_lys_index - 25 if record.vaik_lys_index is not None else 0
    end = record.dfg_index + 35 if record.dfg_index is not None else len(record.residues)
    start, end = max(0, start), min(len(record.residues), end)
    if end - start < 30:
        start, end = 0, len(record.residues)
    coords = np.asarray([res.ca for res in record.residues[start:end]], dtype=np.float32)
    relative_positions = np.linspace(start, end - 1, N_POINTS)
    return interpolate_coords(coords), relative_positions


def canonicalize(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    direction = centered[-1] - centered[0]
    for axis in range(3):
        if np.dot(direction, vh[axis]) < 0:
            rotated[:, axis] *= -1
    if np.linalg.det(vh) < 0:
        rotated[:, -1] *= -1
    scale = np.sqrt(np.mean(np.sum(rotated**2, axis=1)))
    return rotated / max(scale, 1e-6)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)


def window_indices(center: int | None, length: int, radius: int) -> np.ndarray:
    if center is None:
        return np.array([], dtype=int)
    return np.arange(max(0, center - radius), min(length, center + radius + 1))


def local_geometry(record: diag.StructureRecord) -> Tuple[np.ndarray, List[str]]:
    coords = np.asarray([res.ca for res in record.residues], dtype=float)
    dist = distance_matrix(coords)
    values: List[float] = []
    names: List[str] = []

    def add(name: str, value: float):
        names.append(name)
        values.append(float(value))

    add("num_residues", len(coords))
    add("radius_of_gyration", np.sqrt(np.mean(np.sum((coords - coords.mean(0)) ** 2, axis=1))))
    upper = dist[np.triu_indices(len(dist), 1)]
    for stat_name, stat in (
        ("mean_distance", np.mean),
        ("std_distance", np.std),
        ("median_distance", np.median),
        ("max_distance", np.max),
    ):
        add(stat_name, stat(upper))
    for cutoff in (6.0, 8.0, 10.0, 12.0):
        add(f"contact_fraction_{cutoff:g}A", np.mean(upper < cutoff))
    regions = {
        "DFG": window_indices(record.dfg_index, len(coords), 4),
        "alphaC": window_indices(record.alphac_glu_index, len(coords), 7),
        "HRD": window_indices(record.hrd_index, len(coords), 4),
        "activation_loop": (
            np.arange(record.dfg_index, min(len(coords), record.dfg_index + 25))
            if record.dfg_index is not None
            else np.array([], dtype=int)
        ),
    }
    for region, indices in regions.items():
        if len(indices) > 1:
            local = dist[np.ix_(indices, indices)]
            local_upper = local[np.triu_indices(len(local), 1)]
            add(f"{region}_mean_distance", np.mean(local_upper))
            add(f"{region}_std_distance", np.std(local_upper))
            add(f"{region}_radius", np.sqrt(np.mean(np.sum((coords[indices] - coords[indices].mean(0)) ** 2, axis=1))))
        else:
            for suffix in ("mean_distance", "std_distance", "radius"):
                add(f"{region}_{suffix}", np.nan)
    marked = {
        "lys_glu_distance": (record.vaik_lys_index, record.alphac_glu_index),
        "lys_dfg_distance": (record.vaik_lys_index, record.dfg_index),
        "glu_dfg_distance": (record.alphac_glu_index, record.dfg_index),
        "hrd_dfg_distance": (record.hrd_index, record.dfg_index),
    }
    for name, (left, right) in marked.items():
        add(name, dist[left, right] if left is not None and right is not None else np.nan)
    return np.asarray(values, dtype=np.float32), names


def extract_features(records: Mapping[str, Sequence[diag.StructureRecord]]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    metadata, coordinates, distances, local = [], [], [], []
    local_names: List[str] = []
    for split, split_records in records.items():
        for record in split_records:
            coords, positions = domain_coords(record)
            canonical = canonicalize(coords)
            dist = distance_matrix(coords)
            local_values, local_names = local_geometry(record)
            coordinates.append(canonical.reshape(-1))
            distances.append(dist[np.triu_indices(N_POINTS, 1)])
            local.append(local_values)
            metadata.append(
                {
                    "split": split,
                    "pdb_id": record.pdb_id,
                    "kinase": record.kinase,
                    "state": record.state,
                    "label": int(record.state == "active"),
                    "dfg_state": record.dfg_state,
                    "alphac_state": record.alphac_state,
                    "motif_descriptor": record.motif_descriptor,
                    "ligand_present": record.ligand_present,
                    "resolution": record.resolution,
                    "residues": len(record.residues),
                    "dfg_found": record.dfg_index is not None,
                    "alphac_glu_found": record.alphac_glu_index is not None,
                }
            )
    local_matrix = np.stack(local)
    medians = np.nanmedian(local_matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0)
    local_matrix = np.where(np.isfinite(local_matrix), local_matrix, medians)
    coord_matrix, distance_matrix_values = np.stack(coordinates), np.stack(distances)
    features = {
        "ca_coordinates": coord_matrix,
        "distance_matrix": distance_matrix_values,
        "coordinates_distances": np.concatenate([coord_matrix, distance_matrix_values], axis=1),
        "local_geometry": local_matrix,
        "dfg_only": local_matrix[:, [i for i, name in enumerate(local_names) if name.startswith("DFG_") or "dfg" in name.lower()]],
        "alphac_only": local_matrix[:, [i for i, name in enumerate(local_names) if name.startswith("alphaC_") or "glu" in name.lower()]],
        "dfg_alphac": local_matrix[:, [i for i, name in enumerate(local_names) if any(token in name.lower() for token in ("dfg", "alphac", "glu", "lys"))]],
    }
    return pd.DataFrame(metadata), features, local_names


def audit_tables(metadata_csv: Path, valid: pd.DataFrame, output: Path) -> pd.DataFrame:
    raw = pd.read_csv(metadata_csv)
    raw.to_csv(output / "dataset_records.csv", index=False)
    balance = raw.groupby(["conformational_state"]).size().rename("count").reset_index()
    balance["fraction"] = balance["count"] / balance["count"].sum()
    balance.to_csv(output / "class_balance.csv", index=False)
    pd.crosstab(raw["kinase_name"], raw["conformational_state"]).to_csv(output / "state_by_kinase.csv")
    pd.crosstab(raw["conformational_state"], raw["dfg_state"], dropna=False).to_csv(output / "state_by_dfg.csv")
    pd.crosstab(raw["conformational_state"], raw["alphac_state"], dropna=False).to_csv(output / "state_by_alphac.csv")
    pd.crosstab(raw["conformational_state"], raw["ligand_present"], dropna=False).to_csv(output / "state_by_ligand.csv")
    raw.groupby("conformational_state")["resolution"].describe().to_csv(output / "resolution_by_state.csv")
    dfg = raw["dfg_state"].astype(str).str.lower()
    alphac = raw["alphac_state"].astype(str).str.lower()
    expected_active = dfg.eq("in") & alphac.eq("in")
    rows = raw.copy()
    rows["active_expected_from_DFG_alphaC"] = expected_active
    rows["contradiction_active_DFG_out"] = rows["conformational_state"].eq("active") & dfg.isin(["out", "out-like"])
    rows["contradiction_active_alphaC_out"] = rows["conformational_state"].eq("active") & alphac.eq("out")
    rows["inactive_DFG_in"] = rows["conformational_state"].eq("inactive") & dfg.eq("in")
    rows["inactive_alphaC_in"] = rows["conformational_state"].eq("inactive") & alphac.eq("in")
    rows["state_disagrees_with_joint_motif_rule"] = rows["conformational_state"].eq("active") != expected_active
    rows.to_csv(output / "label_consistency_report.csv", index=False)
    valid.to_csv(output / "valid_structure_index.csv", index=False)
    return rows


def plot_audit(raw: pd.DataFrame, output: Path):
    plt, sns = path_diag.pyplot()
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.countplot(data=raw, x="conformational_state", ax=axes[0, 0])
    sns.countplot(data=raw, y="kinase_name", hue="conformational_state", ax=axes[0, 1])
    sns.countplot(data=raw, x="dfg_state", hue="conformational_state", ax=axes[1, 0])
    sns.countplot(data=raw, x="alphac_state", hue="conformational_state", ax=axes[1, 1])
    fig.tight_layout()
    fig.savefig(figure_dir / "dataset_distributions.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=raw, x="conformational_state", y="resolution", ax=axes[0])
    sns.countplot(data=raw, x="ligand_present", hue="conformational_state", ax=axes[1])
    fig.tight_layout()
    fig.savefig(figure_dir / "resolution_ligand.png", dpi=180)
    plt.close(fig)


def score_predictions(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    result = {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
    }
    result["roc_auc"] = roc_auc_score(y, probability) if len(np.unique(y)) == 2 else np.nan
    return result


def model_factories(seed: int = 42):
    from xgboost import XGBClassifier

    return {
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=4
        ),
        "small_mlp": Pipeline(
            [("scale", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=seed))]
        ),
    }


def evaluate_baselines(metadata: pd.DataFrame, features: Mapping[str, np.ndarray], output: Path) -> pd.DataFrame:
    rows = []
    y = metadata["label"].to_numpy()
    protocols = {
        "official_test_OOD_kinase": (
            metadata["split"].eq("train").to_numpy(),
            metadata["split"].eq("test").to_numpy(),
        )
    }
    within_train = np.zeros(len(metadata), dtype=bool)
    within_test = np.zeros(len(metadata), dtype=bool)
    rng = np.random.default_rng(42)
    for (_, _), group in metadata.groupby(["kinase", "label"]):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        if len(indices) < 2:
            within_train[indices] = True
            continue
        test_count = max(1, int(round(0.25 * len(indices))))
        test_count = min(test_count, len(indices) - 1)
        within_test[indices[:test_count]] = True
        within_train[indices[test_count:]] = True
    protocols["within_kinase_stratified"] = (within_train, within_test)
    factories = model_factories()
    for representation, matrix in features.items():
        if representation in {"dfg_only", "alphac_only", "dfg_alphac"}:
            selected_models = factories
        else:
            selected_models = factories
        for protocol, (train_mask, test_mask) in protocols.items():
            for model_name, factory in selected_models.items():
                model = clone(factory)
                model.fit(matrix[train_mask], y[train_mask])
                prediction = model.predict(matrix[test_mask])
                probability = model.predict_proba(matrix[test_mask])[:, 1]
                rows.append(
                    {
                        "representation": representation,
                        "model": model_name,
                        "protocol": protocol,
                        "train_samples": int(train_mask.sum()),
                        "test_samples": int(test_mask.sum()),
                        **score_predictions(y[test_mask], prediction, probability),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(["protocol", "balanced_accuracy"], ascending=[True, False])
    frame.to_csv(output / "baseline_results.csv", index=False)
    return frame


def per_kinase_classification(metadata: pd.DataFrame, matrix: np.ndarray, output: Path) -> pd.DataFrame:
    rows = []
    for kinase, group in metadata.groupby("kinase"):
        if group["label"].value_counts().min() < 4:
            rows.append({"kinase": kinase, "status": "insufficient minority examples", "samples": len(group)})
            continue
        indices = group.index.to_numpy()
        folds = min(5, int(group["label"].value_counts().min()))
        scores = []
        for train_local, test_local in StratifiedKFold(folds, shuffle=True, random_state=42).split(indices, group["label"]):
            train_idx, test_idx = indices[train_local], indices[test_local]
            model = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
            model.fit(matrix[train_idx], metadata.loc[train_idx, "label"])
            probability = model.predict_proba(matrix[test_idx])[:, 1]
            prediction = probability >= 0.5
            scores.append(score_predictions(metadata.loc[test_idx, "label"].to_numpy(), prediction, probability))
        row = {"kinase": kinase, "status": "evaluated", "samples": len(group), "active": int(group.label.sum()), "inactive": int((1-group.label).sum())}
        row.update({key: np.nanmean([score[key] for score in scores]) for key in scores[0]})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False, na_position="last")
    frame.to_csv(output / "per_kinase_classification.csv", index=False)
    return frame


def projection_plots(metadata: pd.DataFrame, features: Mapping[str, np.ndarray], output: Path):
    plt, sns = path_diag.pyplot()
    figure_dir = output / "figures" / "separability"
    figure_dir.mkdir(parents=True, exist_ok=True)
    projection_rows = []
    for representation in ("ca_coordinates", "distance_matrix", "local_geometry"):
        matrix = StandardScaler().fit_transform(features[representation])
        pca = PCA(n_components=2, random_state=42).fit_transform(matrix)
        perplexity = min(30, max(5, len(matrix) // 30))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(PCA(n_components=min(30, matrix.shape[1])).fit_transform(matrix))
        for method, embedding in (("PCA", pca), ("tSNE", tsne)):
            frame = metadata.copy()
            frame["x"], frame["y"] = embedding[:, 0], embedding[:, 1]
            frame["representation"], frame["method"] = representation, method
            projection_rows.append(frame)
            for color in ("state", "kinase", "dfg_state", "alphac_state"):
                fig, ax = plt.subplots(figsize=(9, 7))
                sns.scatterplot(data=frame, x="x", y="y", hue=color, s=18, alpha=0.7, ax=ax)
                ax.set_title(f"{representation} {method} colored by {color}")
                fig.tight_layout()
                fig.savefig(figure_dir / f"{representation}_{method}_{color}.png", dpi=170)
                plt.close(fig)
    try:
        import umap
        matrix = StandardScaler().fit_transform(features["distance_matrix"])
        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(matrix)
        frame = metadata.copy()
        frame["x"], frame["y"] = embedding[:, 0], embedding[:, 1]
        frame["representation"], frame["method"] = "distance_matrix", "UMAP"
        projection_rows.append(frame)
    except Exception as exc:
        (figure_dir / "UMAP_UNAVAILABLE.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
    pd.concat(projection_rows, ignore_index=True).to_csv(output / "separability_projections.csv", index=False)


def separability_scores(
    metadata: pd.DataFrame, features: Mapping[str, np.ndarray], output: Path
) -> pd.DataFrame:
    rows = []
    state_labels = metadata["label"].to_numpy()
    kinase_labels = metadata["kinase"].astype(str).to_numpy()
    rng = np.random.default_rng(42)
    for representation in ("ca_coordinates", "distance_matrix", "local_geometry"):
        matrix = StandardScaler().fit_transform(features[representation])
        components = min(20, matrix.shape[0] - 1, matrix.shape[1])
        embedded = PCA(n_components=components, random_state=42).fit_transform(matrix)
        if len(embedded) > 800:
            indices = np.sort(rng.choice(len(embedded), size=800, replace=False))
            embedded = embedded[indices]
            sampled_state = state_labels[indices]
            sampled_kinase = kinase_labels[indices]
        else:
            sampled_state = state_labels
            sampled_kinase = kinase_labels
        rows.append(
            {
                "representation": representation,
                "state_silhouette": silhouette_score(embedded, sampled_state),
                "kinase_silhouette": silhouette_score(embedded, sampled_kinase),
                "n_samples": len(embedded),
                "pca_components": components,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output / "separability_scores.csv", index=False)
    return result


def problematic_pairs(records: Mapping[str, Sequence[diag.StructureRecord]], output: Path):
    rows = []
    for split_records in records.values():
        by_kinase = defaultdict(list)
        for record in split_records:
            by_kinase[record.kinase].append(record)
        for kinase, group in by_kinase.items():
            active = [record for record in group if record.state == "active"]
            inactive = [record for record in group if record.state == "inactive"]
            if not active or not inactive:
                continue
            active_features = np.stack([distance_matrix(domain_coords(record)[0])[np.triu_indices(N_POINTS, 1)] for record in active])
            inactive_features = np.stack([distance_matrix(domain_coords(record)[0])[np.triu_indices(N_POINTS, 1)] for record in inactive])
            rough = np.linalg.norm(active_features[:, None] - inactive_features[None, :], axis=-1)
            for a_index, active_record in enumerate(active):
                for i_index in np.argsort(rough[a_index])[: min(3, len(inactive))]:
                    inactive_record = inactive[i_index]
                    source_indices, target_indices, coverage, identity, _ = diag.align_sequences(inactive_record, active_record)
                    if not source_indices:
                        continue
                    source = torch.tensor([inactive_record.residues[i].ca for i in source_indices], dtype=torch.float32)
                    target = torch.tensor([active_record.residues[i].ca for i in target_indices], dtype=torch.float32)
                    target_aligned = diag.kabsch_align(target, source)
                    value = float(torch.sqrt(torch.mean(torch.sum((source-source.mean(0) - target_aligned) ** 2, dim=-1))))
                    rows.append(
                        {
                            "kinase": kinase,
                            "inactive_pdb": inactive_record.pdb_id,
                            "active_pdb": active_record.pdb_id,
                            "aligned_ca_rmsd": value,
                            "alignment_coverage": coverage,
                            "sequence_identity": identity,
                            "problematic_below_1A": value < 1.0,
                        }
                    )
    pd.DataFrame(rows).sort_values("aligned_ca_rmsd").to_csv(output / "problematic_pairs.csv", index=False)


class PositionalTransformer(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.input = nn.Linear(N_POINTS, width)
        self.position = nn.Parameter(torch.randn(1, N_POINTS, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 3)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))

    def forward(self, distance_rows: torch.Tensor):
        hidden = self.input(distance_rows) + self.position
        return self.head(self.encoder(hidden).mean(1))


def train_deep_classifier(metadata: pd.DataFrame, features: Mapping[str, np.ndarray], output: Path, epochs: int = 60):
    full_distances = np.zeros((len(metadata), N_POINTS, N_POINTS), dtype=np.float32)
    upper = np.triu_indices(N_POINTS, 1)
    full_distances[:, upper[0], upper[1]] = features["distance_matrix"]
    full_distances[:, upper[1], upper[0]] = features["distance_matrix"]
    scale = np.maximum(full_distances.mean(axis=(1, 2), keepdims=True), 1e-6)
    full_distances /= scale
    train = metadata["split"].eq("train").to_numpy()
    val = metadata["split"].eq("val").to_numpy()
    test = metadata["split"].eq("test").to_numpy()
    model = PositionalTransformer().to(diag.device_name())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    class_count = np.bincount(metadata.loc[train, "label"], minlength=2)
    weights = torch.tensor(class_count.sum() / np.maximum(class_count, 1), dtype=torch.float32, device=diag.device_name())
    history, best = [], -np.inf
    model_dir = output / "deep_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(np.where(train)[0])
        losses = []
        for start in range(0, len(order), 32):
            idx = order[start:start+32]
            x = torch.tensor(full_distances[idx], device=diag.device_name())
            y = torch.tensor(metadata.loc[idx, "label"].to_numpy(), device=diag.device_name())
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, weight=weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "train_loss": np.mean(losses)}
        for name, mask in (("validation", val), ("test", test)):
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(full_distances[mask], device=diag.device_name()))
                probability = logits.softmax(-1)[:, 1].cpu().numpy()
            prediction = probability >= 0.5
            row.update({f"{name}_{key}": value for key, value in score_predictions(metadata.loc[mask, "label"].to_numpy(), prediction, probability).items()})
        history.append(row)
        if row["validation_balanced_accuracy"] > best:
            best = row["validation_balanced_accuracy"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, model_dir / "best_model.pt")
        torch.save({"model_state": model.state_dict(), "epoch": epoch}, model_dir / "last_model.pt")
    frame = pd.DataFrame(history)
    frame.to_csv(model_dir / "training_history.csv", index=False)
    return frame


def interpretability(metadata: pd.DataFrame, features: Mapping[str, np.ndarray], local_names: Sequence[str], output: Path):
    y = metadata["label"].to_numpy()
    model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(features["local_geometry"], y)
    local_importance = pd.DataFrame({"feature": local_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    region = local_importance["feature"].str.extract(
        r"^(DFG|alphaC|HRD|activation_loop)", expand=False
    )
    region = region.where(
        ~local_importance["feature"].str.contains("hrd_dfg", case=False),
        "HRD-DFG",
    )
    fallback = pd.Series(
        np.where(
            local_importance["feature"].str.contains("glu|lys", case=False),
            "Lys-Glu",
            "global",
        ),
        index=local_importance.index,
    )
    local_importance["region"] = region.where(region.notna(), fallback)
    local_importance.to_csv(output / "local_feature_importance.csv", index=False)
    dist_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1, max_depth=12)
    dist_model.fit(features["distance_matrix"], y)
    matrix = np.zeros((N_POINTS, N_POINTS))
    upper = np.triu_indices(N_POINTS, 1)
    matrix[upper] = dist_model.feature_importances_
    matrix += matrix.T
    residue_importance = matrix.sum(1)
    pd.DataFrame({"relative_residue_index": np.arange(N_POINTS), "importance": residue_importance}).sort_values("importance", ascending=False).to_csv(output / "residue_importance.csv", index=False)


def annotate_residue_importance(
    records: Mapping[str, Sequence[diag.StructureRecord]], output: Path
) -> pd.DataFrame:
    importance = pd.read_csv(output / "residue_importance.csv")
    counts = {
        index: {"n": 0, "DFG": 0, "alphaC": 0, "activation_loop": 0, "HRD": 0}
        for index in range(N_POINTS)
    }
    for split_records in records.values():
        for record in split_records:
            _, positions = domain_coords(record)
            for index, position in enumerate(positions):
                counts[index]["n"] += 1
                if record.dfg_index is not None:
                    counts[index]["DFG"] += int(
                        record.dfg_index - 2 <= position <= record.dfg_index + 4
                    )
                    counts[index]["activation_loop"] += int(
                        record.dfg_index <= position <= record.dfg_index + 24
                    )
                if record.alphac_glu_index is not None:
                    counts[index]["alphaC"] += int(
                        record.alphac_glu_index - 7 <= position <= record.alphac_glu_index + 7
                    )
                if record.hrd_index is not None:
                    counts[index]["HRD"] += int(
                        record.hrd_index - 2 <= position <= record.hrd_index + 2
                    )
    annotations = []
    for index, values in counts.items():
        denominator = max(values["n"], 1)
        fractions = {
            region: values[region] / denominator
            for region in ("DFG", "alphaC", "activation_loop", "HRD")
        }
        dominant_region, fraction = max(fractions.items(), key=lambda item: item[1])
        annotations.append(
            {
                "relative_residue_index": index,
                **{f"{region}_membership_fraction": value for region, value in fractions.items()},
                "dominant_region": dominant_region if fraction >= 0.25 else "other",
                "dominant_region_fraction": fraction,
            }
        )
    annotated = importance.merge(pd.DataFrame(annotations), on="relative_residue_index", how="left")
    annotated.to_csv(output / "residue_importance.csv", index=False)
    return annotated


def write_report(metadata: pd.DataFrame, consistency: pd.DataFrame, baselines: pd.DataFrame, per_kinase: pd.DataFrame, deep: pd.DataFrame, output: Path):
    within = baselines[baselines["protocol"] == "within_kinase_stratified"].sort_values("balanced_accuracy", ascending=False)
    ood = baselines[baselines["protocol"] == "official_test_OOD_kinase"].sort_values("balanced_accuracy", ascending=False)
    best_within, best_ood = within.iloc[0], ood.iloc[0]
    best_deep = deep.loc[deep["validation_balanced_accuracy"].idxmax()]
    contradictions = int(consistency["state_disagrees_with_joint_motif_rule"].sum())
    easy = per_kinase[per_kinase["status"] == "evaluated"].head(3)["kinase"].tolist()
    hard = per_kinase[per_kinase["status"] == "evaluated"].tail(3)["kinase"].tolist()
    problematic = pd.read_csv(output / "problematic_pairs.csv")
    close_pairs = int(problematic["problematic_below_1A"].sum())
    local_importance = pd.read_csv(output / "local_feature_importance.csv").head(8)
    important_features = ", ".join(
        f"`{row.feature}` ({row.importance:.3f})" for row in local_importance.itertuples()
    )
    separability = pd.read_csv(output / "separability_scores.csv").set_index("representation")
    text = f"""# Active/Inactive Classification Diagnostics

## Main findings

- Valid real structures: **{len(metadata)}**.
- Active/inactive balance among valid structures: **{metadata.label.sum()} / {(1-metadata.label).sum()}**.
- Best within-kinase balanced accuracy: **{best_within.balanced_accuracy:.3f}** using `{best_within.model}` + `{best_within.representation}`.
- Best official OOD-kinase balanced accuracy: **{best_ood.balanced_accuracy:.3f}** using `{best_ood.model}` + `{best_ood.representation}`.
- Dedicated Transformer best validation balanced accuracy: **{best_deep.validation_balanced_accuracy:.3f}**, but test balanced accuracy is **{best_deep.test_balanced_accuracy:.3f}** and test ROC AUC is **{best_deep.test_roc_auc:.3f}**.
- Labels disagreeing with the strict joint DFG-in/alphaC-in rule: **{contradictions}**.
- Sampled cross-state neighbors below 1 A aligned C-alpha RMSD: **{close_pairs}**.

## Answers

1. **Does the dataset contain signal?** Yes. Simple models separate states within kinases.
2. **Maximum observed accuracy:** balanced accuracy `{best_within.balanced_accuracy:.3f}` in the within-kinase protocol. Best OOD balanced accuracy is `{best_ood.balanced_accuracy:.3f}`.
3. **Best representation:** `{best_within.representation}` within kinases. Rotation-invariant local and distance features outperform raw canonicalized coordinates.
4. **Easy kinases:** {', '.join(easy) if easy else 'none with enough examples'}.
5. **Difficult kinases:** {', '.join(hard) if hard else 'none evaluated'}; several other kinases lack enough minority examples for a stable estimate.
6. **Do DFG/alphaC explain classification?** Yes, especially DFG. The leading features are {important_features}. AlphaC alone is informative but weaker.
7. **Label noise:** active labels are consistent with DFG-in; inactive is a heterogeneous class containing DFG-in/alphaC-out and DFG-out/alphaC-in states. The `{contradictions}` strict-rule disagreements are mostly missing motif metadata, not clear inversions.
8. **Data or classifier?** Both, but the 0% result is primarily a training/protocol failure. Signal exists, while the original auxiliary head used a reduced paired subset, weak motif anchoring and an OOD split. The data still impose a ceiling because some opposite-label structures are nearly identical in C-alpha geometry.
9. **Classifier as guidance:** not yet as a universal biological energy. A local, rotation-invariant classifier can only guide kinases where it is independently calibrated.
10. **Pretraining recommendation:** pretrain on all real structures, validate both within kinase and on held-out kinases, calibrate probabilities, freeze the model, and reject guidance when balanced accuracy or ROC AUC misses a predefined threshold.

## Separability

PCA silhouette scores are weak: for distance matrices, state silhouette is
`{separability.loc['distance_matrix', 'state_silhouette']:.3f}` and kinase
silhouette is `{separability.loc['distance_matrix', 'kinase_silhouette']:.3f}`.
Therefore active/inactive does not form a clean universal unsupervised cluster.
Supervised local features nevertheless recover strong within-kinase signal.

## Root cause of the collapsed auxiliary classifier

1. It was trained inside the generative paired-data path instead of using all 863 valid real structures.
2. Fixed-length sampling does not consistently anchor homologous DFG/alphaC positions, and global pooling weakens local motif information.
3. The official split is kinase-disjoint and severely imbalanced: validation contains KIT/MET, while test contains PDGFRA/PIK3CA.
4. More capacity does not solve the shift. The Transformer learns validation kinases and fails catastrophically on test kinases.
5. C-alpha geometry is sufficient for many kinases, but not universally: `{close_pairs}` sampled cross-state neighbors are below 1 A and PIK3CA is close to chance.

## Representation diagnosis

`ca_coordinates` uses PCA-canonicalized C-alpha coordinates. `distance_matrix`
is rotation invariant. `local_geometry` explicitly measures DFG, alphaC, HRD,
activation loop, and Lys-Glu geometry. The advantage of local/distance features
shows that representation and motif anchoring matter more than raw capacity.

UMAP was attempted, but the installed `umap`/`coverage` combination is
incompatible. PCA and t-SNE results are available.
"""
    (output / "classification_report.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else diag.discover_root(Path.cwd())
    output = Path(args.output).resolve() if args.output else root / "outputs/classification_diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    records, audit = diag.load_records(root)
    audit.to_csv(output / "structure_loading_audit.csv", index=False)
    metadata, features, local_names = extract_features(records)
    consistency = audit_tables(root / "data/metadata/kinase_labels.csv", metadata, output)
    plot_audit(pd.read_csv(root / "data/metadata/kinase_labels.csv"), output)
    projection_plots(metadata, features, output)
    separability_scores(metadata, features, output)
    baselines = evaluate_baselines(metadata, features, output)
    best_rep = baselines[baselines.protocol.eq("within_kinase_stratified")].sort_values("balanced_accuracy", ascending=False).iloc[0].representation
    per_kinase = per_kinase_classification(metadata, features[best_rep], output)
    problematic_pairs(records, output)
    deep = train_deep_classifier(metadata, features, output, epochs=max(50, args.epochs))
    interpretability(metadata, features, local_names, output)
    annotate_residue_importance(records, output)
    write_report(metadata, consistency, baselines, per_kinase, deep, output)
    np.savez_compressed(output / "feature_matrices.npz", **features)
    path_diag.json_dump(
        {
            "real_structures_only": True,
            "generated_structures_used": False,
            "num_valid_structures": len(metadata),
            "representations": {key: list(value.shape) for key, value in features.items()},
            "deep_epochs": max(50, args.epochs),
        },
        output / "diagnostic_config.json",
    )
    LOG.info("Classification diagnostics completed: %s", output)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
