#!/usr/bin/env python3
"""Train on real kinase structures and compare representative active/inactive states."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kinase_state_comparator_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kinase_state_comparator_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import classification_diagnostics as classification
import flow_matching_transition_diagnostics as diagnostics
import transition_path_diagnostics as path_diagnostics


LOG = logging.getLogger("kinase_state_comparator")
DEFAULT_COMPONENTS = ("DFG", "alphaC", "activation_loop", "HRD", "Lys-Glu")
COMPONENT_ALIASES = {
    "dfg": "DFG",
    "dfg_loop": "DFG",
    "alphac": "alphaC",
    "alphac_helix": "alphaC",
    "activation_loop": "activation_loop",
    "aloop": "activation_loop",
    "hrd": "HRD",
    "hrd_motif": "HRD",
    "lys-glu": "Lys-Glu",
    "lys_glu": "Lys-Glu",
    "lysglu": "Lys-Glu",
}


@dataclass
class ComparatorConfig:
    kinase: str | None = None
    components: tuple[str, ...] = DEFAULT_COMPONENTS
    prefer_ligand: bool = False
    max_resolution: float | None = None
    states: tuple[str, ...] = ("active", "inactive")
    use_full_dataset: bool = True
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 20
    seed: int = 42
    resume: bool = False
    all_kinases: bool = False
    force_train: bool = False


class StateMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def normalize_component(value: str) -> str:
    key = value.strip().lower().replace(" ", "_").replace("α", "alpha")
    if key not in COMPONENT_ALIASES:
        raise ValueError(f"Unknown component: {value}")
    return COMPONENT_ALIASES[key]


def flatten_records(
    records: Mapping[str, Sequence[diagnostics.StructureRecord]],
) -> list[diagnostics.StructureRecord]:
    return [record for split_records in records.values() for record in split_records]


def corrected_alphac_glu_index(record: diagnostics.StructureRecord) -> int | None:
    if record.vaik_lys_index is None:
        return None
    candidates = [
        index
        for index in range(
            record.vaik_lys_index + 5,
            min(len(record.sequence), record.vaik_lys_index + 36),
        )
        if record.sequence[index] == "E"
    ]
    return min(candidates, key=lambda index: abs(index - record.vaik_lys_index - 17)) if candidates else None


def correct_motif_mapping(records):
    return {
        split: [
            replace(record, alphac_glu_index=corrected_alphac_glu_index(record))
            for record in split_records
        ]
        for split, split_records in records.items()
    }


def stratified_indices(metadata: pd.DataFrame, validation_split: float, seed: int):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for _, group in metadata.groupby(["kinase", "label"]):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        if len(indices) < 2:
            train.extend(indices.tolist())
            continue
        count = min(len(indices) - 1, max(1, int(round(len(indices) * validation_split))))
        validation.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(validation))


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities >= 0.5
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities) if len(np.unique(labels)) == 2 else np.nan,
    }


def evaluate_model(model, features, labels, device) -> tuple[float, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        logits = model(features.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        probabilities = logits.softmax(-1)[:, 1].cpu().numpy()
    return float(loss), classification_metrics(labels.numpy(), probabilities)


def plot_training(history: pd.DataFrame, directory: Path) -> None:
    plt, _ = path_diagnostics.pyplot()
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.validation_loss, label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Training loss")
    axes[0].legend()
    axes[1].plot(history.epoch, history.train_balanced_accuracy, label="train")
    axes[1].plot(history.epoch, history.validation_balanced_accuracy, label="validation")
    axes[1].plot(history.epoch, history.validation_roc_auc, label="validation ROC AUC")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics", ylim=(0, 1.02))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=180)
    plt.close(fig)


def train_classifier(
    metadata: pd.DataFrame,
    local_features: np.ndarray,
    feature_names: Sequence[str],
    config: ComparatorConfig,
    training_dir: Path,
) -> tuple[StateMLP, dict[str, Any], pd.DataFrame]:
    training_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = training_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"
    last_path = checkpoint_dir / "last_model.pt"
    history_path = training_dir / "training_history.csv"
    config_path = training_dir / "training_config.json"
    device = diagnostics.device_name()
    diagnostics.set_seed(config.seed)

    train_indices, validation_indices = stratified_indices(
        metadata, config.validation_split, config.seed
    )
    mean = local_features[train_indices].mean(0)
    std = local_features[train_indices].std(0)
    std = np.where(std > 1e-6, std, 1.0)
    scaled = ((local_features - mean) / std).astype(np.float32)
    features = torch.tensor(scaled)
    labels = torch.tensor(metadata.label.to_numpy(), dtype=torch.long)
    train_dataset = TensorDataset(features[train_indices], labels[train_indices])
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = StateMLP(local_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    counts = np.bincount(labels[train_indices].numpy(), minlength=2)
    class_weights = torch.tensor(
        counts.sum() / np.maximum(counts, 1), dtype=torch.float32, device=device
    )
    start_epoch, best_score, best_epoch, stale = 1, -math.inf, 0, 0
    rows: list[dict[str, float]] = []
    if config.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        if history_path.exists():
            rows = pd.read_csv(history_path).to_dict("records")

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        losses = []
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_features), batch_labels, weight=class_weights)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        train_loss, train_metrics = evaluate_model(
            model, features[train_indices], labels[train_indices], device
        )
        validation_loss, validation_metrics = evaluate_model(
            model, features[validation_indices], labels[validation_indices], device
        )
        row = {
            "epoch": epoch,
            "batch_train_loss": float(np.mean(losses)),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        rows.append(row)
        score = validation_metrics["balanced_accuracy"] + 0.25 * validation_metrics["roc_auc"]
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": list(feature_names),
            "config": asdict(config),
        }
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            state["best_epoch"], state["best_score"] = best_epoch, best_score
            torch.save(state, best_path)
        else:
            stale += 1
        state["best_epoch"], state["best_score"] = best_epoch, best_score
        torch.save(state, last_path)
        pd.DataFrame(rows).to_csv(history_path, index=False)
        if epoch >= 30 and stale >= config.patience:
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    history = pd.DataFrame(rows)
    plot_training(history, training_dir / "figures")
    best_row = history.loc[history.epoch.eq(checkpoint["epoch"])].iloc[0]
    summary = {
        "total_structures": int(len(metadata)),
        "training_structures": int(len(train_indices)),
        "validation_structures": int(len(validation_indices)),
        "requested_epochs": int(config.epochs),
        "epochs_executed": int(history.epoch.max()),
        "best_epoch": int(checkpoint["epoch"]),
        "training_loss": float(best_row.train_loss),
        "validation_loss": float(best_row.validation_loss),
        "training_accuracy": float(best_row.train_accuracy),
        "validation_accuracy": float(best_row.validation_accuracy),
        "training_balanced_accuracy": float(best_row.train_balanced_accuracy),
        "validation_balanced_accuracy": float(best_row.validation_balanced_accuracy),
        "validation_roc_auc": float(best_row.validation_roc_auc),
        "device": device,
    }
    diagnostics.json_dump(summary, training_dir / "training_summary.json")
    diagnostics.json_dump(
        {
            **asdict(config),
            "feature_names": list(feature_names),
            "training_uses_real_structures_only": True,
            "temporal_prediction": False,
        },
        config_path,
    )
    metadata.assign(
        partition=np.where(
            metadata.index.isin(validation_indices), "validation", "training"
        )
    ).to_csv(training_dir / "dataset_partition.csv", index=False)
    return model, summary, checkpoint


def load_or_train(
    metadata: pd.DataFrame,
    local_features: np.ndarray,
    feature_names: Sequence[str],
    config: ComparatorConfig,
    training_dir: Path,
):
    best_path = training_dir / "checkpoints/best_model.pt"
    if best_path.exists() and not config.force_train and not config.resume:
        checkpoint = torch.load(
            best_path, map_location=diagnostics.device_name(), weights_only=False
        )
        model = StateMLP(local_features.shape[1]).to(diagnostics.device_name())
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        summary = json.loads((training_dir / "training_summary.json").read_text())
        return model, summary, checkpoint
    return train_classifier(metadata, local_features, feature_names, config, training_dir)


def model_probabilities(
    model: StateMLP, checkpoint: Mapping[str, Any], local_features: np.ndarray
) -> np.ndarray:
    mean = np.asarray(checkpoint["feature_mean"])
    std = np.asarray(checkpoint["feature_std"])
    features = torch.tensor((local_features - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        return (
            model(features.to(diagnostics.device_name()))
            .softmax(-1)[:, 1]
            .cpu()
            .numpy()
        )


def quality_score(
    record: diagnostics.StructureRecord,
    active_probability: float,
    config: ComparatorConfig,
) -> float:
    expected_probability = active_probability if record.state == "active" else 1 - active_probability
    motif_count = sum(
        value is not None
        for value in (
            record.dfg_index,
            record.alphac_glu_index,
            record.hrd_index,
            record.vaik_lys_index,
        )
    )
    resolution_score = 1 / record.resolution if record.resolution > 0 else 0.0
    missing_score = 1 / (1 + record.missing_internal_ids)
    ligand_score = record.ligand_present if config.prefer_ligand else 0
    return (
        1.5 * expected_probability
        + resolution_score
        + 0.5 * missing_score
        + 0.15 * motif_count
        + 0.35 * ligand_score
    )


def select_pair(
    records: Sequence[diagnostics.StructureRecord],
    probabilities: Mapping[str, float],
    config: ComparatorConfig,
):
    candidates = [record for record in records if record.kinase.upper() == config.kinase.upper()]
    if config.max_resolution is not None:
        candidates = [
            record
            for record in candidates
            if 0 < record.resolution <= config.max_resolution
        ]
    active = [record for record in candidates if record.state == "active"]
    inactive = [record for record in candidates if record.state == "inactive"]
    if not active or not inactive:
        raise ValueError(
            f"{config.kinase}: active={len(active)}, inactive={len(inactive)} after filters"
        )
    scored = []
    for active_record in active:
        for inactive_record in inactive:
            inactive_indices, active_indices, coverage, identity, mismatches = (
                diagnostics.align_sequences(inactive_record, active_record)
            )
            if len(active_indices) < 10:
                continue
            pair_quality = (
                quality_score(active_record, probabilities[active_record.pdb_id], config)
                + quality_score(inactive_record, probabilities[inactive_record.pdb_id], config)
                + 3 * coverage
                + 2 * identity
                - 0.01 * mismatches
            )
            scored.append(
                (
                    pair_quality,
                    active_record,
                    inactive_record,
                    inactive_indices,
                    active_indices,
                    coverage,
                    identity,
                )
            )
    if not scored:
        raise ValueError(f"{config.kinase}: no alignable active/inactive pair")
    return max(scored, key=lambda item: item[0])


def kabsch_to_reference(mobile: np.ndarray, reference: np.ndarray):
    mobile_center = mobile.mean(0)
    reference_center = reference.mean(0)
    centered_mobile = mobile - mobile_center
    centered_reference = reference - reference_center
    u, _, vh = np.linalg.svd(centered_mobile.T @ centered_reference)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return centered_mobile @ rotation + reference_center


def component_indices(record, aligned_indices, component: str) -> np.ndarray:
    aligned_map = {residue_index: output for output, residue_index in enumerate(aligned_indices)}
    source_indices: list[int] = []
    if component == "DFG" and record.dfg_index is not None:
        source_indices = list(range(record.dfg_index - 2, record.dfg_index + 5))
    elif component == "alphaC" and record.alphac_glu_index is not None:
        source_indices = list(range(record.alphac_glu_index - 7, record.alphac_glu_index + 8))
    elif component == "activation_loop" and record.dfg_index is not None:
        source_indices = list(range(record.dfg_index, record.dfg_index + 25))
    elif component == "HRD" and record.hrd_index is not None:
        source_indices = list(range(record.hrd_index - 2, record.hrd_index + 5))
    elif component == "Lys-Glu":
        source_indices = [
            index for index in (record.vaik_lys_index, record.alphac_glu_index) if index is not None
        ]
    return np.asarray([aligned_map[index] for index in source_indices if index in aligned_map], dtype=int)


def principal_axis(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.full(3, np.nan)
    _, _, vh = np.linalg.svd(coords - coords.mean(0), full_matrices=False)
    return vh[0]


def lys_glu_distance(record: diagnostics.StructureRecord) -> tuple[float, str]:
    if record.vaik_lys_index is None or record.alphac_glu_index is None:
        return np.nan, "not available"
    lysine = record.residues[record.vaik_lys_index]
    glutamate = record.residues[record.alphac_glu_index]
    if "NZ" in lysine.atoms:
        oxygen_atoms = [
            np.asarray(glutamate.atoms[name], dtype=float)
            for name in ("OE1", "OE2")
            if name in glutamate.atoms
        ]
        if oxygen_atoms:
            nz = np.asarray(lysine.atoms["NZ"], dtype=float)
            return min(float(np.linalg.norm(nz - oxygen)) for oxygen in oxygen_atoms), "NZ-OE side-chain"
    return float(
        np.linalg.norm(np.asarray(lysine.ca) - np.asarray(glutamate.ca))
    ), "CA proxy"


def compare_pair(selection, components: Sequence[str]):
    _, active, inactive, inactive_indices, active_indices, coverage, identity = selection
    active_coords = np.asarray([active.residues[index].ca for index in active_indices])
    inactive_coords = np.asarray([inactive.residues[index].ca for index in inactive_indices])
    aligned_inactive = kabsch_to_reference(inactive_coords, active_coords)
    displacement = np.linalg.norm(active_coords - aligned_inactive, axis=1)
    active_dist = classification.distance_matrix(active_coords)
    inactive_dist = classification.distance_matrix(aligned_inactive)
    global_metrics = {
        "global_rmsd": float(np.sqrt(np.mean(displacement**2))),
        "distance_matrix_error": float(np.sqrt(np.mean((active_dist - inactive_dist) ** 2))),
        "alignment_coverage": float(coverage),
        "sequence_identity": float(identity),
        "compared_residues": int(len(active_coords)),
    }
    rows = []
    for component in components:
        indices = component_indices(active, active_indices, component)
        if len(indices) == 0:
            rows.append({"component": component, "available": False, "notes": "not identified"})
            continue
        local_displacement = displacement[indices]
        active_local = active_coords[indices]
        inactive_local = aligned_inactive[indices]
        active_internal = classification.distance_matrix(active_local)
        inactive_internal = classification.distance_matrix(inactive_local)
        active_radius = np.linalg.norm(active_local - active_coords.mean(0), axis=1)
        inactive_radius = np.linalg.norm(inactive_local - aligned_inactive.mean(0), axis=1)
        axis_active, axis_inactive = principal_axis(active_local), principal_axis(inactive_local)
        orientation = (
            abs(float(np.dot(axis_active, axis_inactive)))
            if np.isfinite(axis_active).all() and np.isfinite(axis_inactive).all()
            else np.nan
        )
        largest = indices[np.argsort(local_displacement)[::-1][: min(5, len(indices))]]
        residue_ids = [
            f"{active.residues[active_indices[index]].name3}{active.residues[active_indices[index]].number}"
            for index in largest
        ]
        row = {
            "component": component,
            "available": True,
            "residue_count": len(indices),
            "local_rmsd": float(np.sqrt(np.mean(local_displacement**2))),
            "mean_displacement": float(local_displacement.mean()),
            "max_displacement": float(local_displacement.max()),
            "largest_change_residues": ";".join(residue_ids),
            "internal_distance_change": float(
                np.sqrt(np.mean((active_internal - inactive_internal) ** 2))
            ),
            "center_of_mass_radial_change": float(np.mean(active_radius - inactive_radius)),
            "orientation_axis_cosine": orientation,
            "notes": "sequence motif mapping",
        }
        if component == "Lys-Glu" and len(indices) == 2:
            active_lys_glu, active_method = lys_glu_distance(active)
            inactive_lys_glu, inactive_method = lys_glu_distance(inactive)
            row.update(
                {
                    "active_lys_glu_distance": active_lys_glu,
                    "inactive_lys_glu_distance": inactive_lys_glu,
                    "lys_glu_difference": active_lys_glu - inactive_lys_glu,
                    "notes": (
                        active_method
                        if active_method == inactive_method
                        else f"active={active_method}; inactive={inactive_method}"
                    ),
                }
            )
        rows.append(row)
    residue_rows = []
    region_membership = {
        component: set(component_indices(active, active_indices, component).tolist())
        for component in components
    }
    for output_index, value in enumerate(displacement):
        residue = active.residues[active_indices[output_index]]
        labels = [component for component, members in region_membership.items() if output_index in members]
        residue_rows.append(
            {
                "aligned_index": output_index,
                "active_residue_id": f"{residue.chain}:{residue.name3}{residue.number}{residue.insertion}",
                "displacement": value,
                "components": ";".join(labels) if labels else "other",
            }
        )
    return (
        active,
        inactive,
        np.asarray(active_indices),
        np.asarray(inactive_indices),
        active_coords,
        aligned_inactive,
        active_dist - inactive_dist,
        global_metrics,
        pd.DataFrame(rows),
        pd.DataFrame(residue_rows),
    )


def write_ca_models(path: Path, active, inactive, active_indices, inactive_indices, active_coords, inactive_coords):
    lines = ["TITLE     ACTIVE/INACTIVE ALIGNED CA COMPARISON", "REMARK 900 STATIC ENDPOINT COMPARISON"]
    for model_id, (record, indices, coords, label) in enumerate(
        (
            (active, active_indices, active_coords, "ACTIVE"),
            (inactive, inactive_indices, inactive_coords, "INACTIVE_ALIGNED"),
        ),
        start=1,
    ):
        lines.extend([f"MODEL     {model_id:4d}", f"REMARK 900 STATE {label}"])
        for serial, (residue_index, xyz) in enumerate(zip(indices, coords), start=1):
            residue = record.residues[int(residue_index)]
            chain = residue.chain if residue.chain != "_" else "A"
            lines.append(
                f"ATOM  {serial:5d}  CA  {residue.name3:>3s} {chain[:1]}{residue.number:4d}{(residue.insertion or ' ')[:1]}"
                f"   {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           C"
            )
        lines.extend(["TER", "ENDMDL"])
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def residue_selection(record, component: str) -> str | None:
    ranges = {
        "DFG": (record.dfg_index, -2, 4),
        "alphaC": (record.alphac_glu_index, -7, 7),
        "activation_loop": (record.dfg_index, 0, 24),
        "HRD": (record.hrd_index, -2, 4),
    }
    if component in ranges:
        center, left, right = ranges[component]
        if center is None:
            return None
        residues = record.residues[max(0, center + left) : min(len(record.residues), center + right + 1)]
    elif component == "Lys-Glu":
        indices = [value for value in (record.vaik_lys_index, record.alphac_glu_index) if value is not None]
        residues = [record.residues[index] for index in indices]
    else:
        return None
    return "+".join(f"{residue.number}{residue.insertion}" for residue in residues)


def write_viewer_scripts(directory: Path, active, inactive, components: Sequence[str]) -> None:
    pymol = [
        "reinitialize",
        "load active_selected.pdb, active",
        "load inactive_selected.pdb, inactive",
        "remove solvent",
        "align inactive, active",
        "hide everything",
        "show cartoon, active or inactive",
        "color green, active",
        "color gray70, inactive",
    ]
    chimerax = [
        "close",
        "open active_selected.pdb",
        "open inactive_selected.pdb",
        "matchmaker #2 to #1",
        "cartoon",
        "color #1 green",
        "color #2 gray",
    ]
    colors = {"DFG": "magenta", "alphaC": "cyan", "activation_loop": "orange", "HRD": "yellow", "Lys-Glu": "red"}
    for component in components:
        active_selection = residue_selection(active, component)
        inactive_selection = residue_selection(inactive, component)
        if not active_selection or not inactive_selection:
            continue
        object_name = component.lower().replace("-", "_")
        pymol.extend(
            [
                f"select {object_name}_active, active and resi {active_selection}",
                f"select {object_name}_inactive, inactive and resi {inactive_selection}",
                f"show sticks, {object_name}_active or {object_name}_inactive",
                f"color {colors[component]}, {object_name}_active or {object_name}_inactive",
            ]
        )
        active_chimerax = active_selection.replace("+", ",")
        inactive_chimerax = inactive_selection.replace("+", ",")
        chimerax.extend(
            [
                f"style #1:{active_chimerax} stick",
                f"style #2:{inactive_chimerax} stick",
                f"color #1:{active_chimerax} {colors[component]}",
                f"color #2:{inactive_chimerax} {colors[component]}",
            ]
        )
    pymol.extend(
        [
            "orient active or inactive",
            "zoom active or inactive",
            "bg_color white",
            "ray 1600, 1200",
            "png active_inactive_pymol.png, dpi=200",
        ]
    )
    chimerax.extend(["view", "set bgColor white", "save active_inactive_chimerax.png width 1600 height 1200"])
    (directory / "view_active_inactive.pml").write_text("\n".join(pymol) + "\n")
    (directory / "view_active_inactive.cxc").write_text("\n".join(chimerax) + "\n")


def plot_comparison(
    output: Path,
    active_coords: np.ndarray,
    inactive_coords: np.ndarray,
    difference_matrix: np.ndarray,
    component_metrics: pd.DataFrame,
    residue_metrics: pd.DataFrame,
    active_record,
    active_indices,
    components,
) -> None:
    plt, sns = path_diagnostics.pyplot()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(*active_coords.T, color="green", alpha=0.75, label="Active")
    axis.plot(*inactive_coords.T, color="gray", alpha=0.75, label="Inactive aligned")
    axis.legend()
    axis.set_title("Static active/inactive endpoint overlay")
    fig.tight_layout()
    fig.savefig(figures / "active_inactive_overlay_3d.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 5))
    sns.lineplot(data=residue_metrics, x="aligned_index", y="displacement", ax=axis)
    axis.set(title="C-alpha displacement by aligned residue", ylabel="Displacement (A)")
    fig.tight_layout()
    fig.savefig(figures / "per_residue_displacement.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(difference_matrix, cmap="coolwarm", center=0, ax=axis)
    axis.set_title("Active - inactive internal distance matrix")
    fig.tight_layout()
    fig.savefig(figures / "distance_matrix_difference.png", dpi=180)
    plt.close(fig)

    available = component_metrics[component_metrics.available.eq(True)].copy()
    if not available.empty:
        fig, axis = plt.subplots(figsize=(10, max(2.5, 0.55 * len(available) + 1)))
        axis.axis("off")
        columns = ["component", "local_rmsd", "mean_displacement", "max_displacement"]
        axis.table(
            cellText=available[columns].round(3).values,
            colLabels=columns,
            loc="center",
        )
        fig.tight_layout()
        fig.savefig(figures / "component_metrics_table.png", dpi=180)
        plt.close(fig)

    for component in components:
        indices = component_indices(active_record, active_indices, component)
        if len(indices) == 0:
            continue
        fig = plt.figure(figsize=(8, 6))
        axis = fig.add_subplot(111, projection="3d")
        axis.scatter(*active_coords[indices].T, color="green", label="Active", s=35)
        axis.scatter(*inactive_coords[indices].T, color="gray", label="Inactive", s=35)
        axis.set_title(f"{component}: active vs inactive")
        axis.legend()
        fig.tight_layout()
        safe = component.lower().replace("-", "_")
        fig.savefig(figures / f"{safe}_comparison.png", dpi=180)
        plt.close(fig)


def selected_metadata(record, probability: float) -> dict[str, Any]:
    return {
        "pdb_id": record.pdb_id,
        "state": record.state,
        "resolution": record.resolution,
        "ligand_present": record.ligand_present,
        "dfg_state": record.dfg_state,
        "alphac_state": record.alphac_state,
        "chain": record.chain,
        "residues": len(record.residues),
        "missing_internal_ids": record.missing_internal_ids,
        "model_active_probability": probability,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No components identified."
    columns = [
        "component",
        "residue_count",
        "local_rmsd",
        "mean_displacement",
        "max_displacement",
        "internal_distance_change",
        "orientation_axis_cosine",
    ]
    selected = frame[[column for column in columns if column in frame.columns]].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def write_report(
    output: Path,
    config: ComparatorConfig,
    active,
    inactive,
    probabilities,
    global_metrics,
    component_metrics,
    training_summary,
) -> None:
    available = component_metrics[component_metrics.available.eq(True)].copy()
    largest = (
        available.sort_values("local_rmsd", ascending=False).iloc[0].component
        if not available.empty
        else "not available"
    )
    stable = (
        available.sort_values("local_rmsd").iloc[0].component
        if not available.empty
        else "not available"
    )
    dfg = available[available.component.eq("DFG")]
    alphac = available[available.component.eq("alphaC")]
    aloop = available[available.component.eq("activation_loop")]
    lysglu = available[available.component.eq("Lys-Glu")]

    def movement(frame):
        return f"{frame.iloc[0].local_rmsd:.3f} A local RMSD" if not frame.empty else "not identified"

    lys_text = "not identified"
    if not lysglu.empty and pd.notna(lysglu.iloc[0].get("lys_glu_difference")):
        lys_row = lysglu.iloc[0]
        lys_text = (
            f"active={lys_row.active_lys_glu_distance:.3f} A, "
            f"inactive={lys_row.inactive_lys_glu_distance:.3f} A, "
            f"difference={lys_row.lys_glu_difference:+.3f} A "
            f"({lys_row.notes})"
        )

    rows = markdown_table(available)
    active_meta = selected_metadata(active, probabilities[active.pdb_id])
    inactive_meta = selected_metadata(inactive, probabilities[inactive.pdb_id])
    text = f"""# {config.kinase} Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `{active.pdb_id}`, resolution {active.resolution:.2f} A, ligand={active.ligand_present}, DFG={active.dfg_state}, alphaC={active.alphac_state}.
- Inactive: `{inactive.pdb_id}`, resolution {inactive.resolution:.2f} A, ligand={inactive.ligand_present}, DFG={inactive.dfg_state}, alphaC={inactive.alphac_state}.
- Alignment coverage: {global_metrics['alignment_coverage']:.3f}; sequence identity: {global_metrics['sequence_identity']:.3f}.
- Compared residues: {global_metrics['compared_residues']}.

## Global comparison

- Global C-alpha RMSD: **{global_metrics['global_rmsd']:.3f} A**.
- Distance-matrix error: **{global_metrics['distance_matrix_error']:.3f} A**.

## Functional components

{rows}

## Direct answers

- DFG change: {movement(dfg)}. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: {movement(alphac)}.
- Activation-loop change: {movement(aloop)}.
- Lys-Glu: {lys_text}.
- Largest local change: **{largest}**.
- Most stable requested component: **{stable}**.

## Model training

- Real structures used: {training_summary['total_structures']}.
- Training/validation: {training_summary['training_structures']} / {training_summary['validation_structures']}.
- Epochs executed: {training_summary['epochs_executed']}; best epoch: {training_summary['best_epoch']}.
- Best validation accuracy: {training_summary['validation_accuracy']:.3f}.
- Best validation balanced accuracy: {training_summary['validation_balanced_accuracy']:.3f}.
- Best validation ROC AUC: {training_summary['validation_roc_auc']:.3f}.

The model contributes a state-consistency score to representative selection.
All reported structural differences are computed directly from the selected
real PDB endpoints after sequence alignment and rigid-body superposition.

## Files

- [Active PDB](active_selected.pdb)
- [Inactive PDB](inactive_selected.pdb)
- [Aligned two-model PDB](active_inactive_aligned.pdb)
- [PyMOL script](view_active_inactive.pml)
- [ChimeraX script](view_active_inactive.cxc)
- [Figures](figures/)

## Mapping limitations

No explicit KLIFS-position table is present in the current repository.
Components were mapped from sequence motifs and residue identifiers: DFG and
HRD by exact motif, alphaC from the conserved Glu upstream of VAIK, activation
loop from DFG onward, and Lys-Glu from the detected VAIK Lys/alphaC Glu pair.
Missing components are reported rather than imputed as biological truth.
"""
    (output / "state_comparison_report.md").write_text(text, encoding="utf-8")
    diagnostics.json_dump(
        {"active": active_meta, "inactive": inactive_meta, "global_metrics": global_metrics},
        output / "selection_summary.json",
    )


def analyze_kinase(
    config: ComparatorConfig,
    records,
    metadata,
    probabilities_by_pdb,
    training_summary,
    output_root: Path,
) -> dict[str, Any]:
    selection = select_pair(records, probabilities_by_pdb, config)
    comparison = compare_pair(selection, config.components)
    (
        active,
        inactive,
        active_indices,
        inactive_indices,
        active_coords,
        aligned_inactive,
        difference_matrix,
        global_metrics,
        component_metrics,
        residue_metrics,
    ) = comparison
    output = output_root / config.kinase.upper()
    output.mkdir(parents=True, exist_ok=True)
    component_metrics.to_csv(output / "component_metrics.csv", index=False)
    residue_metrics.to_csv(output / "per_residue_metrics.csv", index=False)
    pd.DataFrame([global_metrics]).to_csv(output / "global_metrics.csv", index=False)
    shutil.copyfile(ROOT / "data/raw/pdbs" / f"{active.pdb_id}.pdb", output / "active_selected.pdb")
    shutil.copyfile(ROOT / "data/raw/pdbs" / f"{inactive.pdb_id}.pdb", output / "inactive_selected.pdb")
    write_ca_models(
        output / "active_inactive_aligned.pdb",
        active,
        inactive,
        active_indices,
        inactive_indices,
        active_coords,
        aligned_inactive,
    )
    write_viewer_scripts(output, active, inactive, config.components)
    plot_comparison(
        output,
        active_coords,
        aligned_inactive,
        difference_matrix,
        component_metrics,
        residue_metrics,
        active,
        active_indices,
        config.components,
    )
    write_report(
        output,
        config,
        active,
        inactive,
        probabilities_by_pdb,
        global_metrics,
        component_metrics,
        training_summary,
    )
    component_lookup = component_metrics.set_index("component")

    def value(component, column):
        if component not in component_lookup.index:
            return np.nan
        result = component_lookup.loc[component].get(column, np.nan)
        return result if pd.notna(result) else np.nan

    available = component_metrics[component_metrics.available.eq(True)]
    return {
        "kinase": config.kinase,
        "active_pdb": active.pdb_id,
        "inactive_pdb": inactive.pdb_id,
        "global_rmsd": global_metrics["global_rmsd"],
        "DFG_rmsd": value("DFG", "local_rmsd"),
        "alphaC_rmsd": value("alphaC", "local_rmsd"),
        "activation_loop_rmsd": value("activation_loop", "local_rmsd"),
        "Lys_Glu_difference": value("Lys-Glu", "lys_glu_difference"),
        "largest_change_component": (
            available.sort_values("local_rmsd", ascending=False).iloc[0].component
            if not available.empty
            else ""
        ),
        "status": "completed",
    }


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
        loader = getattr(yaml, "safe_load", None)
    except ImportError:
        loader = None
    text = path.read_text(encoding="utf-8")
    value = loader(text) if loader else simple_yaml_load(text)
    value = value or {}
    training = value.pop("training", {}) or {}
    return {**value, **training}


def simple_yaml_load(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_mapping = result
    current_list: list[Any] | None = None

    def scalar(raw: str):
        raw = raw.strip()
        if not raw:
            return {}
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            return raw[1:-1]
        if raw.lower() in {"true", "false"}:
            return raw.lower() == "true"
        if raw.lower() in {"null", "none"}:
            return None
        try:
            return float(raw) if any(token in raw.lower() for token in (".", "e")) else int(raw)
        except ValueError:
            return raw

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        line = raw_line.strip()
        if line.startswith("- "):
            if current_list is None:
                raise ValueError(f"List item without a list key: {line}")
            current_list.append(scalar(line[2:]))
            continue
        key, raw_value = line.split(":", 1)
        if indent == 0:
            value = scalar(raw_value)
            result[key] = value
            current_mapping = value if isinstance(value, dict) else result
            current_list = None
            if not raw_value.strip():
                next_is_list = key in {"components", "states"}
                result[key] = [] if next_is_list else {}
                current_mapping = result[key] if isinstance(result[key], dict) else result
                current_list = result[key] if isinstance(result[key], list) else None
        else:
            value = scalar(raw_value)
            current_mapping[key] = value
            current_list = None
    return result


def build_config(args: argparse.Namespace) -> ComparatorConfig:
    values: dict[str, Any] = {}
    if args.config:
        values.update(read_yaml(Path(args.config)))
    cli = vars(args)
    for key in asdict(ComparatorConfig()):
        if key in cli and cli[key] is not None:
            values[key] = cli[key]
    if "components" in values:
        values["components"] = tuple(normalize_component(item) for item in values["components"])
    if "states" in values:
        values["states"] = tuple(values["states"])
    return ComparatorConfig(**{key: value for key, value in values.items() if key in asdict(ComparatorConfig())})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--kinase")
    parser.add_argument("--components", nargs="+")
    parser.add_argument("--max-resolution", type=float)
    parser.add_argument("--prefer-ligand", action="store_true", default=None)
    parser.add_argument("--use-full-dataset", action="store_true", default=None)
    parser.add_argument("--all-kinases", action="store_true", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--validation-split", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--force-train", action="store_true", default=None)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = build_config(parse_args())
    if not config.kinase and not config.all_kinases:
        raise SystemExit("Provide --kinase NAME or --all-kinases")
    output_root = ROOT / "outputs/kinase_state_comparator"
    output_root.mkdir(parents=True, exist_ok=True)
    records_by_split, audit = diagnostics.load_records(ROOT)
    records_by_split = correct_motif_mapping(records_by_split)
    records = flatten_records(records_by_split)
    metadata, features, local_names = classification.extract_features(records_by_split)
    audit.to_csv(output_root / "structure_loading_audit.csv", index=False)
    metadata.to_csv(output_root / "valid_structure_index.csv", index=False)
    model, training_summary, checkpoint = load_or_train(
        metadata,
        features["local_geometry"],
        local_names,
        config,
        output_root / "training",
    )
    probabilities = model_probabilities(model, checkpoint, features["local_geometry"])
    probability_by_pdb = dict(zip(metadata.pdb_id, probabilities))
    pd.DataFrame(
        {
            "pdb_id": metadata.pdb_id,
            "kinase": metadata.kinase,
            "true_state": metadata.state,
            "active_probability": probabilities,
        }
    ).to_csv(output_root / "training/state_probabilities.csv", index=False)

    if config.all_kinases:
        by_kinase = metadata.groupby(["kinase", "state"]).size().unstack(fill_value=0)
        kinases = [
            kinase for kinase, row in by_kinase.iterrows()
            if row.get("active", 0) > 0 and row.get("inactive", 0) > 0
        ]
    else:
        kinases = [config.kinase]
    rows = []
    for kinase in kinases:
        kinase_config = ComparatorConfig(**{**asdict(config), "kinase": kinase})
        try:
            rows.append(
                analyze_kinase(
                    kinase_config,
                    records,
                    metadata,
                    probability_by_pdb,
                    training_summary,
                    output_root,
                )
            )
            LOG.info("Completed %s", kinase)
        except Exception as exc:
            LOG.exception("Failed %s", kinase)
            rows.append({"kinase": kinase, "status": f"failed: {exc}"})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "global_summary.csv", index=False)
    diagnostics.json_dump(asdict(config), output_root / "run_config.json")
    return 0 if summary.status.eq("completed").any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
