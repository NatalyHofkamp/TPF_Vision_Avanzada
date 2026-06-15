#!/usr/bin/env python3
"""Failure-mode analysis for Fold 1 translation quality.

This script keeps the scientific setup intact and only measures:
- identity baseline
- FoldFlow conditional / unconditional / CFG sweeps
- structural validity of sampled translations
- source-target difficulty distribution in the test set
- a small intra-kinase control experiment
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from kinase_data.translation import InactiveActiveTranslationDataset, translation_collate_fn
from models.foldflow_backbone import FoldFlowBackbone
from training.fold1_experiment import (
    Fold1TrainingConfig,
    build_model_and_loaders,
    build_optimizer,
    parameter_trainability_report,
    rmsd_from_coordinates,
    run_epoch,
    structure_ca_coordinates,
    summarize_test_metrics,
)


@dataclass(frozen=True)
class AnalysisConfig:
    fold_id: int = 1
    guidance_scales: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0)
    reverse_steps: int = 6
    source_min_t: float = 0.0
    checkpoint_path: Path = Path("checkpoints/fold1/best_validation.pt")
    report_root: Path = Path("reports/fold1/failure_modes")
    figure_root: Path = Path("figures/fold1/failure_modes")
    same_kinase: str = "EGFR"
    same_kinase_epochs: int = 1
    same_kinase_batch_size: int = 1
    same_kinase_learning_rate: float = 1e-4
    same_kinase_val_fraction: float = 0.15
    same_kinase_train_fraction: float = 0.7
    same_kinase_seed: int = 7
    same_kinase_max_samples: int | None = 24
    max_guidance_samples: int | None = None
    min_success_improvement: float = 0.0


class SampleSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Iterable[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _identity_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    source_ca = structure_ca_coordinates(sample["source_structure"])
    target_ca = structure_ca_coordinates(sample["target_structure"])
    rmsd_source_target = rmsd_from_coordinates(target_ca, source_ca)
    return {
        "kinase": sample["kinase"],
        "source_pdb_id": sample["source_pdb_id"],
        "target_pdb_id": sample["target_pdb_id"],
        "residue_count": sample["residue_count"],
        "sequence_identity": sample["sequence_identity"],
        "rmsd_source_target": rmsd_source_target,
        "rmsd_identity_target": rmsd_source_target,
        "rmsd_improvement": 0.0,
        "successful_translation": False,
        "baseline": "identity",
    }


def _prediction_metrics(sample: dict[str, Any], prediction: dict[str, Any], label: str) -> dict[str, Any]:
    source_ca = structure_ca_coordinates(sample["source_structure"])
    target_ca = structure_ca_coordinates(sample["target_structure"])
    predicted_ca = prediction["rigids"][0, :, 4:7].detach().cpu().numpy()
    rmsd_source_target = rmsd_from_coordinates(target_ca, source_ca)
    rmsd_prediction_target = rmsd_from_coordinates(target_ca, predicted_ca)
    source_prediction_rmsd = rmsd_from_coordinates(source_ca, predicted_ca)
    displacement = predicted_ca - source_ca
    vectorfield_magnitude = 0.5 * (
        prediction["model_output"]["rot_vectorfield"].norm(dim=-1).mean().item()
        + prediction["model_output"]["trans_vectorfield"].norm(dim=-1).mean().item()
    )
    conditional_unconditional_difference = 0.5 * (
        (
            prediction["model_output"]["conditional"]["rot_vectorfield"]
            - prediction["model_output"]["unconditional"]["rot_vectorfield"]
        )
        .abs()
        .mean()
        .item()
        + (
            prediction["model_output"]["conditional"]["trans_vectorfield"]
            - prediction["model_output"]["unconditional"]["trans_vectorfield"]
        )
        .abs()
        .mean()
        .item()
    )
    return {
        "kinase": sample["kinase"],
        "source_pdb_id": sample["source_pdb_id"],
        "target_pdb_id": sample["target_pdb_id"],
        "residue_count": sample["residue_count"],
        "sequence_identity": sample["sequence_identity"],
        "rmsd_source_target": rmsd_source_target,
        "rmsd_prediction_target": rmsd_prediction_target,
        "rmsd_source_prediction": source_prediction_rmsd,
        "rmsd_improvement": rmsd_source_target - rmsd_prediction_target,
        "successful_translation": rmsd_prediction_target < rmsd_source_target,
        "mean_displacement_per_residue": float(np.linalg.norm(displacement, axis=-1).mean()),
        "coordinate_mean": float(predicted_ca.mean()),
        "coordinate_std": float(predicted_ca.std()),
        "coordinate_min": float(predicted_ca.min()),
        "coordinate_max": float(predicted_ca.max()),
        "has_nan": bool(np.isnan(predicted_ca).any()),
        "vectorfield_magnitude": float(vectorfield_magnitude),
        "prediction_difference": float(conditional_unconditional_difference),
        "baseline": label,
    }


def _build_batch(sample: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch = translation_collate_fn([sample])
    foldflow_features = {key: value.to(device) for key, value in batch["foldflow_features"].items()}
    conditioning = {key: value.to(device) for key, value in batch["conditioning"].items()}
    return foldflow_features, conditioning


def _sample_with_guidance(
    model: FoldFlowBackbone,
    sample: dict[str, Any],
    device: torch.device,
    guidance_scale: float,
    reverse_steps: int,
    source_min_t: float,
) -> dict[str, Any]:
    batch = translation_collate_fn([sample])
    foldflow_features = {key: value.to(device) for key, value in batch["foldflow_features"].items()}
    conditioning = {key: value.to(device) for key, value in batch["conditioning"].items()}
    with torch.no_grad():
        return model.sample_translation(
            foldflow_features=foldflow_features,
            conditioning=conditioning,
            guidance_scale=guidance_scale,
            num_steps=reverse_steps,
            min_t=source_min_t,
            noise_scale=0.0,
            center=True,
        )


def _summarize_coordinate_distribution(coordinates: list[np.ndarray]) -> dict[str, float]:
    if not coordinates:
        return {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "z_min": 0.0,
            "z_max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "finite": True,
        }
    coords = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1, 3) for item in coordinates], axis=0)
    return {
        "x_min": float(coords[:, 0].min()),
        "x_max": float(coords[:, 0].max()),
        "y_min": float(coords[:, 1].min()),
        "y_max": float(coords[:, 1].max()),
        "z_min": float(coords[:, 2].min()),
        "z_max": float(coords[:, 2].max()),
        "mean": float(coords.mean()),
        "std": float(coords.std()),
        "finite": bool(np.isfinite(coords).all()),
    }


def _plot_histogram(values: pd.Series, title: str, output_path: Path, bins: int = 20) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_guidance_sweep(frame: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["guidance_scale"], frame["mean_rmsd_prediction_target"], marker="o")
    axes[0].plot(frame["guidance_scale"], frame["mean_rmsd_source_prediction"], marker="o")
    axes[0].set_xlabel("guidance_scale")
    axes[0].set_ylabel("RMSD (Å)")
    axes[0].set_title("Prediction stability vs guidance")
    axes[0].legend(["prediction→target", "source→prediction"])

    axes[1].plot(frame["guidance_scale"], frame["average_vectorfield_magnitude"], marker="o")
    axes[1].plot(frame["guidance_scale"], frame["average_prediction_difference"], marker="o")
    axes[1].set_xlabel("guidance_scale")
    axes[1].set_ylabel("Magnitude / difference")
    axes[1].set_title("CFG response")
    axes[1].legend(["vector-field magnitude", "cond-uncond diff"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _select_intra_kinase_indices(dataset: Dataset, kinase: str) -> list[int]:
    return [index for index in range(len(dataset)) if dataset[index]["kinase"] == kinase]


def _split_indices(indices: list[int], train_fraction: float, val_fraction: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0 or not 0.0 <= val_fraction < 1.0:
        raise ValueError("Fractions must be in (0,1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")
    rng = np.random.default_rng(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * train_fraction)))
    n_val = max(1, int(round(n_total * val_fraction)))
    if n_train + n_val >= n_total:
        n_val = max(1, min(n_val, n_total - n_train - 1))
    n_test = n_total - n_train - n_val
    train_indices = shuffled[:n_train]
    val_indices = shuffled[n_train:n_train + n_val]
    test_indices = shuffled[n_train + n_val:]
    if not test_indices:
        test_indices = [train_indices.pop()]
    return train_indices, val_indices, test_indices


def _build_intra_kinase_loaders(
    dataset: Dataset,
    kinase: str,
    batch_size: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    indices = _select_intra_kinase_indices(dataset, kinase)
    if len(indices) < 3:
        raise ValueError(f"Not enough samples for intra-kinase split: {kinase}")
    train_indices, val_indices, test_indices = _split_indices(
        indices, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed
    )
    train_subset = SampleSubset(dataset, train_indices)
    val_subset = SampleSubset(dataset, val_indices)
    test_subset = SampleSubset(dataset, test_indices)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, collate_fn=translation_collate_fn)
    train_loader = DataLoader(train_subset, shuffle=True, persistent_workers=num_workers > 0, **common)
    val_loader = DataLoader(val_subset, shuffle=False, persistent_workers=num_workers > 0, **common)
    test_loader = DataLoader(test_subset, shuffle=False, persistent_workers=num_workers > 0, **common)
    return train_loader, val_loader, test_loader, {
        "train": len(train_subset),
        "validation": len(val_subset),
        "test": len(test_subset),
    }


def _summarize_sampling_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "number_of_samples": int(len(frame)),
        "mean_rmsd_source_target": float(frame["rmsd_source_target"].mean()),
        "mean_rmsd_prediction_target": float(frame["rmsd_prediction_target"].mean()),
        "mean_rmsd_source_prediction": float(frame["rmsd_source_prediction"].mean()),
        "mean_rmsd_improvement": float(frame["rmsd_improvement"].mean()),
        "median_rmsd_improvement": float(frame["rmsd_improvement"].median()),
        "successful_translation_rate": float(frame["successful_translation"].mean()),
        "median_source_target_rmsd": float(frame["rmsd_source_target"].median()),
        "source_target_below_2A": int((frame["rmsd_source_target"] < 2.0).sum()),
        "source_target_below_3A": int((frame["rmsd_source_target"] < 3.0).sum()),
        "source_target_at_least_5A": int((frame["rmsd_source_target"] >= 5.0).sum()),
        "has_nan_fraction": float(frame["has_nan"].mean()),
        "average_vectorfield_magnitude": float(frame["vectorfield_magnitude"].mean()),
        "average_prediction_difference": float(frame["prediction_difference"].mean()),
    }


def _collect_test_samples(loader: DataLoader) -> list[dict[str, Any]]:
    return [loader.dataset[index] for index in range(len(loader.dataset))]


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    set_seed(config.same_kinase_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = Fold1TrainingConfig(
        fold_id=config.fold_id,
        epochs=1,
        batch_size=1,
        learning_rate=config.same_kinase_learning_rate,
        patience=1,
        use_amp=False,
        reverse_steps=config.reverse_steps,
        reverse_min_t=config.source_min_t,
    )
    model, train_loader, validation_loader, test_loader, fold_reports = build_model_and_loaders(
        model_config, device
    )
    checkpoint = torch.load(config.checkpoint_path, map_location=device, weights_only=False)
    model.condition_encoder.load_state_dict(checkpoint["conditioning_state"])
    model.eval()

    report_root = config.report_root
    figure_root = config.figure_root
    report_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    samples = _collect_test_samples(test_loader)
    identity_rows = [_identity_metrics(sample) for sample in samples]
    identity_frame = pd.DataFrame(identity_rows)
    identity_path = report_root / "identity_baseline.csv"
    identity_frame.to_csv(identity_path, index=False)

    source_target_frame = pd.DataFrame(
        [
            {
                "kinase": sample["kinase"],
                "source_pdb_id": sample["source_pdb_id"],
                "target_pdb_id": sample["target_pdb_id"],
                "residue_count": sample["residue_count"],
                "sequence_identity": sample["sequence_identity"],
                "rmsd_source_target": rmsd_from_coordinates(
                    structure_ca_coordinates(sample["target_structure"]),
                    structure_ca_coordinates(sample["source_structure"]),
                ),
            }
            for sample in samples
        ]
    )
    source_target_path = report_root / "source_target_distribution.csv"
    source_target_frame.to_csv(source_target_path, index=False)
    hist_path = _plot_histogram(
        source_target_frame["rmsd_source_target"],
        "Fold 1 test-set source→target RMSD",
        figure_root / "test_source_target_rmsd_hist.png",
    )

    guidance_reports: dict[str, list[dict[str, Any]]] = {str(scale): [] for scale in config.guidance_scales}
    sampling_examples: dict[str, dict[str, Any]] = {}
    coordinate_samples: list[np.ndarray] = []
    guidance_samples = samples if config.max_guidance_samples is None else samples[: config.max_guidance_samples]
    for sample_index, sample in enumerate(guidance_samples):
        for scale in config.guidance_scales:
            prediction = _sample_with_guidance(
                model,
                sample,
                device,
                guidance_scale=scale,
                reverse_steps=config.reverse_steps,
                source_min_t=config.source_min_t,
            )
            row = _prediction_metrics(sample, prediction, label=f"guidance_{scale}")
            row["sample_index"] = sample_index
            guidance_reports[str(scale)].append(row)
            if sample_index == 0:
                sampling_examples[str(scale)] = {
                    "sample": sample,
                    "prediction": prediction,
                    "metrics": row,
                }
            if float(scale) == 2.0:
                coordinate_samples.append(prediction["rigids"][0, :, 4:7].detach().cpu().numpy())

    guidance_summary_rows = []
    for scale, rows in guidance_reports.items():
        frame = pd.DataFrame(rows)
        frame.to_csv(report_root / f"guidance_{scale}.csv", index=False)
        guidance_summary_rows.append(
            {
                "guidance_scale": float(scale),
                "mean_rmsd_source_target": float(frame["rmsd_source_target"].mean()),
                "mean_rmsd_prediction_target": float(frame["rmsd_prediction_target"].mean()),
                "mean_rmsd_source_prediction": float(frame["rmsd_source_prediction"].mean()),
                "mean_rmsd_improvement": float(frame["rmsd_improvement"].mean()),
                "median_rmsd_improvement": float(frame["rmsd_improvement"].median()),
                "average_vectorfield_magnitude": float(frame["vectorfield_magnitude"].mean()),
                "average_prediction_difference": float(frame["prediction_difference"].mean()),
                "sample_count": int(len(frame)),
                "successful_translation_rate": float(frame["successful_translation"].mean()),
                "has_nan_fraction": float(frame["has_nan"].mean()),
            }
        )
    guidance_summary = pd.DataFrame(guidance_summary_rows).sort_values("guidance_scale")
    guidance_summary_path = report_root / "guidance_summary.csv"
    guidance_summary.to_csv(guidance_summary_path, index=False)
    guidance_plot_path = _plot_guidance_sweep(
        guidance_summary, figure_root / "guidance_sweep.png"
    )

    # Structural validity summary from sampled predictions at guidance=2.0 if available,
    # otherwise from the first guidance scale.
    validation_scale = "2.0" if "2.0" in guidance_reports else str(config.guidance_scales[0])
    sampling_frame = pd.DataFrame(guidance_reports[validation_scale])
    sampling_summary = _summarize_sampling_metrics(sampling_frame)
    coordinate_summary = (
        _summarize_coordinate_distribution(coordinate_samples)
        if coordinate_samples
        else {"finite": True, "mean": 0.0, "std": 0.0, "x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0, "z_min": 0.0, "z_max": 0.0}
    )

    same_dataset = InactiveActiveTranslationDataset(
        fold_id=config.fold_id,
        split="test",
        load_structures=True,
    )
    same_indices = _select_intra_kinase_indices(same_dataset, config.same_kinase)
    if config.same_kinase_max_samples is not None:
        same_indices = same_indices[: config.same_kinase_max_samples]
    same_train, same_val, same_test = _split_indices(
        same_indices,
        train_fraction=config.same_kinase_train_fraction,
        val_fraction=config.same_kinase_val_fraction,
        seed=config.same_kinase_seed,
    )
    same_train_loader = DataLoader(
        SampleSubset(same_dataset, same_train),
        batch_size=config.same_kinase_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=translation_collate_fn,
    )
    same_val_loader = DataLoader(
        SampleSubset(same_dataset, same_val),
        batch_size=config.same_kinase_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=translation_collate_fn,
    )
    same_test_loader = DataLoader(
        SampleSubset(same_dataset, same_test),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=translation_collate_fn,
    )

    intra_model = FoldFlowBackbone.from_pretrained(
        checkpoint_path=model_config.official_checkpoint,
        device=device,
        freeze_backbone=True,
    )
    intra_model.condition_encoder.load_state_dict(checkpoint["conditioning_state"])
    intra_optimizer = build_optimizer(
        intra_model,
        Fold1TrainingConfig(
            fold_id=config.fold_id,
            epochs=config.same_kinase_epochs,
            batch_size=config.same_kinase_batch_size,
            learning_rate=config.same_kinase_learning_rate,
            patience=1,
            use_amp=False,
        ),
    )
    intra_history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, config.same_kinase_epochs + 1):
        train_metrics = run_epoch(
            intra_model,
            same_train_loader,
            device,
            cfg_dropout_probability=0.2,
            optimizer=intra_optimizer,
            grad_clip=1.0,
            use_amp=False,
        )
        val_metrics = run_epoch(
            intra_model,
            same_val_loader,
            device,
            cfg_dropout_probability=0.0,
            optimizer=None,
            use_amp=False,
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_rotation_loss": train_metrics["rotation_loss"],
            "train_translation_loss": train_metrics["translation_loss"],
            "validation_total_loss": val_metrics["total_loss"],
            "validation_rotation_loss": val_metrics["rotation_loss"],
            "validation_translation_loss": val_metrics["translation_loss"],
        }
        intra_history.append(row)
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_epoch = epoch

    intra_test_rows = []
    for sample_index, sample in enumerate(same_test_loader.dataset):
        prediction = _sample_with_guidance(
            intra_model,
            sample,
            device,
            guidance_scale=1.0,
            reverse_steps=config.reverse_steps,
            source_min_t=config.source_min_t,
        )
        row = _prediction_metrics(sample, prediction, label="intra_kinase")
        row["sample_index"] = sample_index
        intra_test_rows.append(row)
    intra_test_frame = pd.DataFrame(intra_test_rows)
    intra_metrics = summarize_test_metrics(
        intra_test_frame.rename(
            columns={
                "rmsd_prediction_target": "rmsd_prediction_target",
                "rmsd_source_target": "rmsd_source_target",
                "rmsd_improvement": "rmsd_improvement",
                "successful_translation": "successful_translation",
            }
        )
    )
    intra_history_frame = pd.DataFrame(intra_history)
    intra_history_path = report_root / "same_kinase_training_history.csv"
    intra_history_frame.to_csv(intra_history_path, index=False)
    intra_test_path = report_root / "same_kinase_test_metrics.csv"
    intra_test_frame.to_csv(intra_test_path, index=False)

    identity_frame_for_summary = identity_frame.copy()
    identity_frame_for_summary["rmsd_prediction_target"] = identity_frame_for_summary["rmsd_identity_target"]
    identity_frame_for_summary["rmsd_source_prediction"] = 0.0
    identity_frame_for_summary["mean_displacement_per_residue"] = 0.0
    identity_frame_for_summary["coordinate_mean"] = 0.0
    identity_frame_for_summary["coordinate_std"] = 0.0
    identity_frame_for_summary["coordinate_min"] = 0.0
    identity_frame_for_summary["coordinate_max"] = 0.0
    identity_frame_for_summary["has_nan"] = False
    identity_frame_for_summary["vectorfield_magnitude"] = 0.0
    identity_frame_for_summary["prediction_difference"] = 0.0
    identity_summary = _summarize_sampling_metrics(identity_frame_for_summary)
    conditional_frame = pd.DataFrame(guidance_reports[validation_scale])

    comparison = pd.DataFrame(
        [
            {
                "baseline": "identity",
                "mean_rmsd_source_target": float(identity_frame["rmsd_source_target"].mean()),
                "mean_rmsd_prediction_target": float(identity_frame["rmsd_identity_target"].mean()),
                "mean_rmsd_improvement": 0.0,
                "successful_translation_rate": 0.0,
            },
            {
                "baseline": "foldflow_conditional",
                "mean_rmsd_source_target": float(conditional_frame["rmsd_source_target"].mean()),
                "mean_rmsd_prediction_target": float(conditional_frame["rmsd_prediction_target"].mean()),
                "mean_rmsd_improvement": float(conditional_frame["rmsd_improvement"].mean()),
                "successful_translation_rate": float(conditional_frame["successful_translation"].mean()),
            },
            {
                "baseline": "foldflow_unconditional",
                "mean_rmsd_source_target": float(guidance_summary.loc[guidance_summary["guidance_scale"] == 0.0, "mean_rmsd_source_target"].iloc[0]),
                "mean_rmsd_prediction_target": float(guidance_summary.loc[guidance_summary["guidance_scale"] == 0.0, "mean_rmsd_prediction_target"].iloc[0]),
                "mean_rmsd_improvement": float(guidance_summary.loc[guidance_summary["guidance_scale"] == 0.0, "mean_rmsd_improvement"].iloc[0]),
                "successful_translation_rate": float(guidance_summary.loc[guidance_summary["guidance_scale"] == 0.0, "successful_translation_rate"].iloc[0]),
            },
        ]
    )
    comparison_path = report_root / "baseline_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    report = {
        "environment": {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "foldflow_revision": "9d2c260813da3c9a2bc944973953f63ea6d71203",
        },
        "fold": fold_reports,
        "parameter_report": {
            "official_backbone": model.official_parameter_count,
            "conditioning_adapter": model.conditioning_parameter_count,
            "trainability": parameter_trainability_report(model),
        },
        "test_set_distribution": {
            "identity_baseline": _summarize_sampling_metrics(identity_frame.rename(columns={"rmsd_identity_target": "rmsd_prediction_target"}).assign(
                rmsd_prediction_target=identity_frame["rmsd_identity_target"],
                rmsd_source_prediction=0.0,
                successful_translation=False,
                has_nan=False,
                mean_displacement_per_residue=0.0,
                coordinate_mean=0.0,
                coordinate_std=0.0,
                coordinate_min=0.0,
                coordinate_max=0.0,
                vectorfield_magnitude=0.0,
                prediction_difference=0.0,
                baseline="identity",
            )),
            "source_target_rmsd": {
                "mean": float(source_target_frame["rmsd_source_target"].mean()),
                "median": float(source_target_frame["rmsd_source_target"].median()),
                "below_2A": int((source_target_frame["rmsd_source_target"] < 2.0).sum()),
                "below_3A": int((source_target_frame["rmsd_source_target"] < 3.0).sum()),
                "at_least_5A": int((source_target_frame["rmsd_source_target"] >= 5.0).sum()),
                "min": float(source_target_frame["rmsd_source_target"].min()),
                "max": float(source_target_frame["rmsd_source_target"].max()),
            },
        },
        "guidance_sweep": {
            "guidance_scales": [float(scale) for scale in config.guidance_scales],
            "summary_csv": str(guidance_summary_path),
            "plot": str(guidance_plot_path),
            "per_scale": guidance_summary.to_dict(orient="records"),
        },
        "sampling_validity": {
            "validation_scale": validation_scale,
            "coordinate_distribution": coordinate_summary,
            "sampling_metrics": sampling_summary,
            "test_metrics_csv": str(report_root / f"guidance_{validation_scale}.csv"),
        },
        "same_kinase_experiment": {
            "kinase": config.same_kinase,
            "split_sizes": {
                "train": len(same_train),
                "validation": len(same_val),
                "test": len(same_test),
            },
            "training_history": intra_history,
            "best_validation_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "test_summary": intra_metrics,
            "history_csv": str(intra_history_path),
            "test_metrics_csv": str(intra_test_path),
        },
        "comparisons": {
            "identity_vs_conditional": comparison.to_dict(orient="records"),
            "identity_summary": identity_summary,
            "conditional_summary": summarize_test_metrics(conditional_frame),
            "conditioning_effective": bool(
                guidance_summary.loc[guidance_summary["guidance_scale"] == 1.0, "mean_rmsd_prediction_target"].iloc[0]
                != guidance_summary.loc[guidance_summary["guidance_scale"] == 0.0, "mean_rmsd_prediction_target"].iloc[0]
            ),
        },
        "files": {
            "identity_baseline_csv": str(identity_path),
            "source_target_distribution_csv": str(source_target_path),
            "source_target_histogram": str(hist_path),
            "baseline_comparison_csv": str(comparison_path),
            "report_root": str(report_root),
            "figure_root": str(figure_root),
        },
    }

    conclusion = {
        "evaluation_issue": (
            "likely not primarily evaluation"
            if comparison.loc[comparison["baseline"] == "identity", "mean_rmsd_prediction_target"].iloc[0]
            <= comparison.loc[comparison["baseline"] == "foldflow_conditional", "mean_rmsd_prediction_target"].iloc[0]
            else "possible evaluation bug"
        ),
        "sampling_issue": (
            "strong"
            if guidance_summary["mean_rmsd_source_prediction"].max() > 5.0
            and guidance_summary["mean_rmsd_prediction_target"].min() > sampling_summary["mean_rmsd_source_target"]
            else "moderate"
        ),
        "generalization_issue": (
            "strong"
            if intra_metrics["mean_rmsd_prediction_target"] < comparison.loc[comparison["baseline"] == "foldflow_conditional", "mean_rmsd_prediction_target"].iloc[0]
            else "uncertain"
        ),
        "conditioning_issue": (
            "unlikely primary cause"
            if guidance_summary.loc[guidance_summary["guidance_scale"] == 1.0, "mean_rmsd_prediction_target"].iloc[0]
            != guidance_summary.loc[guidance_summary["guidance_scale"] == 0.0, "mean_rmsd_prediction_target"].iloc[0]
            else "possible"
        ),
    }
    report["conclusion"] = conclusion

    report_path = report_root / "fold1_failure_modes_report.json"
    _safe_json_write(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fold1/best_validation.pt"))
    parser.add_argument("--report-root", type=Path, default=Path("reports/fold1/failure_modes"))
    parser.add_argument("--figure-root", type=Path, default=Path("figures/fold1/failure_modes"))
    parser.add_argument("--same-kinase", type=str, default="EGFR")
    parser.add_argument("--same-kinase-epochs", type=int, default=1)
    parser.add_argument("--same-kinase-batch-size", type=int, default=1)
    parser.add_argument("--same-kinase-learning-rate", type=float, default=1e-4)
    parser.add_argument("--same-kinase-train-fraction", type=float, default=0.7)
    parser.add_argument("--same-kinase-val-fraction", type=float, default=0.15)
    parser.add_argument("--same-kinase-seed", type=int, default=7)
    parser.add_argument("--same-kinase-max-samples", type=int, default=24)
    parser.add_argument("--guidance-scales", type=float, nargs="+", default=[0.0, 1.0, 2.0, 5.0])
    parser.add_argument("--reverse-steps", type=int, default=6)
    parser.add_argument("--source-min-t", type=float, default=0.0)
    parser.add_argument("--max-guidance-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AnalysisConfig(
        fold_id=args.fold_id,
        guidance_scales=tuple(args.guidance_scales),
        reverse_steps=args.reverse_steps,
        source_min_t=args.source_min_t,
        checkpoint_path=args.checkpoint,
        report_root=args.report_root,
        figure_root=args.figure_root,
        same_kinase=args.same_kinase,
        same_kinase_epochs=args.same_kinase_epochs,
        same_kinase_batch_size=args.same_kinase_batch_size,
        same_kinase_learning_rate=args.same_kinase_learning_rate,
        same_kinase_val_fraction=args.same_kinase_val_fraction,
        same_kinase_train_fraction=args.same_kinase_train_fraction,
        same_kinase_seed=args.same_kinase_seed,
        same_kinase_max_samples=args.same_kinase_max_samples,
        max_guidance_samples=args.max_guidance_samples,
    )
    report = run_analysis(config)
    print(json.dumps(report["conclusion"], indent=2))
    print(f"Report written to {config.report_root / 'fold1_failure_modes_report.json'}")


if __name__ == "__main__":
    main()
