"""Single-fold FoldFlow translation experiment helpers."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kinase_data.analysis import _configure_plotting
from kinase_data.translation import (
    InactiveActiveTranslationDataset,
    create_translation_dataloader,
    translation_collate_fn,
)
from models.foldflow_backbone import CFG_VECTORFIELD_KEYS, FoldFlowBackbone
from models.load_foldflow import DEFAULT_CHECKPOINT, configure_foldflow_import
from training.readiness import ConditioningCheckpointManager

configure_foldflow_import()

from openfold.utils import rigid_utils as ru


@dataclass
class Fold1TrainingConfig:
    fold_id: int = 1
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 3
    cfg_dropout_probability: float = 0.2
    guidance_scale: float = 2.0
    unfreeze_backbone: bool = False
    num_workers: int = 0
    use_amp: bool = True
    reverse_steps: int = 6
    reverse_min_t: float = 0.0
    checkpoint_root: Path = Path("checkpoints/fold1")
    report_root: Path = Path("reports/fold1")
    figure_root: Path = Path("figures/fold1")
    source_root: Path | None = None
    official_checkpoint: Path = DEFAULT_CHECKPOINT
    validate_example_index: int = 0


def _mean_masked_squared_error(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    mask = mask.to(prediction.dtype)
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    squared_error = (prediction - target).pow(2) * mask
    denominator = mask.sum().clamp_min(1.0) * prediction.shape[-1]
    return squared_error.sum() / denominator


def build_model_and_loaders(
    config: Fold1TrainingConfig,
    device: torch.device,
) -> tuple[FoldFlowBackbone, DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    pin_memory = device.type == "cuda"
    model = FoldFlowBackbone.from_pretrained(
        checkpoint_path=config.official_checkpoint,
        device=device,
        freeze_backbone=not config.unfreeze_backbone,
        source_root=config.source_root,
    )
    if config.unfreeze_backbone:
        model.unfreeze_official_backbone()
    train_loader = create_translation_dataloader(
        config.fold_id,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        source_root=config.source_root,
    )
    validation_loader = create_translation_dataloader(
        config.fold_id,
        "validation",
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        source_root=config.source_root,
    )
    test_loader = create_translation_dataloader(
        config.fold_id,
        "test",
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        source_root=config.source_root,
    )
    fold_reports = {
        "train": len(train_loader.dataset),
        "validation": len(validation_loader.dataset),
        "test": len(test_loader.dataset),
    }
    return model, train_loader, validation_loader, test_loader, fold_reports


def build_optimizer(
    model: FoldFlowBackbone,
    config: Fold1TrainingConfig,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters were found")
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _prepare_training_batch(
    batch: dict[str, Any],
    model: FoldFlowBackbone,
    device: torch.device,
    cfg_dropout_probability: float,
    training: bool,
    timing: dict[str, float] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if timing is not None:
        timing.setdefault("cpu_tensor_views_seconds", 0.0)
        timing.setdefault("flow_marginal_seconds", 0.0)
        timing.setdefault("device_transfer_seconds", 0.0)
    foldflow_features_cpu = batch["foldflow_features"]
    conditioning_cpu = batch["conditioning"]
    conversion_start = time.perf_counter()
    batch_size, residue_count = batch["foldflow_features"]["rigids_t"].shape[:2]
    source_rigids = foldflow_features_cpu["rigids_t"].detach().reshape(-1, 7)
    target_rigids = foldflow_features_cpu["rigids_0"].detach().reshape(-1, 7)
    res_mask = foldflow_features_cpu["res_mask"].detach().reshape(-1)
    if timing is not None:
        timing["cpu_tensor_views_seconds"] += time.perf_counter() - conversion_start

    t_value = float(torch.rand(()).mul(0.8).add(0.1).item()) if training else 0.5
    flow_start = time.perf_counter()
    flow_match = model.flow_matcher.forward_marginal(
        rigids_0=ru.Rigid.from_tensor_7(target_rigids),
        t=t_value,
        rigids_1=ru.Rigid.from_tensor_7(source_rigids),
    )
    if timing is not None:
        timing["flow_marginal_seconds"] += time.perf_counter() - flow_start
    transfer_start = time.perf_counter()
    foldflow_features = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in foldflow_features_cpu.items()
    }
    conditioning = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in conditioning_cpu.items()
    }
    foldflow_features["rigids_t"] = flow_match["rigids_t"].reshape(batch_size, residue_count, 7).to(device)
    foldflow_features["t"] = torch.full(
        (foldflow_features["rigids_t"].shape[0],), t_value, device=device
    )
    if model.official_config.model.embed.embed_self_conditioning:
        foldflow_features["sc_ca_t"] = foldflow_features["rigids_t"][..., 4:].detach()
    drop_mask = (
        model.condition_encoder.sample_target_dropout(
            foldflow_features["rigids_t"].shape[0],
            cfg_dropout_probability,
            device=device,
        )
        if training and cfg_dropout_probability > 0.0
        else False
    )
    targets = {
        "rot_vectorfield": flow_match["rot_vectorfield"]
        .reshape(batch_size, residue_count, 3, 3)
        .to(device)
        .float(),
        "trans_vectorfield": flow_match["trans_vectorfield"]
        .reshape(batch_size, residue_count, 3)
        .to(device)
        .float(),
        "res_mask": foldflow_features["res_mask"].to(device),
    }
    if isinstance(drop_mask, torch.Tensor):
        conditioning["target_state_drop_mask"] = drop_mask
    if timing is not None:
        timing["device_transfer_seconds"] += time.perf_counter() - transfer_start
    return foldflow_features, conditioning, targets


def compute_batch_losses(
    model: FoldFlowBackbone,
    batch: dict[str, Any],
    device: torch.device,
    cfg_dropout_probability: float,
    training: bool,
    use_amp: bool = False,
    prepare_timing: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    foldflow_features, conditioning, targets = _prepare_training_batch(
        batch, model, device, cfg_dropout_probability, training, timing=prepare_timing
    )
    drop_state = conditioning.pop("target_state_drop_mask", False)
    amp_context = (
        torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda")
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_context:
        outputs = model(
            foldflow_features,
            conditioning,
            drop_target_state=drop_state,
        )
    rot_loss = _mean_masked_squared_error(
        outputs["rot_vectorfield"], targets["rot_vectorfield"], targets["res_mask"]
    )
    trans_loss = _mean_masked_squared_error(
        outputs["trans_vectorfield"], targets["trans_vectorfield"], targets["res_mask"]
    )
    total_loss = rot_loss + trans_loss
    metrics = {
        "total_loss": float(total_loss.detach().cpu()),
        "rotation_loss": float(rot_loss.detach().cpu()),
        "translation_loss": float(trans_loss.detach().cpu()),
    }
    return total_loss, metrics


def run_epoch(
    model: FoldFlowBackbone,
    loader: DataLoader,
    device: torch.device,
    cfg_dropout_probability: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    collect_timing: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, float]]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()
    totals = {"total_loss": 0.0, "rotation_loss": 0.0, "translation_loss": 0.0}
    timing_totals = {
        "dataloader_seconds": 0.0,
        "prepare_batch_seconds": 0.0,
        "cpu_tensor_views_seconds": 0.0,
        "flow_marginal_seconds": 0.0,
        "device_transfer_seconds": 0.0,
        "forward_seconds": 0.0,
        "backward_seconds": 0.0,
        "optimizer_step_seconds": 0.0,
        "batch_seconds": 0.0,
    }
    count = 0
    iterator = iter(loader)
    for _ in range(len(loader)):
        batch_fetch_start = time.perf_counter()
        batch = next(iterator)
        batch_fetch_time = time.perf_counter() - batch_fetch_start
        count += 1
        if training:
            optimizer.zero_grad(set_to_none=True)
            batch_start = time.perf_counter()
            batch_metrics_start = time.perf_counter()
            prepare_timing: dict[str, float] = {}
            foldflow_features, conditioning, targets = _prepare_training_batch(
                batch, model, device, cfg_dropout_probability, training=True, timing=prepare_timing
            )
            timing_totals["prepare_batch_seconds"] += time.perf_counter() - batch_metrics_start
            for key in ("cpu_tensor_views_seconds", "flow_marginal_seconds", "device_transfer_seconds"):
                timing_totals[key] += prepare_timing.get(key, 0.0)
            drop_state = conditioning.pop("target_state_drop_mask", False)
            amp_context = (
                torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda")
                if device.type == "cuda"
                else nullcontext()
            )
            forward_start = time.perf_counter()
            with amp_context:
                outputs = model(
                    foldflow_features,
                    conditioning,
                    drop_target_state=drop_state,
                )
                rot_loss = _mean_masked_squared_error(
                    outputs["rot_vectorfield"], targets["rot_vectorfield"], targets["res_mask"]
                )
                trans_loss = _mean_masked_squared_error(
                    outputs["trans_vectorfield"], targets["trans_vectorfield"], targets["res_mask"]
                )
                loss = rot_loss + trans_loss
            timing_totals["forward_seconds"] += time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        grad_clip,
                    )
                step_start = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                timing_totals["optimizer_step_seconds"] += time.perf_counter() - step_start
            else:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        grad_clip,
                    )
                step_start = time.perf_counter()
                optimizer.step()
                timing_totals["optimizer_step_seconds"] += time.perf_counter() - step_start
            timing_totals["backward_seconds"] += time.perf_counter() - backward_start
            metrics = {
                "total_loss": float(loss.detach().float().cpu()),
                "rotation_loss": float(rot_loss.detach().float().cpu()),
                "translation_loss": float(trans_loss.detach().float().cpu()),
            }
            timing_totals["batch_seconds"] += time.perf_counter() - batch_start
        else:
            batch_start = time.perf_counter()
            batch_metrics_start = time.perf_counter()
            prepare_timing = {}
            foldflow_features, conditioning, targets = _prepare_training_batch(
                batch, model, device, cfg_dropout_probability, training=False, timing=prepare_timing
            )
            timing_totals["prepare_batch_seconds"] += time.perf_counter() - batch_metrics_start
            for key in ("cpu_tensor_views_seconds", "flow_marginal_seconds", "device_transfer_seconds"):
                timing_totals[key] += prepare_timing.get(key, 0.0)
            drop_state = conditioning.pop("target_state_drop_mask", False)
            forward_start = time.perf_counter()
            with torch.no_grad():
                outputs = model(
                    foldflow_features,
                    conditioning,
                    drop_target_state=drop_state,
                )
                rot_loss = _mean_masked_squared_error(
                    outputs["rot_vectorfield"], targets["rot_vectorfield"], targets["res_mask"]
                )
                trans_loss = _mean_masked_squared_error(
                    outputs["trans_vectorfield"], targets["trans_vectorfield"], targets["res_mask"]
                )
                loss = rot_loss + trans_loss
            timing_totals["forward_seconds"] += time.perf_counter() - forward_start
            metrics = {
                "total_loss": float(loss.detach().float().cpu()),
                "rotation_loss": float(rot_loss.detach().float().cpu()),
                "translation_loss": float(trans_loss.detach().float().cpu()),
            }
            timing_totals["batch_seconds"] += time.perf_counter() - batch_start
        for key in totals:
            totals[key] += metrics[key]
        timing_totals["dataloader_seconds"] += batch_fetch_time
    if count == 0:
        raise ValueError("Data loader produced no batches")
    summary = {key: value / count for key, value in totals.items()}
    timing_summary = {key: value / count for key, value in timing_totals.items()}
    epoch_total_seconds = timing_totals["dataloader_seconds"] + timing_totals["batch_seconds"]
    timing_summary["epoch_total_seconds"] = epoch_total_seconds
    timing_summary["average_batch_total_seconds"] = epoch_total_seconds / count
    timing_summary["batch_count"] = count
    timing_summary["dataloader_fraction"] = timing_totals["dataloader_seconds"] / max(
        epoch_total_seconds, 1e-12
    )
    timing_summary["prepare_fraction"] = timing_totals["prepare_batch_seconds"] / max(
        epoch_total_seconds, 1e-12
    )
    timing_summary["cpu_tensor_views_fraction"] = timing_totals[
        "cpu_tensor_views_seconds"
    ] / max(epoch_total_seconds, 1e-12)
    timing_summary["flow_marginal_fraction"] = timing_totals[
        "flow_marginal_seconds"
    ] / max(epoch_total_seconds, 1e-12)
    timing_summary["device_transfer_fraction"] = timing_totals[
        "device_transfer_seconds"
    ] / max(epoch_total_seconds, 1e-12)
    timing_summary["forward_fraction"] = timing_totals["forward_seconds"] / max(
        epoch_total_seconds, 1e-12
    )
    timing_summary["backward_fraction"] = timing_totals["backward_seconds"] / max(
        epoch_total_seconds, 1e-12
    )
    timing_summary["optimizer_step_fraction"] = timing_totals[
        "optimizer_step_seconds"
    ] / max(epoch_total_seconds, 1e-12)
    if collect_timing:
        return summary, timing_summary
    return summary


def parameter_trainability_report(model: FoldFlowBackbone) -> dict[str, Any]:
    backbone_trainable = [
        name
        for name, parameter in model.official_model.named_parameters()
        if parameter.requires_grad
    ]
    conditioning_trainable = [
        name
        for name, parameter in model.condition_encoder.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "backbone_requires_grad": backbone_trainable,
        "conditioning_requires_grad": conditioning_trainable,
        "backbone_trainable_count": len(backbone_trainable),
        "conditioning_trainable_count": len(conditioning_trainable),
        "official_backbone_frozen": model.forward_is_frozen(),
    }


def recommended_num_workers(device: torch.device) -> int:
    cpu_count = os.cpu_count() or 1
    if device.type == "cuda":
        return max(2, min(8, cpu_count // 2))
    return max(0, min(4, cpu_count // 4))


def validation_cfg_diagnostics(
    model: FoldFlowBackbone,
    loader: DataLoader,
    device: torch.device,
    guidance_scales: Iterable[float] = (0.0, 1.0, 3.0, 5.0),
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    results = {str(scale): {"vectorfield_magnitude": [], "prediction_difference": []} for scale in guidance_scales}
    batch_count = 0
    for batch in loader:
        batch_count += 1
        foldflow_features, conditioning, _ = _prepare_training_batch(
            batch, model, device, cfg_dropout_probability=0.0, training=False
        )
        with torch.no_grad():
            conditional = model.forward(
                foldflow_features, conditioning, drop_target_state=False
            )
            unconditional = model.forward(
                foldflow_features, conditioning, drop_target_state=True
            )
            for scale in guidance_scales:
                guided = model.guided_forward(
                    foldflow_features, conditioning, guidance_scale=scale
                )
                magnitude = 0.5 * (
                    guided["rot_vectorfield"].norm(dim=-1).mean().item()
                    + guided["trans_vectorfield"].norm(dim=-1).mean().item()
                )
                diff = 0.5 * (
                    (conditional["rot_vectorfield"] - unconditional["rot_vectorfield"])
                    .abs()
                    .mean()
                    .item()
                    + (
                        conditional["trans_vectorfield"]
                        - unconditional["trans_vectorfield"]
                    )
                    .abs()
                    .mean()
                    .item()
                )
                results[str(scale)]["vectorfield_magnitude"].append(magnitude)
                results[str(scale)]["prediction_difference"].append(diff)
        if max_batches is not None and batch_count >= max_batches:
            break
    summary = {}
    for scale, payload in results.items():
        summary[scale] = {
            "average_vectorfield_magnitude": float(np.mean(payload["vectorfield_magnitude"])),
            "average_prediction_difference": float(np.mean(payload["prediction_difference"])),
            "batches": len(payload["vectorfield_magnitude"]),
        }
    return summary


def select_example_sample(
    loader: DataLoader, index: int = 0
) -> dict[str, Any]:
    dataset = loader.dataset
    sample = dataset[index]
    batch = translation_collate_fn([sample])
    return {
        "sample": sample,
        "batch": batch,
    }


def run_example_prediction(
    model: FoldFlowBackbone,
    sample: dict[str, Any],
    device: torch.device,
    guidance_scale: float,
    num_steps: int,
    min_t: float,
) -> dict[str, Any]:
    batch = translation_collate_fn([sample])
    foldflow_features = {
        key: value.to(device) for key, value in batch["foldflow_features"].items()
    }
    conditioning = {
        key: value.to(device) for key, value in batch["conditioning"].items()
    }
    with torch.no_grad():
        prediction = model.sample_translation(
            foldflow_features,
            conditioning,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            min_t=min_t,
            noise_scale=0.0,
            center=True,
        )
    payload = {
        "sample": sample,
        "source_structure": sample["source_structure"],
        "target_structure": sample["target_structure"],
        "predicted_structure": {
            "rigids": prediction["rigids"].detach().cpu(),
            "trajectory": prediction["trajectory"].detach().cpu(),
            "atom37": prediction["model_output"]["atom37"].detach().cpu(),
            "atom14": prediction["model_output"]["atom14"].detach().cpu(),
            "psi": prediction["model_output"]["psi"].detach().cpu(),
        },
        "conditioning": {
            key: value.detach().cpu() for key, value in batch["conditioning"].items()
        },
        "foldflow_features": {
            key: value.detach().cpu() for key, value in batch["foldflow_features"].items()
        },
    }
    return payload


def _structure_ca(structure: dict[str, torch.Tensor]) -> np.ndarray:
    if "rigids" in structure:
        rigids = structure["rigids"]
        if rigids.ndim == 3:
            rigids = rigids[0]
        return rigids[:, 4:7].detach().cpu().numpy()
    if "atom37" in structure:
        atom37 = structure["atom37"]
        if atom37.ndim == 4:
            atom37 = atom37[0]
        return atom37[:, 1, :].detach().cpu().numpy()
    raise ValueError("Structure payload does not contain CA coordinates")


def structure_ca_coordinates(structure: dict[str, torch.Tensor]) -> np.ndarray:
    return _structure_ca(structure)


def rmsd_from_coordinates(
    reference: np.ndarray, moving: np.ndarray, align: bool = True
) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape:
        raise ValueError(
            f"Coordinate shapes must match, received {reference.shape} and {moving.shape}"
        )
    if align:
        reference_centered = reference - reference.mean(axis=0, keepdims=True)
        moving_centered = moving - moving.mean(axis=0, keepdims=True)
        covariance = moving_centered.T @ reference_centered
        u, _, vt = np.linalg.svd(covariance)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            vt[-1, :] *= -1
            rotation = u @ vt
        moving = moving_centered @ rotation
        reference = reference_centered
    diff = reference - moving
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


def evaluate_test_loader(
    model: FoldFlowBackbone,
    loader: DataLoader,
    device: torch.device,
    guidance_scale: float,
    num_steps: int,
    min_t: float,
) -> pd.DataFrame:
    rows = []
    model.eval()
    for sample_index, sample in enumerate(loader.dataset):
        batch = translation_collate_fn([sample])
        foldflow_features = {
            key: value.to(device) for key, value in batch["foldflow_features"].items()
        }
        conditioning = {
            key: value.to(device) for key, value in batch["conditioning"].items()
        }
        with torch.no_grad():
            prediction = model.sample_translation(
                foldflow_features,
                conditioning,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                min_t=min_t,
                noise_scale=0.0,
                center=True,
            )
        source_ca = structure_ca_coordinates(sample["source_structure"])
        target_ca = structure_ca_coordinates(sample["target_structure"])
        predicted_ca = prediction["rigids"][0, :, 4:7].detach().cpu().numpy()
        source_target_rmsd = rmsd_from_coordinates(target_ca, source_ca)
        pred_target_rmsd = rmsd_from_coordinates(target_ca, predicted_ca)
        improvement = source_target_rmsd - pred_target_rmsd
        rows.append(
            {
                "sample_index": sample_index,
                "kinase": sample["kinase"],
                "source_pdb_id": sample["source_pdb_id"],
                "target_pdb_id": sample["target_pdb_id"],
                "sequence_identity": sample["sequence_identity"],
                "residue_count": sample["residue_count"],
                "rmsd_source_target": source_target_rmsd,
                "rmsd_prediction_target": pred_target_rmsd,
                "rmsd_improvement": improvement,
                "successful_translation": improvement > 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_test_metrics(test_metrics: pd.DataFrame) -> dict[str, Any]:
    if test_metrics.empty:
        raise ValueError("Test metrics dataframe is empty")
    improvement = test_metrics["rmsd_improvement"]
    success_rate = float(test_metrics["successful_translation"].mean())
    return {
        "number_of_samples": int(len(test_metrics)),
        "mean_rmsd_source_target": float(test_metrics["rmsd_source_target"].mean()),
        "mean_rmsd_prediction_target": float(
            test_metrics["rmsd_prediction_target"].mean()
        ),
        "mean_rmsd_improvement": float(improvement.mean()),
        "median_rmsd_improvement": float(improvement.median()),
        "successful_translation_rate": success_rate,
        "successful_translations": int(test_metrics["successful_translation"].sum()),
        "median_sample_index": int(test_metrics.iloc[len(test_metrics) // 2].sample_index),
    }


def select_test_examples(
    test_metrics: pd.DataFrame,
) -> dict[str, pd.Series]:
    if test_metrics.empty:
        raise ValueError("Test metrics dataframe is empty")
    sorted_metrics = test_metrics.sort_values("rmsd_improvement", ascending=False).reset_index(drop=True)
    successful = sorted_metrics.iloc[0]
    median = sorted_metrics.iloc[len(sorted_metrics) // 2]
    difficult = sorted_metrics.iloc[-1]
    return {"successful": successful, "median": median, "difficult": difficult}


def _plot_structure_traces(
    axes, structures: list[tuple[str, np.ndarray, str]]
) -> None:
    colors = {"inactive": "#1f77b4", "predicted": "#ff7f0e", "active": "#2ca02c"}
    for axis, (title, coords, kind) in zip(axes, structures):
        axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=colors[kind], lw=1.5)
        axis.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="black", s=10)
        axis.set_title(title)
        axis.set_axis_off()
    return None


def plot_prediction_comparison(
    prediction: dict[str, Any],
    output_path: Path | str = Path("figures/fold1/prediction_comparison.png"),
) -> Path:
    _configure_plotting()
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = _structure_ca(prediction["source_structure"])
    predicted = prediction["predicted_structure"]["rigids"][0, :, 4:7].numpy()
    target = _structure_ca(prediction["target_structure"])
    figure = plt.figure(figsize=(15, 5))
    axes = [
        figure.add_subplot(1, 3, index + 1, projection="3d")
        for index in range(3)
    ]
    _plot_structure_traces(
        axes,
        [
            ("Inactive source", source, "inactive"),
            ("Predicted active", predicted, "predicted"),
            ("Real active", target, "active"),
        ],
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_trajectory_snapshots(
    prediction: dict[str, Any],
    output_dir: Path | str = Path("figures/fold1/trajectory"),
) -> list[Path]:
    _configure_plotting()
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = prediction["predicted_structure"]["trajectory"]
    if trajectory.ndim != 4:
        raise ValueError("Expected trajectory tensor with shape [T, B, N, 7]")
    trajectories = trajectory[:, 0, :, 4:7].numpy()
    total = trajectories.shape[0]
    labels = np.linspace(1.0, 0.0, total)
    paths = []
    for index, (coords, t_value) in enumerate(zip(trajectories, labels)):
        label = "final" if index == total - 1 else f"t={t_value:.1f}"
        figure = plt.figure(figsize=(5, 5))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color="#d62728", lw=1.5)
        axis.scatter(coords[0, 0], coords[0, 1], coords[0, 2], color="black", s=10)
        axis.set_title(label)
        axis.set_axis_off()
        figure.tight_layout()
        path = output_dir / f"{index:02d}_{label.replace('=', '').replace('.', '_')}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def save_training_curves(
    metrics_csv: Path | str,
    figure_root: Path | str = Path("figures/fold1"),
) -> None:
    _configure_plotting()
    import matplotlib.pyplot as plt

    metrics_csv = Path(metrics_csv)
    figure_root = Path(figure_root)
    figure_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(metrics_csv)
    mapping = {
        "train_loss.png": ("train_total_loss",),
        "rotation_loss.png": ("train_rotation_loss", "validation_rotation_loss"),
        "translation_loss.png": ("train_translation_loss", "validation_translation_loss"),
    }
    for filename, series in mapping.items():
        figure, axis = plt.subplots(figsize=(7, 4))
        for column in series:
            axis.plot(frame["epoch"], frame[column], label=column)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
        axis.set_title(filename.replace(".png", "").replace("_", " ").title())
        figure.tight_layout()
        figure.savefig(figure_root / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(frame["epoch"], frame["validation_total_loss"], label="validation_total_loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.set_title("Validation Loss")
    figure.tight_layout()
    figure.savefig(figure_root / "validation_loss.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_metrics_csv(rows: list[dict[str, Any]], metrics_path: Path | str) -> Path:
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    return metrics_path


def save_checkpoint(
    path: Path | str,
    model: FoldFlowBackbone,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_validation_loss: float,
    config: Fold1TrainingConfig,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "conditioning_state": model.condition_encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.__dict__.copy(),
        "metrics": metrics,
        "model_metadata": {
            "official_parameters": model.official_parameter_count,
            "conditioning_parameters": model.conditioning_parameter_count,
            "official_checkpoint": str(config.official_checkpoint),
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    model: FoldFlowBackbone,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.condition_encoder.load_state_dict(payload["conditioning_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
