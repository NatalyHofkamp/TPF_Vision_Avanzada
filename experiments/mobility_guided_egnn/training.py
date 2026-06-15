"""Training utilities for the mobility-guided EGNN experiments."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from kinase_data.esm import EmbeddingCache
from kinase_data.translation import InactiveActiveTranslationDataset

from models.mobility_guided_egnn import (
    MobilityEGNNConfig,
    MobilityGuidedEGNN,
    local_pairwise_distance_loss,
    weighted_delta_loss,
)
from .data import load_pdb_chain_residues

LOGGER = logging.getLogger(__name__)


def rmsd_from_coordinates(reference: np.ndarray, moving: np.ndarray, align: bool = True) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if reference.shape != moving.shape or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError("Coordinates must have shape [N, 3] and match in size")
    if reference.size == 0:
        return float("nan")
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        return float("inf")
    if not align:
        diff = reference - moving
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    moving_centered = moving - moving.mean(axis=0, keepdims=True)
    covariance = moving_centered.T @ reference_centered
    try:
        u, _, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        diff = reference_centered - moving_centered
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt
    aligned = moving_centered @ rotation
    diff = reference_centered - aligned
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1))))


@dataclass(frozen=True)
class MobilityPairSample:
    coords_source: torch.Tensor
    coords_target: torch.Tensor
    true_delta: torch.Tensor
    esm: torch.Tensor
    mobility_prior: torch.Tensor
    mobility_binary: torch.Tensor
    mask: torch.Tensor
    metadata: dict[str, Any]


class MobilityConditionedPairDataset(Dataset):
    """Pair dataset augmented with per-residue mobility priors."""

    def __init__(
        self,
        fold_id: int,
        split: str,
        mobility_predictions: pd.DataFrame,
        project_root: Path | str = Path("."),
        folds_dir: Path | str | None = None,
        manifest_csv: Path | str | None = None,
        metadata_csv: Path | str | None = None,
        use_classifier_prior: bool = True,
    ):
        self.project_root = Path(project_root)
        self.base = InactiveActiveTranslationDataset(
            fold_id=fold_id,
            split=split,
            project_root=self.project_root,
            folds_dir=Path(folds_dir or (self.project_root / "data" / "folds")),
            manifest_csv=Path(manifest_csv or (self.project_root / "data" / "esm_manifest.csv")),
            metadata_csv=Path(metadata_csv or (self.project_root / "data" / "metadata" / "kinase_labels.csv")),
            load_structures=False,
        )
        self.cache = EmbeddingCache(self.project_root / "cache" / "esm_embeddings")
        self.use_classifier_prior = use_classifier_prior
        self._logged_hashes: set[str] = set()
        self.prediction_map: dict[int, np.ndarray] = {}
        self.binary_map: dict[int, np.ndarray] = {}
        self._esm_cache: dict[str, torch.Tensor] = {}
        self._residue_cache: dict[tuple[Path, str], list[Any]] = {}
        self._sample_cache: dict[int, dict[str, Any]] = {}
        for sample_index, group in mobility_predictions.groupby("sample_index", sort=True):
            ordered = group.sort_values("residue_index")
            self.prediction_map[int(sample_index)] = ordered["p_change"].to_numpy(dtype=np.float32)
            self.binary_map[int(sample_index)] = ordered["mobility_binary"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> MobilityPairSample:
        cached_sample = self._sample_cache.get(index)
        if cached_sample is not None:
            return MobilityPairSample(
                coords_source=cached_sample["coords_source"].clone(),
                coords_target=cached_sample["coords_target"].clone(),
                true_delta=cached_sample["true_delta"].clone(),
                esm=cached_sample["esm"].clone(),
                mobility_prior=cached_sample["mobility_prior"].clone(),
                mobility_binary=cached_sample["mobility_binary"].clone(),
                mask=cached_sample["mask"].clone(),
                metadata=dict(cached_sample["metadata"]),
            )
        sample = dict(self.base[index])
        source_info = sample["source_structure"]
        target_info = sample["target_structure"]
        source_key = (Path(source_info["pdb_path"]), str(source_info["chain"]))
        target_key = (Path(target_info["pdb_path"]), str(target_info["chain"]))
        if source_key not in self._residue_cache:
            self._residue_cache[source_key] = load_pdb_chain_residues(source_info["pdb_path"], source_info["chain"])
        if target_key not in self._residue_cache:
            self._residue_cache[target_key] = load_pdb_chain_residues(target_info["pdb_path"], target_info["chain"])
        source_residues = self._residue_cache[source_key]
        target_residues = self._residue_cache[target_key]
        source_selected = [source_residues[i] for i in source_info["residue_indices"]]
        target_selected = [target_residues[i] for i in target_info["residue_indices"]]
        if len(source_selected) != len(target_selected):
            raise ValueError(f"Residue count mismatch for sample {index}")

        if sample["source_sequence_hash"] not in self._logged_hashes:
            LOGGER.info("Loading ESM embeddings from cache: %s", sample["source_sequence_hash"])
            print(f"Loading ESM embeddings from cache: {sample['source_sequence_hash']}")
            self._logged_hashes.add(sample["source_sequence_hash"])
        seq_hash = sample["source_sequence_hash"]
        if seq_hash not in self._esm_cache:
            payload = self.cache.load(seq_hash)
            self._esm_cache[seq_hash] = payload["residue_embeddings"].float().detach().clone()
        esm = self._esm_cache[seq_hash][list(source_info["residue_indices"])].float()
        coords_source = torch.from_numpy(np.stack([res.ca for res in source_selected]).astype(np.float32))
        coords_target = torch.from_numpy(np.stack([res.ca for res in target_selected]).astype(np.float32))
        mobility_prior = torch.from_numpy(self.prediction_map[index]).float()
        binary_mobile = torch.from_numpy(self.binary_map[index]).float()
        mask = torch.ones(len(source_selected), dtype=torch.float32)
        metadata = {
            "kinase": sample["kinase"],
            "sample_index": index,
            "source_pdb_id": sample["source_pdb_id"],
            "target_pdb_id": sample["target_pdb_id"],
            "sequence_identity": float(sample["sequence_identity"]),
            "residue_count": len(source_selected),
        }
        result = MobilityPairSample(
            coords_source=coords_source,
            coords_target=coords_target,
            true_delta=coords_target - coords_source,
            esm=esm,
            mobility_prior=mobility_prior,
            mobility_binary=binary_mobile,
            mask=mask,
            metadata=metadata,
        )
        self._sample_cache[index] = {
            "coords_source": coords_source,
            "coords_target": coords_target,
            "true_delta": coords_target - coords_source,
            "esm": esm,
            "mobility_prior": mobility_prior,
            "mobility_binary": binary_mobile,
            "mask": mask,
            "metadata": metadata,
        }
        return result


class MobilityPairCollator:
    def __call__(self, samples: list[MobilityPairSample]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty sample list")
        max_len = max(int(sample.mask.shape[0]) for sample in samples)
        batch_size = len(samples)
        esm_dim = int(samples[0].esm.shape[-1])
        coords_source = torch.zeros(batch_size, max_len, 3)
        coords_target = torch.zeros(batch_size, max_len, 3)
        true_delta = torch.zeros(batch_size, max_len, 3)
        esm = torch.zeros(batch_size, max_len, esm_dim)
        mobility_prior = torch.zeros(batch_size, max_len)
        mobility_binary = torch.zeros(batch_size, max_len)
        mask = torch.zeros(batch_size, max_len)
        metadata: list[dict[str, Any]] = []
        for batch_index, sample in enumerate(samples):
            length = int(sample.mask.shape[0])
            coords_source[batch_index, :length] = sample.coords_source
            coords_target[batch_index, :length] = sample.coords_target
            true_delta[batch_index, :length] = sample.true_delta
            esm[batch_index, :length] = sample.esm
            mobility_prior[batch_index, :length] = sample.mobility_prior
            mobility_binary[batch_index, :length] = sample.mobility_binary
            mask[batch_index, :length] = sample.mask
            metadata.append(sample.metadata)
        return {
            "coords_source": coords_source,
            "coords_target": coords_target,
            "true_delta": true_delta,
            "esm": esm,
            "mobility_prior": mobility_prior,
            "mobility_binary": mobility_binary,
            "mask": mask,
            "metadata": metadata,
        }


def build_conditioned_loaders(
    fold_id: int,
    mobility_predictions: dict[str, pd.DataFrame],
    project_root: Path | str = Path("."),
    batch_size: int = 1,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders = {}
    collator = MobilityPairCollator()
    pin_memory = torch.cuda.is_available()
    for split in ("train", "validation", "test"):
        dataset = MobilityConditionedPairDataset(
            fold_id=fold_id,
            split=split,
            mobility_predictions=mobility_predictions[split],
            project_root=project_root,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=collator,
        )
    return loaders


def _mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def _sample_metrics(
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    pred_coords: np.ndarray,
) -> dict[str, float]:
    source_target = rmsd_from_coordinates(target_coords, source_coords)
    prediction_target = rmsd_from_coordinates(target_coords, pred_coords)
    source_prediction = rmsd_from_coordinates(source_coords, pred_coords)
    return {
        "rmsd_source_target": source_target,
        "rmsd_prediction_target": prediction_target,
        "rmsd_source_prediction": source_prediction,
        "rmsd_improvement": source_target - prediction_target,
        "success": float(prediction_target < source_target),
    }


def predict_egnn_split(
    model: MobilityGuidedEGNN,
    loader: DataLoader,
    device: torch.device,
    *,
    use_binary_mobile_mask: bool = True,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            coords_source = batch["coords_source"].to(device)
            coords_target = batch["coords_target"].to(device)
            true_delta = batch["true_delta"].to(device)
            esm = batch["esm"].to(device)
            mobility_prior = batch["mobility_prior"].to(device)
            mobility_binary = batch["mobility_binary"].to(device)
            mask = batch["mask"].to(device)
            binary_mask = mobility_binary if use_binary_mobile_mask else None
            pred_delta, pred_coords = model(
                coords_source,
                esm,
                mobility_prior,
                mask,
                binary_mobile_mask=binary_mask,
            )
            for sample_offset in range(coords_source.shape[0]):
                n = int(mask[sample_offset].sum().item())
                source_np = coords_source[sample_offset, :n].cpu().numpy()
                target_np = coords_target[sample_offset, :n].cpu().numpy()
                pred_np = pred_coords[sample_offset, :n].cpu().numpy()
                true_delta_np = true_delta[sample_offset, :n].cpu().numpy()
                pred_motion = pred_delta[sample_offset, :n].norm(dim=-1).cpu().numpy()
                true_motion = np.linalg.norm(true_delta_np, axis=-1)
                row = {
                    **batch["metadata"][sample_offset],
                    **_sample_metrics(source_np, target_np, pred_np),
                    "mean_true_motion": float(true_motion.mean()),
                    "mean_pred_motion": float(pred_motion.mean()),
                    "predicted_delta_norm_mean": float(pred_motion.mean()),
                    "true_delta_norm_mean": float(true_motion.mean()),
                    "source_coords": source_np.tolist(),
                    "target_coords": target_np.tolist(),
                    "prediction_coords": pred_np.tolist(),
                    "true_motion": true_motion.tolist(),
                    "pred_motion": pred_motion.tolist(),
                    "residue_error": np.linalg.norm(pred_np - target_np, axis=-1).tolist(),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def train_egnn_experiment(
    name: str,
    model: MobilityGuidedEGNN,
    loaders: dict[str, DataLoader],
    device: torch.device,
    output_dir: Path | str,
    *,
    lambda_distance: float = 0.0,
    use_weighted_loss: bool = False,
    alpha: float = 1.0,
    max_epochs: int = 20,
    patience: int = 5,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    distance_window_radius: int = 4,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda") if use_amp and device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        LOGGER.info(
            "[%s] epoch %d/%d start | best_val=%.6f | patience=%d/%d",
            name,
            epoch,
            max_epochs,
            best_val,
            patience_counter,
            patience,
        )
        model.train()
        train_totals = {"loss": 0.0, "delta": 0.0, "distance": 0.0, "samples": 0}
        for batch in loaders["train"]:
            coords_source = batch["coords_source"].to(device)
            coords_target = batch["coords_target"].to(device)
            true_delta = batch["true_delta"].to(device)
            esm = batch["esm"].to(device)
            mobility_prior = batch["mobility_prior"].to(device)
            mobility_binary = batch["mobility_binary"].to(device)
            mask = batch["mask"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pred_delta, pred_coords = model(coords_source, esm, mobility_prior, mask, binary_mobile_mask=mobility_binary)
                if use_weighted_loss:
                    delta_loss = weighted_delta_loss(pred_delta, true_delta, mobility_prior, mask, alpha=alpha)
                else:
                    delta_loss = ((pred_delta - true_delta).pow(2) * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0) / 3.0
                distance_loss = (
                    local_pairwise_distance_loss(pred_coords, coords_target, mask, window_radius=distance_window_radius)
                    if lambda_distance > 0
                    else torch.zeros((), device=device)
                )
                loss = delta_loss + lambda_distance * distance_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in epoch {epoch} of {name}")
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError(f"Non-finite gradient norm in {name}")
            train_totals["loss"] += float(loss.detach().cpu()) * coords_source.shape[0]
            train_totals["delta"] += float(delta_loss.detach().cpu()) * coords_source.shape[0]
            train_totals["distance"] += float(distance_loss.detach().cpu()) * coords_source.shape[0]
            train_totals["samples"] += int(coords_source.shape[0])

        model.eval()
        val_totals = {"loss": 0.0, "delta": 0.0, "distance": 0.0, "samples": 0}
        with torch.no_grad():
            for batch in loaders["validation"]:
                coords_source = batch["coords_source"].to(device)
                coords_target = batch["coords_target"].to(device)
                true_delta = batch["true_delta"].to(device)
                esm = batch["esm"].to(device)
                mobility_prior = batch["mobility_prior"].to(device)
                mobility_binary = batch["mobility_binary"].to(device)
                mask = batch["mask"].to(device)
                pred_delta, pred_coords = model(coords_source, esm, mobility_prior, mask, binary_mobile_mask=mobility_binary)
                if use_weighted_loss:
                    delta_loss = weighted_delta_loss(pred_delta, true_delta, mobility_prior, mask, alpha=alpha)
                else:
                    delta_loss = ((pred_delta - true_delta).pow(2) * mask.unsqueeze(-1)).sum() / mask.sum().clamp_min(1.0) / 3.0
                distance_loss = (
                    local_pairwise_distance_loss(pred_coords, coords_target, mask, window_radius=distance_window_radius)
                    if lambda_distance > 0
                    else torch.zeros((), device=device)
                )
                loss = delta_loss + lambda_distance * distance_loss
                val_totals["loss"] += float(loss.detach().cpu()) * coords_source.shape[0]
                val_totals["delta"] += float(delta_loss.detach().cpu()) * coords_source.shape[0]
                val_totals["distance"] += float(distance_loss.detach().cpu()) * coords_source.shape[0]
                val_totals["samples"] += int(coords_source.shape[0])

        train_count = max(1, train_totals["samples"])
        val_count = max(1, val_totals["samples"])
        train_metrics = {
            "total_loss": train_totals["loss"] / train_count,
            "delta_loss": train_totals["delta"] / train_count,
            "distance_loss": train_totals["distance"] / train_count,
        }
        val_metrics = {
            "total_loss": val_totals["loss"] / val_count,
            "delta_loss": val_totals["delta"] / val_count,
            "distance_loss": val_totals["distance"] / val_count,
        }
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "train_delta_loss": train_metrics["delta_loss"],
                "train_distance_loss": train_metrics["distance_loss"],
                "val_total_loss": val_metrics["total_loss"],
                "val_delta_loss": val_metrics["delta_loss"],
                "val_distance_loss": val_metrics["distance_loss"],
            }
        )
        torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict()}, last_path)
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict()}, best_path)
            LOGGER.info("[%s] epoch %d improved validation loss to %.6f", name, epoch, best_val)
        else:
            patience_counter += 1
            LOGGER.info(
                "[%s] epoch %d did not improve validation loss (current=%.6f, best=%.6f, patience=%d/%d)",
                name,
                epoch,
                val_metrics["total_loss"],
                best_val,
                patience_counter,
                patience,
            )
        if patience_counter >= patience:
            LOGGER.info("[%s] early stopping triggered at epoch %d", name, epoch)
            break

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    return {
        "name": name,
        "history": history_frame,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "training_seconds": time.perf_counter() - started,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
    }


def summarize_prediction_frame(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "rmsd_source_target": float(frame["rmsd_source_target"].mean()),
        "rmsd_prediction_target": float(frame["rmsd_prediction_target"].mean()),
        "rmsd_source_prediction": float(frame["rmsd_source_prediction"].mean()),
        "rmsd_improvement": float(frame["rmsd_improvement"].mean()),
        "success_rate": float(frame["success"].mean()),
        "mean_true_motion": float(frame["mean_true_motion"].mean()),
        "mean_pred_motion": float(frame["mean_pred_motion"].mean()),
    }
