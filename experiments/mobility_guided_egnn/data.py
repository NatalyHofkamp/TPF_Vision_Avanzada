"""Residue-level data extraction for mobility-guided kinase prediction."""

from __future__ import annotations

import logging
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from kinase_data.esm import EmbeddingCache
from kinase_data.translation import InactiveActiveTranslationDataset

LOGGER = logging.getLogger(__name__)

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ORDER)}
RESNAME_TO_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


@dataclass(frozen=True)
class ChainResidue:
    """Backbone residue extracted from a PDB chain."""

    residue_index: tuple[str, str]
    resname: str
    n: np.ndarray | None
    ca: np.ndarray
    c: np.ndarray | None


@dataclass(frozen=True)
class MobilityResidueSplit:
    """Residue-level tensors and metadata for one Fold split."""

    split: str
    metadata: pd.DataFrame
    feature_matrix: np.ndarray
    mobility_regression: np.ndarray
    mobility_binary: np.ndarray
    sample_slices: dict[int, slice]
    embeddings_loaded_from_cache: bool = True

    def to_frame(self) -> pd.DataFrame:
        return self.metadata.copy()


def _split_cache_dir(project_root: Path) -> Path:
    return project_root / "reports" / "fold1" / "mobility_guided_egnn" / "cache"


def _split_cache_paths(project_root: Path, split_name: str) -> dict[str, Path]:
    cache_dir = _split_cache_dir(project_root)
    return {
        "metadata": cache_dir / f"{split_name}_metadata.csv",
        "arrays": cache_dir / f"{split_name}_arrays.npz",
        "manifest": cache_dir / f"{split_name}_manifest.json",
    }


def _load_split_cache(project_root: Path, split_name: str) -> MobilityResidueSplit | None:
    paths = _split_cache_paths(project_root, split_name)
    if not paths["metadata"].is_file() or not paths["arrays"].is_file() or not paths["manifest"].is_file():
        return None
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    arrays = np.load(paths["arrays"], allow_pickle=False)
    metadata = pd.read_csv(paths["metadata"])
    sample_slices = {
        int(sample_index): slice(int(start), int(end))
        for sample_index, start, end in zip(
            manifest["sample_indices"],
            manifest["sample_slice_starts"],
            manifest["sample_slice_ends"],
        )
    }
    return MobilityResidueSplit(
        split=split_name,
        metadata=metadata,
        feature_matrix=arrays["feature_matrix"],
        mobility_regression=arrays["mobility_regression"],
        mobility_binary=arrays["mobility_binary"],
        sample_slices=sample_slices,
        embeddings_loaded_from_cache=bool(manifest.get("embeddings_loaded_from_cache", True)),
    )


def _save_split_cache(project_root: Path, split: MobilityResidueSplit) -> None:
    paths = _split_cache_paths(project_root, split.split)
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    split.metadata.to_csv(paths["metadata"], index=False)
    sample_indices = sorted(split.sample_slices)
    manifest = {
        "split": split.split,
        "embeddings_loaded_from_cache": bool(split.embeddings_loaded_from_cache),
        "sample_indices": sample_indices,
        "sample_slice_starts": [int(split.sample_slices[idx].start) for idx in sample_indices],
        "sample_slice_ends": [int(split.sample_slices[idx].stop) for idx in sample_indices],
    }
    np.savez_compressed(
        paths["arrays"],
        feature_matrix=split.feature_matrix,
        mobility_regression=split.mobility_regression,
        mobility_binary=split.mobility_binary,
    )
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _one_hot_aa(resname: str) -> np.ndarray:
    aa = RESNAME_TO_AA.get(resname.upper(), "X")
    one_hot = np.zeros(len(AA_ORDER), dtype=np.float32)
    one_hot[AA_TO_INDEX[aa]] = 1.0
    return one_hot


def _normalize_sequence_positions(length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros((length, 3), dtype=np.float32)
    positions = np.arange(length, dtype=np.float32)
    fraction = positions / float(length - 1)
    angles = 2.0 * np.pi * fraction
    return np.stack([fraction, np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)


def _local_geometry_features(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    length = len(coords)
    if length == 0:
        return np.zeros((0, 9), dtype=np.float32)
    centroid = coords.mean(axis=0, keepdims=True)
    centered = coords - centroid
    dist_centroid = np.linalg.norm(centered, axis=-1)
    prev_dist = np.zeros(length, dtype=np.float32)
    next_dist = np.zeros(length, dtype=np.float32)
    if length > 1:
        prev_dist[1:] = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)
        next_dist[:-1] = np.linalg.norm(coords[1:] - coords[:-1], axis=-1)
    local_radius_3 = np.zeros(length, dtype=np.float32)
    local_radius_5 = np.zeros(length, dtype=np.float32)
    curvature = np.zeros(length, dtype=np.float32)
    for i in range(length):
        win3 = coords[max(0, i - 3) : min(length, i + 4)]
        win5 = coords[max(0, i - 5) : min(length, i + 6)]
        local_radius_3[i] = float(np.linalg.norm(win3 - win3.mean(axis=0), axis=-1).mean()) if len(win3) > 1 else 0.0
        local_radius_5[i] = float(np.linalg.norm(win5 - win5.mean(axis=0), axis=-1).mean()) if len(win5) > 1 else 0.0
        if 0 < i < length - 1:
            a = coords[i] - coords[i - 1]
            b = coords[i + 1] - coords[i]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na > 1e-6 and nb > 1e-6:
                curvature[i] = float(np.dot(a, b) / (na * nb))
    return np.stack(
        [
            dist_centroid.astype(np.float32),
            prev_dist,
            next_dist,
            local_radius_3,
            local_radius_5,
            curvature,
        ],
        axis=1,
    )


def _parse_pdb_chain_residues(pdb_path: Path | str, chain_id: str) -> list[ChainResidue]:
    pdb_path = Path(pdb_path)
    residues: list[ChainResidue] = []
    residue_lookup: dict[tuple[str, str], int] = {}
    atom_store: list[dict[str, np.ndarray]] = []
    resname_store: list[str] = []
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            parsed_chain = line[21].strip() or "_"
            if parsed_chain != chain_id:
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C"}:
                continue
            residue_key = (line[22:26].strip(), line[26].strip())
            if residue_key not in residue_lookup:
                residue_lookup[residue_key] = len(atom_store)
                atom_store.append({})
                resname_store.append(line[17:20].strip().upper())
            idx = residue_lookup[residue_key]
            atom_store[idx][atom_name] = np.array(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=np.float32,
            )
    for residue_key, idx in residue_lookup.items():
        atoms = atom_store[idx]
        if "CA" not in atoms:
            continue
        residues.append(
            ChainResidue(
                residue_index=residue_key,
                resname=resname_store[idx],
                n=atoms.get("N"),
                ca=atoms["CA"],
                c=atoms.get("C"),
            )
        )
    return residues


def load_pdb_chain_residues(pdb_path: Path | str, chain_id: str) -> list[ChainResidue]:
    """Public wrapper around the lightweight PDB parser."""

    return _parse_pdb_chain_residues(pdb_path, chain_id)


def _select_residues(residues: list[ChainResidue], indices: Iterable[int]) -> list[ChainResidue]:
    selected = [residues[int(index)] for index in indices]
    if not selected:
        raise ValueError("No residues selected for mobility feature extraction")
    return selected


def _relative_position_features(length: int) -> np.ndarray:
    return _normalize_sequence_positions(length)


def _load_cached_esm_embedding(
    cache: EmbeddingCache,
    sequence_hash: str,
    *,
    log_once: set[str] | None = None,
) -> dict[str, Any]:
    if log_once is None or sequence_hash not in log_once:
        LOGGER.info("Loading ESM embeddings from cache: %s", sequence_hash)
        print(f"Loading ESM embeddings from cache: {sequence_hash}")
        if log_once is not None:
            log_once.add(sequence_hash)
    return cache.load(sequence_hash)


def build_residue_feature_matrix(
    sample: dict[str, Any],
    cache: EmbeddingCache,
    mobility_threshold: float = 2.0,
    log_once: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build one residue-level table from a paired inactive→active kinase sample."""

    source_info = sample["source_structure"]
    target_info = sample["target_structure"]
    source_residues = _parse_pdb_chain_residues(source_info["pdb_path"], source_info["chain"])
    target_residues = _parse_pdb_chain_residues(target_info["pdb_path"], target_info["chain"])
    source_selected = _select_residues(source_residues, source_info["residue_indices"])
    target_selected = _select_residues(target_residues, target_info["residue_indices"])
    if len(source_selected) != len(target_selected):
        raise ValueError(
            f"Aligned residue count mismatch for {sample['source_pdb_id']} -> {sample['target_pdb_id']}"
        )

    source_coords = np.stack([res.ca for res in source_selected], axis=0).astype(np.float32)
    target_coords = np.stack([res.ca for res in target_selected], axis=0).astype(np.float32)
    movement = np.linalg.norm(target_coords - source_coords, axis=-1).astype(np.float32)
    binary_mobile = (movement > mobility_threshold).astype(np.float32)

    embedding_payload = _load_cached_esm_embedding(cache, sample["source_sequence_hash"], log_once=log_once)
    residue_embeddings = embedding_payload["residue_embeddings"][list(source_info["residue_indices"])].float().numpy()

    aa_features = np.stack([_one_hot_aa(res.resname) for res in source_selected], axis=0)
    pos_features = _relative_position_features(len(source_selected))
    geom_features = _local_geometry_features(source_coords)
    if geom_features.shape[0] != len(source_selected):
        raise RuntimeError("Local geometry feature shape mismatch")
    feature_matrix = np.concatenate([residue_embeddings, aa_features, pos_features, geom_features], axis=1).astype(np.float32)

    rows = []
    for residue_index, (source_residue, target_residue) in enumerate(zip(source_selected, target_selected)):
        rows.append(
            {
                "kinase": sample["kinase"],
                "sample_index": sample.get("sample_index", -1),
                "residue_index": residue_index,
                "sequence_position": residue_index + 1,
                "residue_count": len(source_selected),
                "source_pdb_id": sample["source_pdb_id"],
                "target_pdb_id": sample["target_pdb_id"],
                "sequence_identity": float(sample["sequence_identity"]),
                "sequence_hash": sample["source_sequence_hash"],
                "residue_name": source_residue.resname,
                "movement": float(movement[residue_index]),
                "mobile": int(binary_mobile[residue_index]),
                "source_x": float(source_residue.ca[0]),
                "source_y": float(source_residue.ca[1]),
                "source_z": float(source_residue.ca[2]),
                "target_x": float(target_residue.ca[0]),
                "target_y": float(target_residue.ca[1]),
                "target_z": float(target_residue.ca[2]),
                "source_coord_norm": float(np.linalg.norm(source_residue.ca)),
                "target_coord_norm": float(np.linalg.norm(target_residue.ca)),
            }
        )

    metadata = pd.DataFrame.from_records(rows)
    return feature_matrix, movement, binary_mobile, metadata


def build_fold1_mobility_splits(
    project_root: Path | str = Path("."),
    folds_dir: Path | str | None = None,
    manifest_csv: Path | str | None = None,
    metadata_csv: Path | str | None = None,
    mobility_threshold: float = 2.0,
) -> dict[str, MobilityResidueSplit]:
    """Build residue-level mobility tables for the Fold 1 train/validation/test splits."""

    project_root = Path(project_root)
    cached_splits = {split_name: _load_split_cache(project_root, split_name) for split_name in ("train", "validation", "test")}
    if all(split is not None for split in cached_splits.values()):
        LOGGER.info("Loaded Fold 1 mobility residue tables from cache")
        return {split_name: split for split_name, split in cached_splits.items() if split is not None}

    folds_dir = Path(folds_dir or (project_root / "data" / "folds"))
    manifest_csv = Path(manifest_csv or (project_root / "data" / "esm_manifest.csv"))
    metadata_csv = Path(metadata_csv or (project_root / "data" / "metadata" / "kinase_labels.csv"))

    dataset_kwargs = dict(
        fold_id=1,
        project_root=project_root,
        folds_dir=folds_dir,
        manifest_csv=manifest_csv,
        metadata_csv=metadata_csv,
        load_structures=False,
    )
    splits = {}
    cache = EmbeddingCache(project_root / "cache" / "esm_embeddings")
    log_once: set[str] = set()
    for split_name in ("train", "validation", "test"):
        dataset = InactiveActiveTranslationDataset(split=split_name, **dataset_kwargs)
        feature_rows, regression_rows, binary_rows, metadata_rows = [], [], [], []
        sample_slices: dict[int, slice] = {}
        start = 0
        for sample_index, sample in enumerate(dataset):
            sample = dict(sample)
            sample["sample_index"] = sample_index
            feature_matrix, movement, binary_mobile, metadata = build_residue_feature_matrix(
                sample,
                cache=cache,
                mobility_threshold=mobility_threshold,
                log_once=log_once,
            )
            end = start + len(metadata)
            sample_slices[sample_index] = slice(start, end)
            start = end
            feature_rows.append(feature_matrix)
            regression_rows.append(movement)
            binary_rows.append(binary_mobile)
            metadata_rows.append(metadata)
        splits[split_name] = MobilityResidueSplit(
            split=split_name,
            metadata=pd.concat(metadata_rows, ignore_index=True),
            feature_matrix=np.concatenate(feature_rows, axis=0),
            mobility_regression=np.concatenate(regression_rows, axis=0),
            mobility_binary=np.concatenate(binary_rows, axis=0),
            sample_slices=sample_slices,
            embeddings_loaded_from_cache=True,
        )
        _save_split_cache(project_root, splits[split_name])
    return splits
