"""Inactive-to-active translation pairs built from existing LOKO folds."""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from torch.utils.data import DataLoader, Dataset

from models.conditioning import KINASE_TO_ID, STATE_TO_ID
from models.load_foldflow import configure_foldflow_import

from .esm import EmbeddingCache


def _load_tensor(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@lru_cache(maxsize=4096)
def _aligned_indices(source_sequence: str, target_sequence: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(source_sequence, target_sequence)[0]
    source_indices: list[int] = []
    target_indices: list[int] = []
    matches = 0
    for (source_start, source_end), (target_start, target_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        block_length = min(source_end - source_start, target_end - target_start)
        for offset in range(block_length):
            source_index = int(source_start + offset)
            target_index = int(target_start + offset)
            source_indices.append(source_index)
            target_indices.append(target_index)
            matches += source_sequence[source_index] == target_sequence[target_index]
    if not source_indices:
        raise ValueError("Source and target sequences have no aligned residues")
    identity = matches / len(source_indices)
    return tuple(source_indices), tuple(target_indices), float(identity)


@lru_cache(maxsize=4096)
def _backbone_complete_mask(pdb_path: str, chain_id: str) -> tuple[bool, ...]:
    """Return N/CA/C completeness in the same CA-residue order as the manifest."""
    residues: list[set[str]] = []
    residue_lookup: dict[tuple[str, str], int] = {}
    with Path(pdb_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            parsed_chain = line[21].strip() or "_"
            if parsed_chain != chain_id or line[16].strip() not in {"", "A"}:
                continue
            residue_key = (line[22:26].strip(), line[26].strip())
            if residue_key not in residue_lookup:
                residue_lookup[residue_key] = len(residues)
                residues.append(set())
            residues[residue_lookup[residue_key]].add(line[12:16].strip())
    return tuple(
        {"N", "CA", "C"}.issubset(atom_names)
        for atom_names in residues
        if "CA" in atom_names
    )


def _indices_have_complete_backbone(
    record: dict[str, Any], residue_indices: tuple[int, ...]
) -> bool:
    complete = _backbone_complete_mask(str(record["filepath"]), str(record["chain"]))
    return bool(residue_indices) and max(residue_indices) < len(complete) and all(
        complete[index] for index in residue_indices
    )


def _read_foldflow_structure(
    pdb_path: Path,
    chain_id: str,
    residue_indices: tuple[int, ...],
    source_root: Path | str | None,
) -> dict[str, torch.Tensor]:
    configure_foldflow_import(source_root)
    from openfold.data import data_transforms
    from openfold.np import residue_constants
    from openfold.utils import rigid_utils

    residues: list[dict[str, Any]] = []
    residue_lookup: dict[tuple[str, str], int] = {}
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
            residue_key = (line[22:26].strip(), line[26].strip())
            atom_name = line[12:16].strip()
            if atom_name not in residue_constants.atom_order:
                continue
            if residue_key not in residue_lookup:
                residue_lookup[residue_key] = len(residues)
                residues.append(
                    {
                        "resname": line[17:20].strip().upper(),
                        "atoms": {},
                    }
                )
            residue = residues[residue_lookup[residue_key]]
            residue["atoms"].setdefault(
                atom_name,
                np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float64,
                ),
            )

    ca_residues = [residue for residue in residues if "CA" in residue["atoms"]]
    selected = [ca_residues[index] for index in residue_indices]
    atom_positions = np.zeros(
        (len(selected), residue_constants.atom_type_num, 3), dtype=np.float64
    )
    atom_mask = np.zeros(
        (len(selected), residue_constants.atom_type_num), dtype=np.float64
    )
    aatype = np.zeros(len(selected), dtype=np.int64)
    for residue_index, residue in enumerate(selected):
        short_name = residue_constants.restype_3to1.get(residue["resname"], "X")
        aatype[residue_index] = residue_constants.restype_order.get(
            short_name, residue_constants.restype_num
        )
        for atom_name, coordinates in residue["atoms"].items():
            atom_index = residue_constants.atom_order[atom_name]
            atom_positions[residue_index, atom_index] = coordinates
            atom_mask[residue_index, atom_index] = 1.0

    required_atoms = [
        residue_constants.atom_order[name] for name in ("N", "CA", "C")
    ]
    valid_backbone = atom_mask[:, required_atoms].all(axis=1)
    if not valid_backbone.all():
        missing = np.where(~valid_backbone)[0].tolist()
        raise ValueError(
            f"Missing N/CA/C atoms in aligned residues {missing} from {pdb_path}"
        )

    features = {
        "aatype": torch.from_numpy(aatype).long(),
        "all_atom_positions": torch.from_numpy(atom_positions).double(),
        "all_atom_mask": torch.from_numpy(atom_mask).double(),
    }
    features = data_transforms.atom37_to_frames(features)
    features = data_transforms.atom37_to_torsion_angles()(features)
    rigids = rigid_utils.Rigid.from_tensor_4x4(features["rigidgroups_gt_frames"])[
        :, 0
    ].to_tensor_7()
    return {
        "rigids": rigids.float(),
        "atom37": features["all_atom_positions"].float(),
        "atom37_mask": features["all_atom_mask"].float(),
        "aatype": features["aatype"].long(),
        "torsion_angles_sin_cos": features["torsion_angles_sin_cos"].float(),
    }


class InactiveActiveTranslationDataset(Dataset):
    """Pair inactive sources with active targets inside one existing LOKO split.

    Each unique inactive PDB is paired to the active PDB of the same kinase with
    the highest deterministic sequence-similarity score. Global alignment then
    defines one-to-one residue correspondence. Both structures and the cached
    source ESM embedding are sliced to those aligned residues.
    """

    VALID_SPLITS = {"train", "validation", "test"}

    def __init__(
        self,
        fold_id: int,
        split: str = "train",
        project_root: Path | str = Path("."),
        folds_dir: Path | str = Path("data/folds"),
        manifest_csv: Path | str = Path("data/esm_manifest.csv"),
        metadata_csv: Path | str = Path("data/metadata/kinase_labels.csv"),
        min_sequence_identity: float = 0.7,
        source_root: Path | str | None = None,
        load_structures: bool = True,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}")
        if not 0.0 <= min_sequence_identity <= 1.0:
            raise ValueError("min_sequence_identity must be in [0, 1]")
        self.fold_id = int(fold_id)
        self.split = split
        self.project_root = Path(project_root)
        self.source_root = source_root
        self.load_structures = load_structures
        fold_path = Path(folds_dir) / f"fold_{fold_id:02d}" / f"{split}.csv"
        if not fold_path.is_file():
            raise FileNotFoundError(f"Existing LOKO fold not found: {fold_path}")
        fold = pd.read_csv(fold_path)
        self.fold_kinases = set(fold["kinase"].unique())
        manifest = pd.read_csv(manifest_csv)
        metadata = pd.read_csv(metadata_csv)
        metadata_paths = (
            metadata.rename(columns={"kinase_name": "kinase"})
            [["kinase", "pdb_id", "filepath"]]
            .drop_duplicates(["kinase", "pdb_id"])
        )
        rows = fold.merge(
            manifest,
            on=[
                "kinase",
                "pdb_id",
                "chain",
                "conformational_state",
                "sequence_hash",
                "embedding_file",
            ],
            how="left",
            validate="many_to_many",
        )
        rows = rows.merge(
            metadata_paths, on=["kinase", "pdb_id"], how="left", validate="many_to_one"
        )
        if rows[["sequence", "filepath"]].isna().any().any():
            raise ValueError("Fold rows could not be resolved to manifest sequences/PDB files")
        rows = rows.drop_duplicates(
            ["kinase", "pdb_id", "chain", "conformational_state", "sequence_hash"]
        )
        self.pairs, self.excluded_pairs = self._build_pairs(
            rows, min_sequence_identity
        )
        self.cache = EmbeddingCache(self.project_root / "cache/esm_embeddings")

    @staticmethod
    def _build_pairs(
        rows: pd.DataFrame, min_identity: float
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        pairs = []
        excluded = []
        for kinase, kinase_rows in rows.groupby("kinase", sort=True):
            inactive = kinase_rows[
                kinase_rows["conformational_state"] == "inactive"
            ].sort_values("pdb_id")
            active = kinase_rows[
                kinase_rows["conformational_state"] == "active"
            ].sort_values("pdb_id")
            if inactive.empty or active.empty:
                continue
            active_records = active.to_dict("records")
            for source in inactive.to_dict("records"):
                ranked_targets = sorted(
                    active_records,
                    key=lambda candidate: (
                        SequenceMatcher(
                            None, source["sequence"], candidate["sequence"]
                        ).ratio(),
                        -abs(len(source["sequence"]) - len(candidate["sequence"])),
                        candidate["pdb_id"],
                    ),
                    reverse=True,
                )
                selected = None
                rejection_reason = "sequence_identity"
                for target in ranked_targets:
                    source_indices, target_indices, identity = _aligned_indices(
                        source["sequence"], target["sequence"]
                    )
                    if identity < min_identity:
                        continue
                    rejection_reason = "incomplete_backbone"
                    if _indices_have_complete_backbone(
                        source, source_indices
                    ) and _indices_have_complete_backbone(target, target_indices):
                        selected = (
                            target,
                            source_indices,
                            target_indices,
                            identity,
                        )
                        break
                if selected is None:
                    excluded.append(
                        {
                            "kinase": kinase,
                            "source_pdb_id": source["pdb_id"],
                            "target_pdb_id": None,
                            "reason": rejection_reason,
                        }
                    )
                    continue
                target, source_indices, target_indices, identity = selected
                pairs.append(
                    {
                        "kinase": kinase,
                        "source": source,
                        "target": target,
                        "source_indices": source_indices,
                        "target_indices": target_indices,
                        "sequence_identity": identity,
                    }
                )
        return pairs, excluded

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        source = pair["source"]
        target = pair["target"]
        source_indices = pair["source_indices"]
        target_indices = pair["target_indices"]
        embedding_payload = self.cache.load(source["sequence_hash"])
        esm_embedding = embedding_payload["residue_embeddings"][
            list(source_indices)
        ].float()
        sample = {
            "kinase": pair["kinase"],
            "kinase_id": torch.tensor(KINASE_TO_ID[pair["kinase"]], dtype=torch.long),
            "target_state": torch.tensor(STATE_TO_ID["active"], dtype=torch.long),
            "esm_embedding": esm_embedding,
            "source_pdb_id": source["pdb_id"],
            "target_pdb_id": target["pdb_id"],
            "source_sequence_hash": source["sequence_hash"],
            "target_sequence_hash": target["sequence_hash"],
            "sequence_identity": pair["sequence_identity"],
            "residue_count": len(source_indices),
        }
        if self.load_structures:
            sample["source_structure"] = _read_foldflow_structure(
                Path(source["filepath"]),
                source["chain"],
                source_indices,
                self.source_root,
            )
            sample["target_structure"] = _read_foldflow_structure(
                Path(target["filepath"]),
                target["chain"],
                target_indices,
                self.source_root,
            )
        else:
            sample["source_structure"] = {
                "pdb_path": str(source["filepath"]),
                "chain": source["chain"],
                "residue_indices": source_indices,
            }
            sample["target_structure"] = {
                "pdb_path": str(target["filepath"]),
                "chain": target["chain"],
                "residue_indices": target_indices,
            }
        return sample

    def validate_fold_integrity(self) -> dict[str, Any]:
        pair_kinases = {pair["kinase"] for pair in self.pairs}
        unexpected = pair_kinases - self.fold_kinases
        if unexpected:
            raise ValueError(f"Translation pairs leaked kinases: {sorted(unexpected)}")
        return {
            "valid": True,
            "fold_id": self.fold_id,
            "split": self.split,
            "fold_kinases": sorted(self.fold_kinases),
            "paired_kinases": sorted(pair_kinases),
            "number_of_pairs": len(self.pairs),
            "excluded_pairs": len(self.excluded_pairs),
        }


def translation_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad aligned translation samples and construct official FoldFlow features."""
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    if "rigids" not in samples[0]["source_structure"]:
        raise ValueError("Dataset must use load_structures=True for batching")
    max_length = max(sample["residue_count"] for sample in samples)
    batch_size = len(samples)
    esm_dim = samples[0]["esm_embedding"].shape[-1]
    esm = torch.zeros(batch_size, max_length, esm_dim)
    res_mask = torch.zeros(batch_size, max_length)
    rigids_t = torch.zeros(batch_size, max_length, 7)
    rigids_0 = torch.zeros(batch_size, max_length, 7)
    rigids_t[..., 0] = 1.0
    rigids_0[..., 0] = 1.0
    torsion = torch.zeros(batch_size, max_length, 7, 2)
    target_atom37 = torch.zeros(batch_size, max_length, 37, 3)
    target_atom37_mask = torch.zeros(batch_size, max_length, 37)

    for batch_index, sample in enumerate(samples):
        length = sample["residue_count"]
        esm[batch_index, :length] = sample["esm_embedding"]
        res_mask[batch_index, :length] = 1.0
        rigids_t[batch_index, :length] = sample["source_structure"]["rigids"]
        rigids_0[batch_index, :length] = sample["target_structure"]["rigids"]
        torsion[batch_index, :length] = sample["target_structure"][
            "torsion_angles_sin_cos"
        ]
        target_atom37[batch_index, :length] = sample["target_structure"]["atom37"]
        target_atom37_mask[batch_index, :length] = sample["target_structure"][
            "atom37_mask"
        ]

    seq_idx = torch.arange(1, max_length + 1).expand(batch_size, -1)
    foldflow_features = {
        "rigids_t": rigids_t,
        "rigids_0": rigids_0,
        "res_mask": res_mask,
        "fixed_mask": torch.zeros_like(res_mask),
        "seq_idx": seq_idx,
        "t": torch.ones(batch_size),
        "sc_ca_t": torch.zeros(batch_size, max_length, 3),
        "torsion_angles_sin_cos": torsion,
        "atom37_pos": target_atom37,
        "atom37_mask": target_atom37_mask,
    }
    conditioning = {
        "esm_embedding": esm,
        "kinase_id": torch.stack([sample["kinase_id"] for sample in samples]),
        "target_state": torch.stack([sample["target_state"] for sample in samples]),
    }
    return {
        "foldflow_features": foldflow_features,
        "conditioning": conditioning,
        "metadata": [
            {
                key: sample[key]
                for key in (
                    "kinase",
                    "source_pdb_id",
                    "target_pdb_id",
                    "sequence_identity",
                    "residue_count",
                )
            }
            for sample in samples
        ],
    }


def create_translation_dataloader(
    fold_id: int,
    split: str,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    **dataset_kwargs,
) -> DataLoader:
    dataset = InactiveActiveTranslationDataset(
        fold_id=fold_id, split=split, **dataset_kwargs
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=translation_collate_fn,
    )
