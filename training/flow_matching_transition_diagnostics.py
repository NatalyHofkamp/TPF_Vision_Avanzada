#!/usr/bin/env python3
"""Diagnose and improve kinase conformational transition learning.

This experiment reads the existing CSV/PDB artifacts without modifying the
download or preprocessing pipelines. It restores chain, residue and sequence
information directly from PDB files, builds sequence-aligned endpoint pairs,
trains latent-path Flow Matching variants and writes an automatic root-cause
report. Intermediate conformations are generated hypotheses, never labels.

Examples
--------
python training/flow_matching_transition_diagnostics.py audit
python training/flow_matching_transition_diagnostics.py smoke
python training/flow_matching_transition_diagnostics.py run --epochs 50
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kinase_diagnostics_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kinase_diagnostics_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


LOG = logging.getLogger("transition_diagnostics")
SPLITS = ("train", "val", "test")
STATE_NAMES = ("active", "inactive")
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}


@dataclass
class DiagnosticConfig:
    seed: int = 42
    num_points: int = 192
    batch_size: int = 8
    hidden_dim: int = 128
    embedding_dim: int = 32
    num_layers: int = 4
    dropout: float = 0.1
    condition_dropout: float = 0.15
    guidance_scale: float = 2.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    integration_steps: int = 24
    trajectory_train_steps: int = 4
    max_pairs_per_kinase: int = 64
    min_alignment_coverage: float = 0.75
    min_sequence_identity: float = 0.85
    max_length_difference: float = 0.20
    max_resolution_difference: float = 1.0
    require_same_ligand_status: bool = True
    flow_weight: float = 1.0
    distance_weight: float = 0.15
    endpoint_weight: float = 0.35
    motif_weight: float = 0.30
    smoothness_weight: float = 0.05
    directionality_weight: float = 0.20
    self_consistency_weight: float = 0.15
    classifier_weight: float = 0.15
    use_kinase_condition: bool = True
    use_ligand_condition: bool = False
    use_resolution_condition: bool = False
    bidirectional_training: bool = False


@dataclass(frozen=True)
class Residue:
    chain: str
    number: int
    insertion: str
    name3: str
    aa: str
    ca: Tuple[float, float, float]
    atoms: Mapping[str, Tuple[float, float, float]]


@dataclass(frozen=True)
class StructureRecord:
    split: str
    pdb_id: str
    kinase: str
    state: str
    motif_descriptor: str
    dfg_state: str
    alphac_state: str
    ligand_present: int
    resolution: float
    chain: str
    residues: Tuple[Residue, ...]
    sequence: str
    missing_internal_ids: int
    dfg_index: Optional[int]
    hrd_index: Optional[int]
    vaik_lys_index: Optional[int]
    alphac_glu_index: Optional[int]


@dataclass(frozen=True)
class AlignedPair:
    split: str
    kinase: str
    source_pdb: str
    target_pdb: str
    source_state: str
    target_state: str
    source_chain: str
    target_chain: str
    aligned_source_indices: Tuple[int, ...]
    aligned_target_indices: Tuple[int, ...]
    aligned_residues: int
    alignment_coverage: float
    sequence_identity: float
    resolution_difference: float
    same_ligand_status: bool
    inferred_sequence_mismatches: int
    source_missing_ids: int
    target_missing_ids: int
    score: float


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


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return plt, sns


def discover_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents, Path(__file__).resolve().parents[1]]:
        if (
            (candidate / "data/metadata/kinase_labels.csv").exists()
            and (candidate / "data/splits/train.csv").exists()
            and (candidate / "data/raw/pdbs").is_dir()
        ):
            return candidate
    raise FileNotFoundError("Could not locate the KLIFS project data.")


def motif_descriptor(dfg: Any, alphac: Any) -> str:
    dfg_value = str(dfg).strip().lower()
    alphac_value = str(alphac).strip().lower()
    dfg_in = dfg_value == "in"
    dfg_out = dfg_value in {"out", "out-like"}
    alphac_in = alphac_value == "in"
    alphac_out = alphac_value == "out"
    if dfg_in and alphac_in:
        return "active"
    if dfg_out and alphac_out:
        return "inactive"
    if dfg_in and alphac_out:
        return "dfg_in_alphac_out"
    if dfg_out and alphac_in:
        return "dfg_out_alphac_in"
    return "unknown"


def parse_pdb_chains(path: Path) -> Dict[str, List[Residue]]:
    atoms: Dict[Tuple[str, int, str, str], Dict[str, Tuple[float, float, float]]] = {}
    order: List[Tuple[str, int, str, str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                continue
            atom = line[12:16].strip()
            name3 = line[17:20].strip().upper()
            if name3 not in AA3_TO_1:
                continue
            chain = line[21:22].strip() or "_"
            try:
                number = int(line[22:26])
                insertion = line[26:27].strip()
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            key = (chain, number, insertion, name3)
            if key not in atoms:
                atoms[key] = {}
                order.append(key)
            atoms[key].setdefault(atom, xyz)
    chains: Dict[str, List[Residue]] = defaultdict(list)
    for chain, number, insertion, name3 in order:
        residue_atoms = atoms[(chain, number, insertion, name3)]
        if "CA" not in residue_atoms:
            continue
        chains[chain].append(
            Residue(
                chain=chain,
                number=number,
                insertion=insertion,
                name3=name3,
                aa=AA3_TO_1[name3],
                ca=residue_atoms["CA"],
                atoms=residue_atoms,
            )
        )
    return dict(chains)


def motif_indices(sequence: str) -> Dict[str, Optional[int]]:
    dfg_match = re.search("DFG", sequence)
    hrd_match = re.search("HRD", sequence)
    vaik_matches = list(re.finditer(r"[VIL][A-Z][IVL]K", sequence))
    vaik_lys = vaik_matches[0].start() + 3 if vaik_matches else None
    alphac_glu = None
    if vaik_lys is not None:
        candidates = [index for index in range(max(0, vaik_lys - 40), vaik_lys - 5) if sequence[index] == "E"]
        if candidates:
            alphac_glu = candidates[-1]
    return {
        "dfg": dfg_match.start() if dfg_match else None,
        "hrd": hrd_match.start() if hrd_match else None,
        "vaik_lys": vaik_lys,
        "alphac_glu": alphac_glu,
    }


def count_internal_missing_ids(residues: Sequence[Residue]) -> int:
    missing = 0
    for left, right in zip(residues[:-1], residues[1:]):
        if left.chain == right.chain and not left.insertion and not right.insertion:
            missing += max(0, right.number - left.number - 1)
    return missing


def choose_chain(chains: Mapping[str, Sequence[Residue]]) -> Optional[str]:
    if not chains:
        return None
    motif_chains = [
        (chain, residues)
        for chain, residues in chains.items()
        if "DFG" in "".join(residue.aa for residue in residues)
    ]
    candidates = motif_chains or list(chains.items())
    return max(candidates, key=lambda item: len(item[1]))[0]


def load_records(root: Path) -> Tuple[Dict[str, List[StructureRecord]], pd.DataFrame]:
    records: Dict[str, List[StructureRecord]] = {}
    audit_rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        frame = pd.read_csv(root / "data/splits" / f"{split}.csv")
        frame["_pdb"] = frame["pdb_id"].astype(str).str.lower()
        frame = frame.sort_values(
            "resolution", key=lambda values: pd.to_numeric(values, errors="coerce")
        ).drop_duplicates("_pdb")
        split_records: List[StructureRecord] = []
        for _, row in frame.iterrows():
            pdb_id = str(row["pdb_id"]).lower()
            path = root / "data/raw/pdbs" / f"{pdb_id}.pdb"
            if not path.exists():
                audit_rows.append({"split": split, "pdb_id": pdb_id, "valid": False, "reason": "missing_pdb"})
                continue
            chains = parse_pdb_chains(path)
            chain = choose_chain(chains)
            if chain is None:
                audit_rows.append({"split": split, "pdb_id": pdb_id, "valid": False, "reason": "no_ca_chain"})
                continue
            residues = tuple(chains[chain])
            sequence = "".join(residue.aa for residue in residues)
            motifs = motif_indices(sequence)
            state = str(row["conformational_state"]).strip().lower()
            if state not in STATE_NAMES:
                state = "unknown"
            descriptor = motif_descriptor(row["dfg_state"], row["alphac_state"])
            record = StructureRecord(
                split=split,
                pdb_id=pdb_id,
                kinase=str(row["kinase_name"]),
                state=state,
                motif_descriptor=descriptor,
                dfg_state=str(row["dfg_state"]),
                alphac_state=str(row["alphac_state"]),
                ligand_present=int(row["ligand_present"]),
                resolution=float(row["resolution"]),
                chain=chain,
                residues=residues,
                sequence=sequence,
                missing_internal_ids=count_internal_missing_ids(residues),
                dfg_index=motifs["dfg"],
                hrd_index=motifs["hrd"],
                vaik_lys_index=motifs["vaik_lys"],
                alphac_glu_index=motifs["alphac_glu"],
            )
            split_records.append(record)
            audit_rows.append(
                {
                    "split": split,
                    "pdb_id": pdb_id,
                    "kinase": record.kinase,
                    "state": state,
                    "motif_descriptor": descriptor,
                    "chain": chain,
                    "chains_in_pdb": len(chains),
                    "residues": len(residues),
                    "missing_internal_ids": record.missing_internal_ids,
                    "dfg_found": record.dfg_index is not None,
                    "hrd_found": record.hrd_index is not None,
                    "vaik_lys_found": record.vaik_lys_index is not None,
                    "alphac_glu_found": record.alphac_glu_index is not None,
                    "valid": True,
                }
            )
        records[split] = split_records
    return records, pd.DataFrame(audit_rows)


def align_sequences(source: StructureRecord, target: StructureRecord) -> Tuple[Tuple[int, ...], Tuple[int, ...], float, float, int]:
    source_by_id = {
        (residue.number, residue.insertion): index
        for index, residue in enumerate(source.residues)
    }
    target_by_id = {
        (residue.number, residue.insertion): index
        for index, residue in enumerate(target.residues)
    }
    shared_ids = sorted(set(source_by_id) & set(target_by_id))
    id_pairs = [(source_by_id[key], target_by_id[key]) for key in shared_ids]
    id_coverage = len(id_pairs) / max(len(source.sequence), len(target.sequence), 1)
    if source.chain == target.chain and id_coverage >= 0.60:
        pairs = id_pairs
    else:
        matcher = difflib.SequenceMatcher(
            None, source.sequence, target.sequence, autojunk=False
        )
        pairs = []
        for block in matcher.get_matching_blocks():
            pairs.extend(
                (block.a + offset, block.b + offset)
                for offset in range(block.size)
            )
    source_indices = [left for left, _ in pairs]
    target_indices = [right for _, right in pairs]
    matches = sum(
        source.sequence[left] == target.sequence[right] for left, right in pairs
    )
    mismatches = len(pairs) - matches
    aligned = len(source_indices)
    coverage = aligned / max(len(source.sequence), len(target.sequence), 1)
    identity = matches / max(aligned, 1)
    return tuple(source_indices), tuple(target_indices), coverage, identity, mismatches


def pair_score(
    source: StructureRecord,
    target: StructureRecord,
    coverage: float,
    identity: float,
    mismatches: int,
) -> float:
    length_difference = abs(len(source.residues) - len(target.residues)) / max(
        len(source.residues), len(target.residues), 1
    )
    resolution_difference = abs(source.resolution - target.resolution)
    ligand_penalty = float(source.ligand_present != target.ligand_present)
    chain_penalty = float(source.chain != target.chain)
    missing_penalty = (source.missing_internal_ids + target.missing_internal_ids) / 50.0
    return (
        4.0 * (1.0 - coverage)
        + 4.0 * (1.0 - identity)
        + length_difference
        + 0.25 * resolution_difference
        + 0.5 * ligand_penalty
        + 0.25 * chain_penalty
        + 0.05 * mismatches
        + 0.05 * missing_penalty
    )


def build_aligned_pairs(
    records: Mapping[str, Sequence[StructureRecord]],
    source_states: Sequence[str],
    target_states: Sequence[str],
    config: DiagnosticConfig,
) -> Tuple[Dict[str, List[AlignedPair]], pd.DataFrame]:
    result: Dict[str, List[AlignedPair]] = {}
    report_rows: List[Dict[str, Any]] = []
    for split, split_records in records.items():
        selected: List[AlignedPair] = []
        by_kinase: Dict[str, List[StructureRecord]] = defaultdict(list)
        for record in split_records:
            by_kinase[record.kinase].append(record)
        for kinase, kinase_records in sorted(by_kinase.items()):
            sources = [record for record in kinase_records if record.state in source_states]
            targets = [record for record in kinase_records if record.state in target_states]
            candidates: List[AlignedPair] = []
            rejection_counts: Counter[str] = Counter()
            for source in sources:
                for target in targets:
                    source_indices, target_indices, coverage, identity, mismatches = align_sequences(source, target)
                    length_difference = abs(len(source.residues) - len(target.residues)) / max(
                        len(source.residues), len(target.residues), 1
                    )
                    resolution_difference = abs(source.resolution - target.resolution)
                    same_ligand = source.ligand_present == target.ligand_present
                    reasons = []
                    if coverage < config.min_alignment_coverage:
                        reasons.append("coverage")
                    if identity < config.min_sequence_identity:
                        reasons.append("identity")
                    if length_difference > config.max_length_difference:
                        reasons.append("length")
                    if resolution_difference > config.max_resolution_difference:
                        reasons.append("resolution")
                    if config.require_same_ligand_status and not same_ligand:
                        reasons.append("ligand")
                    if reasons:
                        rejection_counts.update(reasons)
                        continue
                    candidates.append(
                        AlignedPair(
                            split=split,
                            kinase=kinase,
                            source_pdb=source.pdb_id,
                            target_pdb=target.pdb_id,
                            source_state=source.state,
                            target_state=target.state,
                            source_chain=source.chain,
                            target_chain=target.chain,
                            aligned_source_indices=source_indices,
                            aligned_target_indices=target_indices,
                            aligned_residues=len(source_indices),
                            alignment_coverage=coverage,
                            sequence_identity=identity,
                            resolution_difference=resolution_difference,
                            same_ligand_status=same_ligand,
                            inferred_sequence_mismatches=mismatches,
                            source_missing_ids=source.missing_internal_ids,
                            target_missing_ids=target.missing_internal_ids,
                            score=pair_score(source, target, coverage, identity, mismatches),
                        )
                    )
            candidates.sort(key=lambda pair: pair.score)
            chosen = candidates[: config.max_pairs_per_kinase]
            selected.extend(chosen)
            report_rows.append(
                {
                    "split": split,
                    "kinase": kinase,
                    "source_states": "|".join(source_states),
                    "target_states": "|".join(target_states),
                    "source_structures": len(sources),
                    "target_structures": len(targets),
                    "candidate_pairs": len(sources) * len(targets),
                    "valid_pairs": len(candidates),
                    "selected_pairs": len(chosen),
                    "mean_aligned_residues": np.mean([pair.aligned_residues for pair in chosen]) if chosen else np.nan,
                    "mean_alignment_coverage": np.mean([pair.alignment_coverage for pair in chosen]) if chosen else np.nan,
                    "mean_sequence_identity": np.mean([pair.sequence_identity for pair in chosen]) if chosen else np.nan,
                    "same_chain_id_rate": np.mean(
                        [pair.source_chain == pair.target_chain for pair in chosen]
                    )
                    if chosen
                    else np.nan,
                    "same_ligand_rate": np.mean(
                        [pair.same_ligand_status for pair in chosen]
                    )
                    if chosen
                    else np.nan,
                    "mean_resolution_difference": np.mean(
                        [pair.resolution_difference for pair in chosen]
                    )
                    if chosen
                    else np.nan,
                    "mean_sequence_mismatches": np.mean(
                        [pair.inferred_sequence_mismatches for pair in chosen]
                    )
                    if chosen
                    else np.nan,
                    "mean_missing_residue_ids": np.mean(
                        [
                            pair.source_missing_ids + pair.target_missing_ids
                            for pair in chosen
                        ]
                    )
                    if chosen
                    else np.nan,
                    "rejections": dict(rejection_counts),
                }
            )
        result[split] = selected
    return result, pd.DataFrame(report_rows)


def record_lookup(records: Mapping[str, Sequence[StructureRecord]]) -> Dict[str, StructureRecord]:
    return {record.pdb_id: record for values in records.values() for record in values}


def kabsch_align(mobile: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mobile = mobile - mobile.mean(0, keepdim=True)
    target = target - target.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(mobile.T @ target)
    rotation = u @ vh
    if torch.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return mobile @ rotation


def motif_mask(
    target_record: StructureRecord,
    aligned_target_indices: Sequence[int],
    motif: str,
) -> torch.Tensor:
    aligned_map = {residue_index: output_index for output_index, residue_index in enumerate(aligned_target_indices)}
    indices: List[int] = []
    if motif == "dfg" and target_record.dfg_index is not None:
        indices = list(range(target_record.dfg_index - 2, target_record.dfg_index + 5))
    elif motif == "alphac" and target_record.alphac_glu_index is not None:
        indices = list(range(target_record.alphac_glu_index - 7, target_record.alphac_glu_index + 8))
    elif motif == "activation_loop" and target_record.dfg_index is not None:
        indices = list(range(target_record.dfg_index, target_record.dfg_index + 25))
    elif motif == "hrd" and target_record.hrd_index is not None:
        indices = list(range(target_record.hrd_index - 2, target_record.hrd_index + 5))
    output = torch.zeros(len(aligned_target_indices), dtype=torch.bool)
    for residue_index in indices:
        if residue_index in aligned_map:
            output[aligned_map[residue_index]] = True
    return output


def motif_preserving_indices(
    aligned_target_indices: Sequence[int],
    target_record: StructureRecord,
    num_points: int,
) -> torch.Tensor:
    mandatory_residues: set[int] = set()
    if target_record.dfg_index is not None:
        mandatory_residues.update(
            range(target_record.dfg_index - 2, target_record.dfg_index + 25)
        )
    if target_record.alphac_glu_index is not None:
        mandatory_residues.update(
            range(target_record.alphac_glu_index - 7, target_record.alphac_glu_index + 8)
        )
    if target_record.vaik_lys_index is not None:
        mandatory_residues.add(target_record.vaik_lys_index)
    if target_record.hrd_index is not None:
        mandatory_residues.update(
            range(target_record.hrd_index - 2, target_record.hrd_index + 5)
        )
    target_to_output = {
        residue_index: output_index
        for output_index, residue_index in enumerate(aligned_target_indices)
    }
    mandatory = sorted(
        target_to_output[index]
        for index in mandatory_residues
        if index in target_to_output
    )
    evenly_spaced = (
        torch.linspace(0, len(aligned_target_indices) - 1, num_points)
        .round()
        .long()
        .tolist()
    )
    selected = list(dict.fromkeys(mandatory + evenly_spaced))
    if len(selected) > num_points:
        mandatory_set = set(mandatory[:num_points])
        optional = [index for index in selected if index not in mandatory_set]
        selected = sorted(list(mandatory_set) + optional[: max(0, num_points - len(mandatory_set))])
    elif len(selected) < num_points:
        remaining = [
            index for index in range(len(aligned_target_indices)) if index not in set(selected)
        ]
        selected.extend(remaining[: num_points - len(selected)])
        selected.sort()
    return torch.tensor(selected[:num_points], dtype=torch.long)


def nearest_residue_mask(
    aligned_target_indices: Sequence[int], residue_index: Optional[int]
) -> torch.Tensor:
    output = torch.zeros(len(aligned_target_indices), dtype=torch.bool)
    if residue_index is None or not aligned_target_indices:
        return output
    position = min(
        range(len(aligned_target_indices)),
        key=lambda index: abs(aligned_target_indices[index] - residue_index),
    )
    output[position] = True
    return output


class AlignedDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[AlignedPair],
        lookup: Mapping[str, StructureRecord],
        config: DiagnosticConfig,
        kinase_vocab: Mapping[str, int],
        alignment_mode: str = "sequence",
    ):
        self.pairs = list(pairs)
        self.lookup = lookup
        self.config = config
        self.kinase_vocab = kinase_vocab
        self.alignment_mode = alignment_mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        source_record = self.lookup[pair.source_pdb]
        target_record = self.lookup[pair.target_pdb]
        if self.alignment_mode == "index":
            source_indices = tuple(range(len(source_record.residues)))
            target_indices = tuple(range(len(target_record.residues)))
            source = torch.tensor([residue.ca for residue in source_record.residues], dtype=torch.float32)
            target = torch.tensor([residue.ca for residue in target_record.residues], dtype=torch.float32)
            source = F.interpolate(
                source.T[None], size=self.config.num_points, mode="linear", align_corners=True
            )[0].T
            target = F.interpolate(
                target.T[None], size=self.config.num_points, mode="linear", align_corners=True
            )[0].T
            aligned_target_indices = tuple(
                torch.linspace(0, len(target_indices) - 1, self.config.num_points)
                .round()
                .long()
                .tolist()
            )
        else:
            source_indices = pair.aligned_source_indices
            target_indices = pair.aligned_target_indices
            aligned_target_indices = target_indices
            source = torch.tensor(
                [source_record.residues[i].ca for i in source_indices],
                dtype=torch.float32,
            )
            target = torch.tensor(
                [target_record.residues[i].ca for i in target_indices],
                dtype=torch.float32,
            )
        source = source - source.mean(0, keepdim=True)
        target = kabsch_align(target, source)
        scale = source.square().sum(-1).mean().sqrt().clamp_min(1e-6)
        source, target = source / scale, target / scale
        if len(source) > self.config.num_points:
            keep = motif_preserving_indices(
                aligned_target_indices, target_record, self.config.num_points
            )
            source, target = source[keep], target[keep]
            aligned_target_indices = tuple(
                aligned_target_indices[int(position)] for position in keep
            )
        masks = {
            motif: motif_mask(target_record, aligned_target_indices, motif)
            for motif in ("dfg", "alphac", "hrd", "activation_loop")
        }
        lys_mask = nearest_residue_mask(
            aligned_target_indices, target_record.vaik_lys_index
        )
        glu_mask = nearest_residue_mask(
            aligned_target_indices, target_record.alphac_glu_index
        )
        dfg_asp_mask = nearest_residue_mask(
            aligned_target_indices, target_record.dfg_index
        )
        dfg_phe_mask = nearest_residue_mask(
            aligned_target_indices,
            target_record.dfg_index + 1
            if target_record.dfg_index is not None
            else None,
        )
        length = min(len(source), self.config.num_points)
        padding = self.config.num_points - len(source)
        valid_mask = torch.ones(len(source), dtype=torch.bool)
        if padding:
            source = F.pad(source, (0, 0, 0, padding))
            target = F.pad(target, (0, 0, 0, padding))
            valid_mask = F.pad(valid_mask, (0, padding), value=False)
            masks = {name: F.pad(mask, (0, padding), value=False) for name, mask in masks.items()}
            lys_mask = F.pad(lys_mask, (0, padding), value=False)
            glu_mask = F.pad(glu_mask, (0, padding), value=False)
            dfg_asp_mask = F.pad(dfg_asp_mask, (0, padding), value=False)
            dfg_phe_mask = F.pad(dfg_phe_mask, (0, padding), value=False)
        state_id = STATE_NAMES.index(pair.target_state)
        return {
            "x0": source,
            "x1": target,
            "mask": valid_mask,
            "dfg_mask": masks["dfg"],
            "alphac_mask": masks["alphac"],
            "hrd_mask": masks["hrd"],
            "activation_loop_mask": masks["activation_loop"],
            "lys_mask": lys_mask,
            "glu_mask": glu_mask,
            "dfg_asp_mask": dfg_asp_mask,
            "dfg_phe_mask": dfg_phe_mask,
            "target_state": torch.tensor(state_id),
            "kinase": torch.tensor(self.kinase_vocab.get(pair.kinase, 0)),
            "ligand": torch.tensor(target_record.ligand_present),
            "resolution": torch.tensor(target_record.resolution, dtype=torch.float32),
            "scale": scale,
            "kinase_name": pair.kinase,
            "source_pdb": pair.source_pdb,
            "target_pdb": pair.target_pdb,
            "source_state_name": pair.source_state,
            "target_state_name": pair.target_state,
            "aligned_residues": pair.aligned_residues,
            "alignment_coverage": pair.alignment_coverage,
            "sequence_identity": pair.sequence_identity,
            "length": length,
        }


class DiagnosticFlowField(nn.Module):
    def __init__(self, config: DiagnosticConfig, num_kinases: int):
        super().__init__()
        self.config = config
        e = config.embedding_dim
        self.state_embedding = nn.Embedding(len(STATE_NAMES) + 1, e)
        self.kinase_embedding = nn.Embedding(max(1, num_kinases), e)
        self.ligand_embedding = nn.Embedding(2, e)
        self.time_embedding = nn.Sequential(nn.Linear(3, e), nn.SiLU(), nn.Linear(e, e))
        condition_parts = 2
        condition_parts += int(config.use_kinase_condition)
        condition_parts += int(config.use_ligand_condition)
        condition_parts += int(config.use_resolution_condition)
        self.input = nn.Linear(3 + e * condition_parts + 2, config.hidden_dim)
        blocks = []
        for _ in range(config.num_layers):
            blocks.extend(
                [
                    nn.Conv1d(config.hidden_dim, config.hidden_dim, 5, padding=2),
                    nn.GroupNorm(8 if config.hidden_dim % 8 == 0 else 1, config.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        self.body = nn.Sequential(*blocks)
        self.velocity = nn.Conv1d(config.hidden_dim, 3, 1)
        self.classifier_encoder = nn.Sequential(
            nn.Conv1d(3, config.hidden_dim, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, 5, padding=2),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, len(STATE_NAMES)),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
        unconditional: bool = False,
        condition_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, points, _ = x.shape
        keep = torch.ones(batch_size, 1, device=x.device)
        if unconditional:
            keep.zero_()
        elif condition_dropout and self.config.condition_dropout:
            keep = (
                torch.rand(batch_size, 1, device=x.device) >= self.config.condition_dropout
            ).float()
        time_features = torch.stack(
            [t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1
        )
        conditions = [self.time_embedding(time_features), self.state_embedding(batch["target_state"]) * keep]
        if self.config.use_kinase_condition:
            conditions.append(self.kinase_embedding(batch["kinase"]) * keep)
        if self.config.use_ligand_condition:
            conditions.append(self.ligand_embedding(batch["ligand"]) * keep)
        if self.config.use_resolution_condition:
            resolution = batch["resolution"][:, None].expand(-1, self.config.embedding_dim)
            conditions.append((resolution / 5.0) * keep)
        context = torch.cat(conditions, dim=-1)[:, None].expand(-1, points, -1)
        position = torch.linspace(0, 1, points, device=x.device)
        position = torch.stack([position, torch.sin(2 * math.pi * position)], -1)
        position = position[None].expand(batch_size, -1, -1)
        hidden = self.input(torch.cat([x, context, position], dim=-1)).transpose(1, 2)
        hidden = self.body(hidden)
        velocity = self.velocity(hidden).transpose(1, 2)
        mask = batch["mask"].float()[:, None]
        pooled = (hidden * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        logits = self.classifier(pooled)
        return velocity, logits

    def classify(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.classifier_encoder(x.transpose(1, 2))
        weights = mask.float()[:, None]
        pooled = (hidden * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        return self.classifier(pooled)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def masked_distance_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_dist = torch.cdist(prediction, prediction)
    target_dist = torch.cdist(target, target)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    return masked_mean((pred_dist - target_dist).abs(), pair_mask)


def endpoint_state_ids(state_ids: torch.Tensor) -> torch.Tensor:
    active_id = STATE_NAMES.index("active")
    inactive_id = STATE_NAMES.index("inactive")
    return torch.where(
        state_ids == active_id,
        torch.full_like(state_ids, inactive_id),
        torch.full_like(state_ids, active_id),
    )


def reverse_batch(
    batch: Mapping[str, torch.Tensor], start: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    reversed_batch = dict(batch)
    reversed_batch["x0"] = batch["x1"] if start is None else start
    reversed_batch["x1"] = batch["x0"]
    reversed_batch["target_state"] = endpoint_state_ids(batch["target_state"])
    return reversed_batch


def differentiable_rollout(
    model: DiagnosticFlowField,
    batch: Mapping[str, torch.Tensor],
    steps: int,
) -> List[torch.Tensor]:
    x = batch["x0"]
    states = [x]
    dt = 1.0 / steps
    for step in range(steps):
        time = torch.full(
            (x.shape[0],), step / steps, device=x.device, dtype=x.dtype
        )
        velocity, _ = model(x, time, batch, condition_dropout=False)
        x = x + dt * velocity * batch["mask"][:, :, None]
        states.append(x)
    return states


def trajectory_regularizers(
    states: Sequence[torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0, x1, mask = batch["x0"], batch["x1"], batch["mask"]
    distance0 = torch.cdist(x0, x0)
    distance1 = torch.cdist(x1, x1)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    geometry_terms = []
    motif_terms = []
    source_distances = []
    target_distances = []
    motif_mask = (
        batch["dfg_mask"]
        | batch["alphac_mask"]
        | batch["hrd_mask"]
        | batch["activation_loop_mask"]
    )
    motif_pair_mask = motif_mask[:, :, None] & motif_mask[:, None, :]
    denominator = max(len(states) - 1, 1)
    for index, state in enumerate(states):
        fraction = index / denominator
        expected_distance = (1 - fraction) * distance0 + fraction * distance1
        state_distance = torch.cdist(state, state)
        geometry_terms.append(
            masked_mean((state_distance - expected_distance).abs(), pair_mask)
        )
        motif_terms.append(
            masked_mean(
                (state_distance - expected_distance).abs(), motif_pair_mask
            )
        )
        source_distances.append(
            masked_mean((state - x0).square().sum(-1), mask)
        )
        target_distances.append(
            masked_mean((state - x1).square().sum(-1), mask)
        )
    directionality_terms = []
    for index in range(1, len(states)):
        directionality_terms.append(
            F.relu(target_distances[index] - target_distances[index - 1])
            + F.relu(source_distances[index - 1] - source_distances[index])
        )
    geometry = torch.stack(geometry_terms).mean()
    motif = torch.stack(motif_terms).mean()
    directionality = (
        torch.stack(directionality_terms).mean()
        if directionality_terms
        else geometry.new_zeros(())
    )
    return geometry, motif, directionality


def compute_losses(
    model: DiagnosticFlowField,
    batch: Mapping[str, torch.Tensor],
    config: DiagnosticConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    x0, x1, mask = batch["x0"], batch["x1"], batch["mask"]
    t = torch.rand(x0.shape[0], device=x0.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target_velocity = x1 - x0
    velocity, _ = model(xt, t, batch, condition_dropout=True)
    flow = masked_mean((velocity - target_velocity).square().sum(-1), mask)
    states = differentiable_rollout(
        model, batch, max(2, config.trajectory_train_steps)
    )
    endpoint = states[-1]
    endpoint_loss = masked_mean((endpoint - x1).square().sum(-1), mask)
    distance, motif, directionality = trajectory_regularizers(states, batch)
    accelerations = [
        states[index + 1] - 2 * states[index] + states[index - 1]
        for index in range(1, len(states) - 1)
    ]
    smoothness = masked_mean(
        torch.stack(accelerations).square().sum(-1), mask[None]
    ) if accelerations else flow.new_zeros(())
    if config.self_consistency_weight:
        recovered = differentiable_rollout(
            model,
            reverse_batch(batch, start=endpoint),
            max(2, config.trajectory_train_steps),
        )[-1]
        self_consistency = masked_mean(
            (recovered - x0).square().sum(-1), mask
        )
    else:
        self_consistency = flow.new_zeros(())
    classifier = (
        F.cross_entropy(model.classify(endpoint, mask), batch["target_state"])
        + F.cross_entropy(model.classify(x1, mask), batch["target_state"])
        + F.cross_entropy(
            model.classify(x0, mask), endpoint_state_ids(batch["target_state"])
        )
    ) / 3
    total = (
        config.flow_weight * flow
        + config.endpoint_weight * endpoint_loss
        + config.distance_weight * distance
        + config.motif_weight * motif
        + config.smoothness_weight * smoothness
        + config.directionality_weight * directionality
        + config.self_consistency_weight * self_consistency
        + config.classifier_weight * classifier
    )
    values = {
        "total_loss": float(total.detach()),
        "flow_loss": float(flow.detach()),
        "endpoint_loss": float(endpoint_loss.detach()),
        "distance_loss": float(distance.detach()),
        "motif_loss": float(motif.detach()),
        "smoothness_loss": float(smoothness.detach()),
        "directionality_loss": float(directionality.detach()),
        "self_consistency_loss": float(self_consistency.detach()),
        "classifier_loss": float(classifier.detach()),
    }
    return total, values


def move_batch(batch: Mapping[str, Any], device: str) -> Dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def integrate(
    model: DiagnosticFlowField,
    batch: Mapping[str, torch.Tensor],
    config: DiagnosticConfig,
) -> Dict[float, torch.Tensor]:
    x = batch["x0"].clone()
    result = {0.0: x.cpu()}
    requested = (0.25, 0.5, 0.75, 1.0)
    dt = 1.0 / config.integration_steps
    for step in range(config.integration_steps):
        time = torch.full((x.shape[0],), step / config.integration_steps, device=x.device)
        conditional, _ = model(x, time, batch)
        unconditional, _ = model(x, time, batch, unconditional=True)
        velocity = unconditional + config.guidance_scale * (conditional - unconditional)
        x = x + dt * velocity * batch["mask"][:, :, None]
        current = (step + 1) / config.integration_steps
        for requested_time in requested:
            if requested_time not in result and current + 1e-9 >= requested_time:
                result[requested_time] = x.cpu()
    return result


def local_rmsd(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    squared = (prediction - target).square().sum(-1)
    value = (squared * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    value = value.sqrt() * scale
    return torch.where(mask.sum(-1) > 0, value, torch.full_like(value, float("nan")))


def local_distance_error(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pred_dist = torch.cdist(prediction, prediction)
    target_dist = torch.cdist(target, target)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    error = ((pred_dist - target_dist).abs() * pair_mask).sum((1, 2))
    error = error / pair_mask.sum((1, 2)).clamp_min(1)
    return error * scale


def marked_distance(
    coords: torch.Tensor,
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    values = []
    for index in range(len(coords)):
        first = torch.where(first_mask[index])[0]
        second = torch.where(second_mask[index])[0]
        if len(first) == 0 or len(second) == 0:
            values.append(float("nan"))
        else:
            distance = torch.linalg.vector_norm(
                coords[index, first[0]] - coords[index, second[0]]
            )
            values.append(float(distance * scale[index]))
    return torch.tensor(values)


@torch.no_grad()
def evaluate(
    model: DiagnosticFlowField,
    dataset: AlignedDataset,
    config: DiagnosticConfig,
    output: Path,
    name: str,
) -> pd.DataFrame:
    device = device_name()
    data = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    rows: List[Dict[str, Any]] = []
    hypothesis_rows: List[Dict[str, Any]] = []
    model.eval()
    for raw in data:
        batch = move_batch(raw, device)
        paths = integrate(model, batch, config)
        generated = paths[1.0]
        x0, x1, mask = raw["x0"], raw["x1"], raw["mask"]
        scale = raw["scale"]
        global_rmsd = local_rmsd(generated, x1, mask, scale)
        initial_rmsd = local_rmsd(x0, x1, mask, scale)
        source_rmsd = local_rmsd(generated, x0, mask, scale)
        dfg_rmsd = local_rmsd(generated, x1, raw["dfg_mask"], scale)
        alphac_rmsd = local_rmsd(generated, x1, raw["alphac_mask"], scale)
        hrd_rmsd = local_rmsd(generated, x1, raw["hrd_mask"], scale)
        activation_loop_rmsd = local_rmsd(generated, x1, raw["activation_loop_mask"], scale)
        local_union = (
            raw["dfg_mask"]
            | raw["alphac_mask"]
            | raw["hrd_mask"]
            | raw["activation_loop_mask"]
        )
        local_dist = local_distance_error(generated, x1, local_union, scale)
        generated_lys_glu = marked_distance(
            generated, raw["lys_mask"], raw["glu_mask"], scale
        )
        target_lys_glu = marked_distance(
            x1, raw["lys_mask"], raw["glu_mask"], scale
        )
        generated_dfg_distance = marked_distance(
            generated, raw["dfg_asp_mask"], raw["dfg_phe_mask"], scale
        )
        target_dfg_distance = marked_distance(
            x1, raw["dfg_asp_mask"], raw["dfg_phe_mask"], scale
        )
        logits = model.classify(generated.to(device), batch["mask"])
        target_class = logits.argmax(-1).cpu()
        reversed_batch = reverse_batch(batch, start=generated.to(device))
        reverse_paths = integrate(model, reversed_batch, config)
        recovered = reverse_paths[1.0]
        cycle_rmsd = local_rmsd(recovered, x0, mask, scale)
        ordered_times = sorted(paths)
        path_stack = torch.stack([paths[time] for time in ordered_times], dim=1)
        accelerations = (
            path_stack[:, 2:] - 2 * path_stack[:, 1:-1] + path_stack[:, :-2]
        )
        smoothness = (
            accelerations.square().sum(-1).mean((1, 2)).sqrt() * scale
            if accelerations.shape[1]
            else torch.zeros(len(x0))
        )
        distance0 = torch.cdist(x0, x0)
        distance1 = torch.cdist(x1, x1)
        for index in range(len(x0)):
            final = float(global_rmsd[index])
            initial = float(initial_rmsd[index])
            source_curve = []
            target_curve = []
            geometry_curve = []
            for time in ordered_times:
                state = paths[time][index : index + 1]
                state_mask = mask[index : index + 1]
                state_scale = scale[index : index + 1]
                source_value = float(
                    local_rmsd(state, x0[index : index + 1], state_mask, state_scale)[0]
                )
                target_value = float(
                    local_rmsd(state, x1[index : index + 1], state_mask, state_scale)[0]
                )
                expected = (1 - time) * distance0[index] + time * distance1[index]
                state_distance = torch.cdist(state[0], state[0])
                pair_mask = state_mask[0, :, None] & state_mask[0, None, :]
                geometry_value = float(
                    ((state_distance - expected).abs() * pair_mask).sum()
                    / pair_mask.sum().clamp_min(1)
                    * state_scale[0]
                )
                source_curve.append(source_value)
                target_curve.append(target_value)
                geometry_curve.append(geometry_value)
                if time == 0.0:
                    supervision_status = "observed source endpoint"
                    is_observed_structure = True
                elif time == 1.0:
                    supervision_status = "generated endpoint supervised against observed target"
                    is_observed_structure = False
                else:
                    supervision_status = "latent generated hypothesis"
                    is_observed_structure = False
                hypothesis_rows.append(
                    {
                        "variant": name,
                        "kinase": raw["kinase_name"][index],
                        "source_pdb": raw["source_pdb"][index],
                        "target_pdb": raw["target_pdb"][index],
                        "t": time,
                        "distance_to_source": source_value,
                        "distance_to_target": target_value,
                        "distance_matrix_consistency_error": geometry_value,
                        "is_observed_structure": is_observed_structure,
                        "supervision_status": supervision_status,
                    }
                )
            monotonic_steps = [
                target_curve[position] <= target_curve[position - 1] + 1e-6
                and source_curve[position] >= source_curve[position - 1] - 1e-6
                for position in range(1, len(ordered_times))
            ]
            rows.append(
                {
                    "variant": name,
                    "kinase": raw["kinase_name"][index],
                    "source_pdb": raw["source_pdb"][index],
                    "target_pdb": raw["target_pdb"][index],
                    "source_state": raw["source_state_name"][index],
                    "target_state": raw["target_state_name"][index],
                    "aligned_residues": int(raw["aligned_residues"][index]),
                    "alignment_coverage": float(raw["alignment_coverage"][index]),
                    "sequence_identity": float(raw["sequence_identity"][index]),
                    "global_ca_rmsd": final,
                    "initial_target_rmsd": initial,
                    "relative_improvement": (initial - final) / max(initial, 1e-8),
                    "generated_to_source_rmsd": float(source_rmsd[index]),
                    "dfg_loop_rmsd": float(dfg_rmsd[index]),
                    "alphac_rmsd": float(alphac_rmsd[index]),
                    "hrd_rmsd": float(hrd_rmsd[index]),
                    "activation_loop_rmsd": float(activation_loop_rmsd[index]),
                    "local_distance_matrix_mae": float(local_dist[index]),
                    "lys_glu_ca_distance": float(generated_lys_glu[index]),
                    "target_lys_glu_ca_distance": float(target_lys_glu[index]),
                    "lys_glu_distance_error": float(
                        abs(generated_lys_glu[index] - target_lys_glu[index])
                    ),
                    "dfg_ca_distance": float(generated_dfg_distance[index]),
                    "target_dfg_ca_distance": float(target_dfg_distance[index]),
                    "dfg_distance_error": float(
                        abs(generated_dfg_distance[index] - target_dfg_distance[index])
                    ),
                    "target_state_predicted": STATE_NAMES[int(target_class[index])],
                    "target_state_correct": int(target_class[index]) == int(raw["target_state"][index]),
                    "closer_to_target_than_initial": final < initial,
                    "crosses_to_target": final < float(source_rmsd[index]),
                    "trajectory_smoothness": float(smoothness[index]),
                    "directionality_fraction": float(np.mean(monotonic_steps)),
                    "mean_path_geometry_error": float(np.mean(geometry_curve)),
                    "cycle_rmsd": float(cycle_rmsd[index]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics" / f"{name}_per_pair.csv", index=False)
    pd.DataFrame(hypothesis_rows).to_csv(
        output / "metrics" / f"{name}_latent_hypotheses.csv", index=False
    )
    return frame


def oracle_linear_baseline(
    dataset: AlignedDataset, output: Path
) -> pd.DataFrame:
    rows = []
    hypothesis_rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        x0, x1 = sample["x0"], sample["x1"]
        mask, scale = sample["mask"], sample["scale"]
        times = (0.0, 0.25, 0.5, 0.75, 1.0)
        states = [(1 - time) * x0 + time * x1 for time in times]
        distance0, distance1 = torch.cdist(x0, x0), torch.cdist(x1, x1)
        pair_mask = mask[:, None] & mask[None, :]
        initial = float(local_rmsd(x0[None], x1[None], mask[None], scale[None])[0])
        for time, state in zip(times, states):
            if time == 0.0:
                supervision_status = "observed source endpoint"
            elif time == 1.0:
                supervision_status = "observed target endpoint used by oracle"
            else:
                supervision_status = (
                    "oracle geometric interpolation using the observed target"
                )
            hypothesis_rows.append(
                {
                    "variant": "linear_interpolation_oracle",
                    "kinase": sample["kinase_name"],
                    "source_pdb": sample["source_pdb"],
                    "target_pdb": sample["target_pdb"],
                    "t": time,
                    "distance_to_source": float(
                        local_rmsd(
                            state[None], x0[None], mask[None], scale[None]
                        )[0]
                    ),
                    "distance_to_target": float(
                        local_rmsd(
                            state[None], x1[None], mask[None], scale[None]
                        )[0]
                    ),
                    "distance_matrix_consistency_error": float(
                        (
                            (
                                torch.cdist(state, state)
                                - ((1 - time) * distance0 + time * distance1)
                            ).abs()
                            * pair_mask
                        ).sum()
                        / pair_mask.sum().clamp_min(1)
                        * scale
                    ),
                    "is_observed_structure": time in {0.0, 1.0},
                    "supervision_status": supervision_status,
                }
            )
        rows.append(
            {
                "variant": "linear_interpolation_oracle",
                "global_ca_rmsd": 0.0,
                "initial_target_rmsd": initial,
                "relative_improvement": 1.0,
                "dfg_loop_rmsd": 0.0,
                "alphac_rmsd": 0.0,
                "hrd_rmsd": 0.0,
                "activation_loop_rmsd": 0.0,
                "local_distance_matrix_mae": 0.0,
                "target_state_correct": True,
                "closer_to_target_than_initial": True,
                "crosses_to_target": True,
                "trajectory_smoothness": 0.0,
                "directionality_fraction": 1.0,
                "mean_path_geometry_error": 0.0,
                "cycle_rmsd": 0.0,
            }
        )
    pd.DataFrame(hypothesis_rows).to_csv(
        output / "metrics/linear_interpolation_oracle_latent_hypotheses.csv",
        index=False,
    )
    return pd.DataFrame(rows)


def train_variant(
    config: DiagnosticConfig,
    train_dataset: AlignedDataset,
    val_dataset: AlignedDataset,
    output: Path,
    name: str,
) -> Tuple[DiagnosticFlowField, pd.DataFrame]:
    set_seed(config.seed)
    device = device_name()
    model = DiagnosticFlowField(config, len(train_dataset.kinase_vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    best = float("inf")
    stale = 0
    history = []
    model_dir = output / "models" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_values: Dict[str, List[float]] = defaultdict(list)
        for raw in train_loader:
            batch = move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, values = compute_losses(model, batch, config)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in values.items():
                train_values[key].append(value)
        model.eval()
        validation = []
        with torch.no_grad():
            for raw in val_loader:
                batch = move_batch(raw, device)
                loss, _ = compute_losses(model, batch, config)
                validation.append(float(loss))
        val_loss = float(np.mean(validation))
        row = {"epoch": epoch, "validation_loss": val_loss}
        row.update({f"train_{key}": float(np.mean(values)) for key, values in train_values.items()})
        history.append(row)
        LOG.info("%s epoch=%d train=%.5f val=%.5f", name, epoch, row["train_total_loss"], val_loss)
        checkpoint = {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "validation_loss": val_loss,
        }
        torch.save(checkpoint, model_dir / "last_model.pt")
        if val_loss < best:
            best = val_loss
            stale = 0
            torch.save(checkpoint, model_dir / "best_model.pt")
        else:
            stale += 1
        if stale >= config.patience:
            break
    best_checkpoint = torch.load(model_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(model_dir / "training_history.csv", index=False)
    json_dump(asdict(config), model_dir / "config.json")
    return model, history_frame


def variant_configs(base: DiagnosticConfig) -> Dict[str, DiagnosticConfig]:
    def changed(**values: Any) -> DiagnosticConfig:
        config = asdict(base)
        config.update(values)
        return DiagnosticConfig(**config)

    return {
        "current_index_matching": changed(
            distance_weight=0,
            endpoint_weight=0,
            motif_weight=0,
            smoothness_weight=0,
            directionality_weight=0,
            self_consistency_weight=0,
            classifier_weight=0,
        ),
        "aligned_endpoint_flow": changed(
            distance_weight=0,
            endpoint_weight=0,
            motif_weight=0,
            smoothness_weight=0,
            directionality_weight=0,
            self_consistency_weight=0,
            classifier_weight=0,
        ),
        "aligned_endpoint_constraints": changed(
            self_consistency_weight=0,
        ),
        "aligned_full_conditioning": changed(
            use_ligand_condition=True,
            use_resolution_condition=True,
            self_consistency_weight=0,
        ),
        "bidirectional_self_consistent": changed(
            bidirectional_training=True,
        ),
        "inverse_deactivation": changed(
            bidirectional_training=False,
        ),
    }


def diagnostic_holdout(
    pairs: Mapping[str, Sequence[AlignedPair]],
) -> Tuple[Dict[str, List[AlignedPair]], Optional[str]]:
    result = {split: list(values) for split, values in pairs.items()}
    if result["test"]:
        return result, None
    if len(result["val"]) < 2:
        return result, "No test pairs and insufficient validation pairs for a diagnostic holdout."
    validation = sorted(result["val"], key=lambda pair: (pair.kinase, pair.score))
    result["val"] = validation[::2]
    result["test"] = validation[1::2]
    return (
        result,
        "The original test split has no strict inactive structures. Validation "
        "pairs were deterministically divided into validation and diagnostic "
        "holdout subsets; this is not an independent kinase-level test.",
    )


def mirror_pairs(
    pairs: Mapping[str, Sequence[AlignedPair]]
) -> Dict[str, List[AlignedPair]]:
    mirrored: Dict[str, List[AlignedPair]] = {}
    for split, values in pairs.items():
        mirrored[split] = [
            AlignedPair(
                split=pair.split,
                kinase=pair.kinase,
                source_pdb=pair.target_pdb,
                target_pdb=pair.source_pdb,
                source_state=pair.target_state,
                target_state=pair.source_state,
                source_chain=pair.target_chain,
                target_chain=pair.source_chain,
                aligned_source_indices=pair.aligned_target_indices,
                aligned_target_indices=pair.aligned_source_indices,
                aligned_residues=pair.aligned_residues,
                alignment_coverage=pair.alignment_coverage,
                sequence_identity=pair.sequence_identity,
                resolution_difference=pair.resolution_difference,
                same_ligand_status=pair.same_ligand_status,
                inferred_sequence_mismatches=pair.inferred_sequence_mismatches,
                source_missing_ids=pair.target_missing_ids,
                target_missing_ids=pair.source_missing_ids,
                score=pair.score,
            )
            for pair in values
        ]
    return mirrored


def summarize_variants(frames: Sequence[pd.DataFrame], output: Path) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby("variant")
        .agg(
            pairs=("global_ca_rmsd", "size"),
            global_ca_rmsd=("global_ca_rmsd", "mean"),
            improvement=("relative_improvement", "mean"),
            dfg_rmsd=("dfg_loop_rmsd", "mean"),
            alphac_rmsd=("alphac_rmsd", "mean"),
            hrd_rmsd=("hrd_rmsd", "mean"),
            activation_loop_rmsd=("activation_loop_rmsd", "mean"),
            local_distance_error=("local_distance_matrix_mae", "mean"),
            state_accuracy=("target_state_correct", "mean"),
            closer_to_target_rate=("closer_to_target_than_initial", "mean"),
            crossing_rate=("crosses_to_target", "mean"),
            trajectory_smoothness=("trajectory_smoothness", "mean"),
            directionality=("directionality_fraction", "mean"),
            path_geometry_error=("mean_path_geometry_error", "mean"),
            cycle_rmsd=("cycle_rmsd", "mean"),
        )
        .reset_index()
        .sort_values("global_ca_rmsd")
    )
    summary.to_csv(output / "diagnostic_variant_comparison.csv", index=False)
    baseline = summary[summary["variant"] == "current_index_matching"]
    effects = []
    if not baseline.empty:
        baseline_rmsd = float(baseline.iloc[0]["global_ca_rmsd"])
        for _, row in summary.iterrows():
            if row["variant"] == "linear_interpolation_oracle":
                continue
            effects.append(
                {
                    "variant": row["variant"],
                    "rmsd_change_vs_current_index": float(row["global_ca_rmsd"])
                    - baseline_rmsd,
                    "rmsd_reduction_vs_current_index": baseline_rmsd
                    - float(row["global_ca_rmsd"]),
                    "directly_comparable_pair_set": row["variant"]
                    in {
                        "current_index_matching",
                        "aligned_endpoint_flow",
                        "aligned_endpoint_constraints",
                        "aligned_full_conditioning",
                        "bidirectional_self_consistent",
                    },
                }
            )
    pd.DataFrame(effects).to_csv(output / "diagnostic_variant_effects.csv", index=False)
    plt, sns = pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    sns.barplot(data=summary, x="variant", y="global_ca_rmsd", ax=axes[0])
    sns.barplot(data=summary, x="variant", y="improvement", ax=axes[1])
    sns.barplot(data=summary, x="variant", y="state_accuracy", ax=axes[2])
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    axes[0].set_title("Global C-alpha RMSD")
    axes[1].set_title("Relative improvement")
    axes[2].set_title("Generated state classification")
    figure.tight_layout()
    figure.savefig(output / "plots" / "diagnostic_variant_comparison.png", dpi=200)
    plt.close(figure)
    return summary


def write_historical_baseline(root: Path, output: Path) -> None:
    path = (
        root
        / "results/flow_matching_cfg_transition_optuna/metrics/checkpoint_per_pair.csv"
    )
    if not path.exists():
        return
    frame = pd.read_csv(path)
    if "rmsd_generated_to_active" not in frame:
        return
    pd.DataFrame(
        [
            {
                "variant": "historical_binary_index_model",
                "pairs": len(frame),
                "global_ca_rmsd": frame["rmsd_generated_to_active"].mean(),
                "relative_improvement": frame["relative_improvement"].mean(),
                "closer_to_target_rate": (
                    frame["rmsd_generated_to_active"]
                    < frame["rmsd_initial_to_active"]
                ).mean(),
                "note": (
                    "Historical reference only: binary labels, normalized-index "
                    "matching and a different PDGFRA test-pair definition."
                ),
            }
        ]
    ).to_csv(output / "historical_current_model_reference.csv", index=False)


def write_root_cause_report(
    structure_audit: pd.DataFrame,
    pair_report: pd.DataFrame,
    endpoint_counts: pd.DataFrame,
    descriptor_counts: pd.DataFrame,
    summary: Optional[pd.DataFrame],
    output: Path,
) -> None:
    valid = structure_audit[structure_audit["valid"] == True]  # noqa: E712
    motif_coverage = {
        column: float(valid[column].mean())
        for column in ("dfg_found", "hrd_found", "vaik_lys_found", "alphac_glu_found")
    }
    selected = pair_report[pair_report["selected_pairs"] > 0]
    mean_coverage = float(selected["mean_alignment_coverage"].mean()) if not selected.empty else float("nan")
    mean_identity = float(selected["mean_sequence_identity"].mean()) if not selected.empty else float("nan")
    inactive_count = int(
        endpoint_counts.loc[endpoint_counts["state"] == "inactive", "count"].sum()
    )
    active_count = int(
        endpoint_counts.loc[endpoint_counts["state"] == "active", "count"].sum()
    )
    descriptor_heterogeneity = int(
        descriptor_counts["motif_descriptor"].nunique()
    )
    data_problem = inactive_count == 0 or active_count == 0
    alignment_problem = not np.isfinite(mean_coverage) or mean_coverage < 0.85
    pairing_problem = selected["selected_pairs"].sum() < 20 if not selected.empty else True
    best_text = "No model comparison has been run."
    best_change_text = "No comparable variant effects are available."
    architecture_problem = "undetermined"
    supervision_problem = (
        "not a labeling problem: intermediates are intentionally latent"
    )
    if summary is not None and not summary.empty:
        learned_summary = summary[
            summary["variant"] != "linear_interpolation_oracle"
        ]
        best = learned_summary.sort_values("global_ca_rmsd").iloc[0]
        if float(learned_summary["improvement"].max()) <= 0:
            best_text = (
                "No variant learned a target-directed transition in this run. "
                f"The lowest RMSD was {best['variant']} at "
                f"{best['global_ca_rmsd']:.3f} A, but its improvement remained "
                f"negative ({100 * best['improvement']:.2f}%)."
            )
        else:
            best_text = (
                f"Best variant: {best['variant']} with global RMSD "
                f"{best['global_ca_rmsd']:.3f} A and improvement "
                f"{100 * best['improvement']:.2f}%."
            )
        aligned = summary[summary["variant"] == "aligned_endpoint_flow"]
        local = summary[summary["variant"] == "aligned_endpoint_constraints"]
        current = summary[summary["variant"] == "current_index_matching"]
        comparable = summary[
            summary["variant"].isin(
                [
                    "current_index_matching",
                    "aligned_endpoint_flow",
                    "aligned_endpoint_constraints",
                    "aligned_full_conditioning",
                    "bidirectional_self_consistent",
                ]
            )
        ]
        if not current.empty and not comparable.empty:
            winner = comparable.sort_values("global_ca_rmsd").iloc[0]
            reduction = float(current.iloc[0]["global_ca_rmsd"]) - float(
                winner["global_ca_rmsd"]
            )
            best_change_text = (
                f"Among directly comparable variants, {winner['variant']} gives "
                f"the largest RMSD reduction versus index matching: "
                f"{reduction:.3f} A."
            )
        if not aligned.empty and not local.empty:
            architecture_problem = (
                "auxiliary/local supervision improves the same architecture"
                if float(local.iloc[0]["global_ca_rmsd"]) < float(aligned.iloc[0]["global_ca_rmsd"])
                else "auxiliary losses did not improve the architecture in this run"
            )
    text = f"""# Automatic transition-learning diagnosis

## Main findings

- Endpoint availability problem: **{'yes' if data_problem else 'not dominant'}**.
  Known inactive endpoints: {inactive_count}; known active endpoints: {active_count}.
- Pairing problem: **{'yes' if pairing_problem else 'controlled after filtering'}**.
- Alignment problem: **{'yes' if alignment_problem else 'substantially reduced'}**.
  Mean selected-pair coverage: {mean_coverage:.3f}; identity: {mean_identity:.3f}.
- Architecture/local supervision: **{architecture_problem}**.
- Intermediate supervision: **{supervision_problem}**.
- DFG/alphaC descriptor combinations observed: {descriptor_heterogeneity};
  these are analysis descriptors, not supervised intermediate labels.
- {best_text}
- {best_change_text}

## Why the original model failed

1. Normalized-index resampling paired residues that were not guaranteed to be
   homologous, corrupting the velocity target and local geometry.
2. Random same-kinase pairing ignored resolution, ligand status, sequence
   mismatches and missing residues.
3. C-alpha-only tensors removed chain IDs, residue IDs, sequence and side-chain
   atoms needed for motif-aware supervision.
4. Flow loss alone did not explicitly enforce endpoint geometry, local motifs,
   directionality, path smoothness or cycle consistency.
5. The generated intermediate conformations have no experimental ground truth;
   they must be evaluated as latent hypotheses rather than classified as correct
   or incorrect structures.
6. Distance-matrix targets along the path are endpoint-derived geometric
   regularizers, not experimentally observed intermediate geometries.

## Available annotation quality

- DFG motif detection: {100 * motif_coverage['dfg_found']:.1f}%.
- HRD motif detection: {100 * motif_coverage['hrd_found']:.1f}%.
- VAIK Lys detection: {100 * motif_coverage['vaik_lys_found']:.1f}%.
- alphaC Glu proxy detection: {100 * motif_coverage['alphac_glu_found']:.1f}%.
- KLIFS pocket-position mappings are not present locally; sequence alignment is
  used as the best available fallback.
- Explicit mutation annotations are absent. Sequence mismatches are reported as
  possible mutations or construct differences.
- Lys-Glu generated-state distance can only be approximated with C-alpha
  coordinates unless an all-atom model is introduced.

## Recommended next decision

Use `diagnostic_variant_comparison.csv` to select the change with the largest
reduction in endpoint/local RMSD and the best smoothness, directionality,
geometry and cycle metrics. Do not claim that a generated path is the real
activation mechanism without molecular simulation or experimental validation.

## Epistemic status of intermediate conformations

Only t=0 and t=1 are observed structures. States generated at t=0.25, 0.50 and
0.75 are model hypotheses constrained by endpoint data and geometric losses.
They are not labels, experimentally observed intermediates or ground truth.
For learned trajectories, the t=1 row is a generated endpoint prediction
evaluated against the observed target; it is not itself an observed structure.
Success means coherent geometry, stable motifs, monotonic progress and endpoint
recovery; it does not establish the biological mechanism.
"""
    (output / "diagnostic_report.md").write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else discover_root(Path.cwd())
    output = (
        Path(args.output).resolve()
        if args.output
        else root / "results" / "flow_matching_latent_transition_diagnostics"
    )
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    config = DiagnosticConfig(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        num_points=args.num_points,
        max_pairs_per_kinase=args.max_pairs_per_kinase,
    )
    records, structure_audit = load_records(root)
    structure_audit.to_csv(output / "structure_annotation_audit.csv", index=False)
    endpoint_counts = (
        pd.DataFrame(
            [
                {"split": split, "state": state, "count": count}
                for split, values in records.items()
                for state, count in Counter(record.state for record in values).items()
            ]
        )
        .sort_values(["split", "state"])
    )
    endpoint_counts.to_csv(output / "endpoint_state_counts.csv", index=False)
    descriptor_counts = (
        structure_audit[structure_audit["valid"] == True]  # noqa: E712
        .groupby(["split", "motif_descriptor"])
        .size()
        .reset_index(name="count")
    )
    descriptor_counts.to_csv(output / "dfg_alphac_descriptor_counts.csv", index=False)
    direct_pairs, direct_report = build_aligned_pairs(
        records, ["inactive"], ["active"], config
    )
    direct_pairs, direct_holdout_note = diagnostic_holdout(direct_pairs)
    direct_report.to_csv(output / "direct_pairing_alignment_report.csv", index=False)
    inverse_pairs, inverse_report = build_aligned_pairs(
        records, ["active"], ["inactive"], config
    )
    inverse_pairs, inverse_holdout_note = diagnostic_holdout(inverse_pairs)
    inverse_report.to_csv(output / "inverse_pairing_alignment_report.csv", index=False)
    write_root_cause_report(
        structure_audit,
        direct_report,
        endpoint_counts,
        descriptor_counts,
        None,
        output,
    )
    json_dump(asdict(config), output / "diagnostic_config.json")
    write_historical_baseline(root, output)
    json_dump(
        {
            "direct": direct_holdout_note,
            "inverse": inverse_holdout_note,
        },
        output / "evaluation_split_notes.json",
    )
    LOG.info(
        "records=%s direct_pairs=%s inverse_pairs=%s",
        {split: len(values) for split, values in records.items()},
        {split: len(values) for split, values in direct_pairs.items()},
        {split: len(values) for split, values in inverse_pairs.items()},
    )
    if args.command == "audit":
        return 0
    lookup = record_lookup(records)
    kinase_vocab = {
        kinase: index
        for index, kinase in enumerate(
            sorted({record.kinase for record in records["train"]}), start=0
        )
    }
    variants = variant_configs(config)
    frames = []
    for name, variant_config in variants.items():
        pair_set = inverse_pairs if name == "inverse_deactivation" else direct_pairs
        if variant_config.bidirectional_training:
            mirrored = mirror_pairs(direct_pairs)
            pair_set = {
                split: direct_pairs[split] + mirrored[split]
                for split in SPLITS
            }
        if not all(pair_set[split] for split in SPLITS):
            LOG.warning("Skipping %s because one or more splits have no valid pairs.", name)
            continue
        alignment_mode = "index" if name == "current_index_matching" else "sequence"
        train_dataset = AlignedDataset(
            pair_set["train"], lookup, variant_config, kinase_vocab, alignment_mode
        )
        val_dataset = AlignedDataset(
            pair_set["val"], lookup, variant_config, kinase_vocab, alignment_mode
        )
        test_dataset = AlignedDataset(
            pair_set["test"], lookup, variant_config, kinase_vocab, alignment_mode
        )
        model, _ = train_variant(
            variant_config, train_dataset, val_dataset, output, name
        )
        frames.append(evaluate(model, test_dataset, variant_config, output, name))
    if direct_pairs["test"]:
        oracle_dataset = AlignedDataset(
            direct_pairs["test"], lookup, config, kinase_vocab, "sequence"
        )
        frames.append(oracle_linear_baseline(oracle_dataset, output))
    summary = summarize_variants(frames, output) if frames else None
    write_root_cause_report(
        structure_audit,
        direct_report,
        endpoint_counts,
        descriptor_counts,
        summary,
        output,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "smoke", "run"])
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-points", type=int, default=192)
    parser.add_argument("--max-pairs-per-kinase", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    if args.command == "smoke":
        args.epochs = 1
        args.patience = 1
        args.num_points = min(args.num_points, 96)
        args.max_pairs_per_kinase = min(args.max_pairs_per_kinase, 4)
    set_seed(args.seed)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
