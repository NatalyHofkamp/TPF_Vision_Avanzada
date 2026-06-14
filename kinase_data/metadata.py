"""Build ESM-ready metadata from the existing KLIFS manifest."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .esm import EmbeddingCache, normalize_sequence, sequence_hash

LOGGER = logging.getLogger(__name__)

AMINO_ACIDS = {
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
    "PYL": "O",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

MANIFEST_COLUMNS = [
    "kinase",
    "pdb_id",
    "chain",
    "conformational_state",
    "sequence",
    "sequence_length",
    "sequence_hash",
    "embedding_file",
    "embedding_dimension",
]


def _relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def extract_pdb_sequence(pdb_path: Path | str) -> tuple[str, str]:
    """Extract the single protein chain represented by a normalized KLIFS PDB."""
    pdb_path = Path(pdb_path)
    chains: dict[str, list[str]] = {}
    seen_residues: set[tuple[str, str, str]] = set()
    with pdb_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            chain = line[21].strip() or "_"
            residue_key = (chain, line[22:26].strip(), line[26].strip())
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            residue = AMINO_ACIDS.get(line[17:20].strip().upper(), "X")
            chains.setdefault(chain, []).append(residue)
    if not chains:
        raise ValueError(f"No protein C-alpha residues found in {pdb_path}")
    if len(chains) > 1:
        lengths = {chain: len(residues) for chain, residues in chains.items()}
        raise ValueError(f"Expected one KLIFS protein chain in {pdb_path}, found {lengths}")
    chain, residues = next(iter(chains.items()))
    return chain, normalize_sequence("".join(residues))


class ESMManifestBuilder:
    """Augment existing KLIFS metadata with sequences and cache locations."""

    def __init__(
        self,
        metadata_csv: Path | str = Path("data/metadata/kinase_labels.csv"),
        manifest_csv: Path | str = Path("data/esm_manifest.csv"),
        cache_dir: Path | str = Path("cache/esm_embeddings"),
        project_root: Path | str = Path("."),
    ):
        self.metadata_csv = Path(metadata_csv)
        self.manifest_csv = Path(manifest_csv)
        self.project_root = Path(project_root)
        self.cache = EmbeddingCache(cache_dir)
        if not self.metadata_csv.is_file():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_csv}")

    def build(
        self, embedder: Any | None = None, embedding_batch_size: int = 32
    ) -> pd.DataFrame:
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        metadata = pd.read_csv(self.metadata_csv)
        required = {"kinase_name", "pdb_id", "conformational_state", "filepath"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

        sequence_by_file: dict[str, tuple[str, str]] = {}
        records = []
        unique_sequences: dict[str, str] = {}
        for row in metadata.itertuples(index=False):
            filepath = str(row.filepath)
            if filepath not in sequence_by_file:
                sequence_by_file[filepath] = extract_pdb_sequence(filepath)
            chain, sequence = sequence_by_file[filepath]
            digest = sequence_hash(sequence)
            unique_sequences.setdefault(digest, sequence)
            embedding_path = self.cache.path_for(digest)
            records.append(
                {
                    "kinase": str(row.kinase_name),
                    "pdb_id": str(row.pdb_id).lower(),
                    "chain": chain,
                    "conformational_state": str(row.conformational_state).lower(),
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "sequence_hash": digest,
                    "embedding_file": _relative_or_absolute(
                        embedding_path, self.project_root
                    ),
                    "embedding_dimension": None,
                }
            )
        manifest = pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)
        dimensions: dict[str, int] = {}
        sequence_items = list(unique_sequences.items())
        if embedder is not None:
            for offset in range(0, len(sequence_items), embedding_batch_size):
                batch_items = sequence_items[offset : offset + embedding_batch_size]
                LOGGER.info(
                    "Embedding unique sequences %d-%d of %d",
                    offset + 1,
                    offset + len(batch_items),
                    len(sequence_items),
                )
                payloads = embedder.embed_batch(
                    [sequence for _, sequence in batch_items], use_cache=True
                )
                for (digest, _), payload in zip(batch_items, payloads):
                    dimensions[digest] = int(payload["embedding_dim"])
        else:
            for digest, _ in sequence_items:
                if self.cache.exists(digest):
                    dimensions[digest] = int(self.cache.load(digest)["embedding_dim"])
        manifest["embedding_dimension"] = manifest["sequence_hash"].map(dimensions)
        self.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(self.manifest_csv, index=False)
        return manifest

    def write_statistics(
        self,
        manifest: pd.DataFrame,
        output_path: Path | str = Path("reports/esm_statistics.json"),
    ) -> dict[str, Any]:
        dimensions = sorted(
            int(value) for value in manifest["embedding_dimension"].dropna().unique()
        )
        statistics = {
            "total_proteins": int(len(manifest)),
            "unique_sequences": int(manifest["sequence_hash"].nunique()),
            "average_length": float(manifest["sequence_length"].mean()),
            "min_length": int(manifest["sequence_length"].min()),
            "max_length": int(manifest["sequence_length"].max()),
            "embedding_dimension": dimensions[0] if len(dimensions) == 1 else dimensions,
            "number_of_cached_embeddings": int(
                sum(
                    self.cache.path_for(digest).is_file()
                    for digest in manifest["sequence_hash"].unique()
                )
            ),
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
        return statistics

    def validate_embeddings(
        self,
        manifest: pd.DataFrame,
        output_path: Path | str = Path("reports/embedding_validation.json"),
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        issues = []
        dimensions: set[int] = set()
        checked: set[str] = set()
        for row in manifest.itertuples(index=False):
            if not row.sequence:
                issues.append({"sequence_hash": row.sequence_hash, "error": "empty_sequence"})
                continue
            if row.sequence_hash in checked:
                continue
            checked.add(row.sequence_hash)
            path = self.project_root / row.embedding_file
            if not path.is_file():
                issues.append({"sequence_hash": row.sequence_hash, "error": "missing_file"})
                continue
            try:
                payload = self.cache.load(row.sequence_hash)
                residue = payload["residue_embeddings"]
                pooled = payload["pooled_embedding"]
                dimensions.add(int(payload["embedding_dim"]))
                if not residue.isfinite().all().item() or not pooled.isfinite().all().item():
                    issues.append({"sequence_hash": row.sequence_hash, "error": "non_finite"})
                if int(payload["sequence_length"]) != len(row.sequence):
                    issues.append(
                        {"sequence_hash": row.sequence_hash, "error": "length_mismatch"}
                    )
            except Exception as exc:
                issues.append(
                    {
                        "sequence_hash": row.sequence_hash,
                        "error": "corrupted_file",
                        "detail": str(exc),
                    }
                )
        if len(dimensions) > 1:
            issues.append(
                {"error": "inconsistent_embedding_dimensions", "dimensions": sorted(dimensions)}
            )
        report = {
            "valid": not issues,
            "total_manifest_rows": int(len(manifest)),
            "unique_sequences": int(manifest["sequence_hash"].nunique()),
            "checked_embedding_files": len(checked),
            "embedding_dimensions": sorted(dimensions),
            "issues": issues,
        }
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if issues and raise_on_error:
            raise ValueError(f"Embedding validation failed with {len(issues)} issue(s)")
        return report
