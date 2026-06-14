"""PyTorch dataset API for exported LOKO folds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from .esm import EmbeddingCache


class KinaseFoldDataset(Dataset):
    """Load fold metadata and globally cached ESM embeddings on demand."""

    VALID_SPLITS = {"train", "validation", "test"}

    def __init__(
        self,
        fold_id: int,
        split: str = "train",
        folds_dir: Path | str = Path("data/folds"),
        project_root: Path | str = Path("."),
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        if fold_id < 1:
            raise ValueError("fold_id must be at least 1")
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}")
        self.fold_id = int(fold_id)
        self.split = split
        self.project_root = Path(project_root)
        self.transform = transform
        self.manifest_path = (
            Path(folds_dir) / f"fold_{self.fold_id:02d}" / f"{split}.csv"
        )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Fold manifest not found: {self.manifest_path}")
        self.metadata = pd.read_csv(self.manifest_path)
        self.cache = EmbeddingCache(self.project_root / "cache/esm_embeddings")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        embedding_path = self.project_root / str(row["embedding_file"])
        try:
            payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(embedding_path, map_location="cpu")
        if payload.get("sequence_hash") != row["sequence_hash"]:
            raise ValueError(f"Embedding mismatch for row {index}: {embedding_path}")
        item = {
            "kinase": str(row["kinase"]),
            "pdb_id": str(row["pdb_id"]),
            "chain": str(row["chain"]),
            "conformational_state": str(row["conformational_state"]),
            "state_label": torch.tensor(
                1 if row["conformational_state"] == "active" else 0, dtype=torch.long
            ),
            "sequence_hash": str(row["sequence_hash"]),
            "embedding_file": str(row["embedding_file"]),
            "sequence_length": int(payload["sequence_length"]),
            "residue_embeddings": payload["residue_embeddings"].float(),
            "pooled_embedding": payload["pooled_embedding"].float(),
        }
        return self.transform(item) if self.transform else item
