"""Deterministic ESM2 embedding extraction with a global sequence cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"


def normalize_sequence(sequence: str) -> str:
    """Return a canonical amino-acid sequence accepted by ESM."""
    sequence = "".join(str(sequence).split()).upper()
    if not sequence:
        raise ValueError("Protein sequence is empty")
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZUO")
    invalid = sorted(set(sequence) - allowed)
    if invalid:
        raise ValueError(f"Sequence contains unsupported residues: {invalid}")
    return sequence


def sequence_hash(sequence: str) -> str:
    """Compute the stable cache identity for a canonical sequence."""
    return hashlib.sha256(normalize_sequence(sequence).encode("ascii")).hexdigest()


class EmbeddingCache:
    """One-file-per-sequence storage for residue and pooled embeddings."""

    def __init__(self, cache_dir: Path | str = Path("cache/esm_embeddings")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, sequence_or_hash: str) -> Path:
        value = str(sequence_or_hash)
        digest = value if len(value) == 64 and all(c in "0123456789abcdef" for c in value) else sequence_hash(value)
        return self.cache_dir / f"{digest}.pt"

    def exists(self, sequence_or_hash: str) -> bool:
        return self.path_for(sequence_or_hash).is_file()

    def load(self, sequence_or_hash: str) -> dict[str, Any]:
        path = self.path_for(sequence_or_hash)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid embedding cache payload: {path}")
        required = {
            "model_name",
            "sequence_hash",
            "sequence_length",
            "embedding_dim",
            "residue_embeddings",
            "pooled_embedding",
        }
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Embedding cache file {path} is missing keys: {sorted(missing)}")
        expected_hash = path.stem
        if payload["sequence_hash"] != expected_hash:
            raise ValueError(f"Embedding hash mismatch in {path}")
        residue = payload["residue_embeddings"]
        pooled = payload["pooled_embedding"]
        if not isinstance(residue, torch.Tensor) or residue.ndim != 2:
            raise ValueError(f"Invalid residue embedding tensor in {path}")
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 1:
            raise ValueError(f"Invalid pooled embedding tensor in {path}")
        if residue.shape != (payload["sequence_length"], payload["embedding_dim"]):
            raise ValueError(f"Residue embedding shape mismatch in {path}")
        if pooled.shape[0] != payload["embedding_dim"]:
            raise ValueError(f"Pooled embedding shape mismatch in {path}")
        return payload

    def save(self, sequence: str, payload: dict[str, Any]) -> Path:
        digest = sequence_hash(sequence)
        path = self.path_for(digest)
        if path.exists():
            return path
        payload = dict(payload)
        payload["sequence_hash"] = digest
        payload["sequence_length"] = len(normalize_sequence(sequence))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{digest}.", suffix=".tmp", dir=self.cache_dir
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path


class ESMEmbedder:
    """Reusable HuggingFace ESM2 embedder with safe long-sequence handling."""

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        device: str | torch.device | None = None,
        batch_size: int = 2,
        chunk_overlap: int = 128,
        cache_dir: Path | str = Path("cache/esm_embeddings"),
        cache_dtype: torch.dtype = torch.float32,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.chunk_overlap = chunk_overlap
        self.cache = EmbeddingCache(cache_dir)
        self.cache_dtype = cache_dtype
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "HuggingFace Transformers is required. Install the project requirements."
            ) from exc
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    def _max_residues(self) -> int:
        max_positions = int(getattr(self.model.config, "max_position_embeddings", 1026))
        special_tokens = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        max_residues = max_positions - special_tokens
        if max_residues < 1:
            raise ValueError(f"Invalid model token limit: {max_positions}")
        return max_residues

    def _chunks(self, sequence: str) -> list[tuple[int, str]]:
        max_residues = self._max_residues()
        if len(sequence) <= max_residues:
            return [(0, sequence)]
        if self.chunk_overlap >= max_residues:
            raise ValueError("chunk_overlap must be smaller than the ESM residue limit")
        stride = max_residues - self.chunk_overlap
        starts = list(range(0, len(sequence), stride))
        chunks = [(start, sequence[start : start + max_residues]) for start in starts]
        return [(start, chunk) for start, chunk in chunks if chunk]

    def _embed_uncached(self, sequences: Sequence[str]) -> list[dict[str, Any]]:
        chunk_records: list[tuple[int, int, str]] = []
        for sequence_index, sequence in enumerate(sequences):
            for start, chunk in self._chunks(sequence):
                chunk_records.append((sequence_index, start, chunk))

        sums = [
            torch.zeros((len(sequence), self.embedding_dim), dtype=torch.float32)
            for sequence in sequences
        ]
        counts = [torch.zeros(len(sequence), dtype=torch.float32) for sequence in sequences]

        with torch.no_grad():
            for offset in range(0, len(chunk_records), self.batch_size):
                batch = chunk_records[offset : offset + self.batch_size]
                chunk_sequences = [record[2] for record in batch]
                tokens = self.tokenizer(
                    chunk_sequences,
                    add_special_tokens=True,
                    padding=True,
                    return_tensors="pt",
                )
                model_inputs = {key: value.to(self.device) for key, value in tokens.items()}
                hidden = self.model(**model_inputs).last_hidden_state.detach().cpu()
                for batch_index, (sequence_index, start, chunk) in enumerate(batch):
                    residue_hidden = hidden[batch_index, 1 : len(chunk) + 1].float()
                    end = start + len(chunk)
                    sums[sequence_index][start:end] += residue_hidden
                    counts[sequence_index][start:end] += 1

        results = []
        for sequence, embedding_sum, count in zip(sequences, sums, counts):
            if torch.any(count == 0):
                raise RuntimeError("Long-sequence chunking left uncovered residues")
            residue_embeddings = embedding_sum / count.unsqueeze(-1)
            pooled_embedding = residue_embeddings.mean(dim=0)
            results.append(
                {
                    "model_name": self.model_name,
                    "embedding_dim": residue_embeddings.shape[1],
                    "residue_embeddings": residue_embeddings.to(self.cache_dtype),
                    "pooled_embedding": pooled_embedding.to(self.cache_dtype),
                }
            )
        return results

    def embed_batch(
        self, sequences: Iterable[str], use_cache: bool = True
    ) -> list[dict[str, Any]]:
        """Embed sequences, loading and populating the global cache as needed."""
        canonical = [normalize_sequence(sequence) for sequence in sequences]
        results: list[dict[str, Any] | None] = [None] * len(canonical)
        missing_by_hash: dict[str, tuple[str, list[int]]] = {}

        for index, sequence in enumerate(canonical):
            digest = sequence_hash(sequence)
            if use_cache and self.cache.exists(digest):
                results[index] = self.cache.load(digest)
            else:
                if digest not in missing_by_hash:
                    missing_by_hash[digest] = (sequence, [])
                missing_by_hash[digest][1].append(index)

        missing_sequences = [item[0] for item in missing_by_hash.values()]
        if missing_sequences:
            embedded = self._embed_uncached(missing_sequences)
            for (digest, (sequence, indices)), payload in zip(missing_by_hash.items(), embedded):
                if use_cache:
                    path = self.cache.save(sequence, payload)
                    cached_payload = self.cache.load(digest)
                    cached_payload["embedding_file"] = str(path)
                    payload = cached_payload
                else:
                    payload["sequence_hash"] = digest
                    payload["sequence_length"] = len(sequence)
                for index in indices:
                    results[index] = payload

        return [result for result in results if result is not None]

    def embed(self, sequence: str, use_cache: bool = True) -> dict[str, Any]:
        """Embed one sequence and return residue-level and pooled representations."""
        return self.embed_batch([sequence], use_cache=use_cache)[0]
