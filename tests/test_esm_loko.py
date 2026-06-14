from pathlib import Path

import pandas as pd
import pytest
import torch

from kinase_data.dataset import KinaseFoldDataset
from kinase_data.esm import EmbeddingCache, sequence_hash
from kinase_data.loko import LeaveOneKinaseOutSplitter
from kinase_data.metadata import ESMManifestBuilder, extract_pdb_sequence


class FakeEmbedder:
    def __init__(self, cache_dir: Path, dimension: int = 8):
        self.cache = EmbeddingCache(cache_dir)
        self.dimension = dimension
        self.calls = 0

    def embed(self, sequence: str, use_cache: bool = True):
        digest = sequence_hash(sequence)
        if self.cache.exists(digest):
            return self.cache.load(digest)
        self.calls += 1
        residue = torch.arange(len(sequence) * self.dimension, dtype=torch.float32).reshape(
            len(sequence), self.dimension
        )
        self.cache.save(
            sequence,
            {
                "model_name": "fake-esm",
                "embedding_dim": self.dimension,
                "residue_embeddings": residue,
                "pooled_embedding": residue.mean(dim=0),
            },
        )
        return self.cache.load(digest)

    def embed_batch(self, sequences, use_cache: bool = True):
        return [self.embed(sequence, use_cache=use_cache) for sequence in sequences]


def write_pdb(path: Path, chain: str, sequence: list[str]) -> None:
    lines = []
    for index, residue in enumerate(sequence, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  {residue:>3s} {chain}{index:4d}    "
            f"{index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C  \n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def test_manifest_cache_folds_and_dataset(tmp_path: Path):
    pdb_a = tmp_path / "a.pdb"
    pdb_b = tmp_path / "b.pdb"
    pdb_c = tmp_path / "c.pdb"
    write_pdb(pdb_a, "A", ["ALA", "CYS", "ASP"])
    write_pdb(pdb_b, "B", ["GLU", "PHE", "GLY"])
    write_pdb(pdb_c, "C", ["HIS", "ILE", "LYS"])
    metadata = pd.DataFrame(
        {
            "kinase_name": ["A", "A", "B", "C"],
            "pdb_id": ["1aaa", "1aab", "2bbb", "3ccc"],
            "conformational_state": ["active", "inactive", "active", "inactive"],
            "filepath": [pdb_a, pdb_a, pdb_b, pdb_c],
        }
    )
    metadata_path = tmp_path / "metadata.csv"
    manifest_path = tmp_path / "data" / "esm_manifest.csv"
    cache_dir = tmp_path / "cache" / "esm_embeddings"
    metadata.to_csv(metadata_path, index=False)
    embedder = FakeEmbedder(cache_dir)
    builder = ESMManifestBuilder(
        metadata_path, manifest_path, cache_dir=cache_dir, project_root=tmp_path
    )
    manifest = builder.build(embedder)
    assert embedder.calls == 3
    assert manifest["sequence_hash"].nunique() == 3
    assert set(manifest["chain"]) == {"A", "B", "C"}
    assert set(manifest["embedding_dimension"]) == {8}
    assert builder.validate_embeddings(
        manifest, tmp_path / "validation.json"
    )["valid"]

    folds_dir = tmp_path / "data" / "folds"
    splitter = LeaveOneKinaseOutSplitter(
        manifest, output_dir=folds_dir, kinase_order=["A", "B", "C"]
    )
    statistics = splitter.generate(
        tmp_path / "loko_statistics.json", tmp_path / "loko_validation.json"
    )
    assert statistics["number_of_folds"] == 3
    dataset = KinaseFoldDataset(
        1, "test", folds_dir=folds_dir, project_root=tmp_path
    )
    assert len(dataset) == 2
    assert dataset[0]["residue_embeddings"].shape == (3, 8)
    assert dataset[0]["kinase"] == "A"


def test_splitter_detects_leakage():
    train = pd.DataFrame({"kinase": ["A", "B"]})
    validation = pd.DataFrame({"kinase": ["B"]})
    test = pd.DataFrame({"kinase": ["C"]})
    with pytest.raises(ValueError, match="leakage"):
        LeaveOneKinaseOutSplitter.validate_fold(train, validation, test)


def test_sequence_extraction_rejects_multiple_chains(tmp_path: Path):
    pdb_path = tmp_path / "multi.pdb"
    write_pdb(pdb_path, "A", ["ALA"])
    with pdb_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "ATOM      2  CA  GLY B   1       2.000   0.000   0.000  1.00  0.00           C  \n"
        )
    with pytest.raises(ValueError, match="Expected one"):
        extract_pdb_sequence(pdb_path)
