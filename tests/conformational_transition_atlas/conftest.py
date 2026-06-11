from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import pytest


@pytest.fixture()
def atlas_data(tmp_path: Path):
    data_root = tmp_path / "data"
    metadata_dir = data_root / "metadata"
    splits_dir = data_root / "splits"
    processed_dir = data_root / "processed"
    metadata_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    rows = [
        {"structure_id": 1, "pdb_id": "AAAA", "kinase_name": "KIN1", "kinase_family": "Fam", "kinase_group": "G", "species": "Human", "conformational_state": "inactive", "dfg_state": "out-like", "alphac_state": "in", "ligand_present": 1, "resolution": 2.0, "filepath": "x"},
        {"structure_id": 2, "pdb_id": "AAAB", "kinase_name": "KIN1", "kinase_family": "Fam", "kinase_group": "G", "species": "Human", "conformational_state": "active", "dfg_state": "in", "alphac_state": "out", "ligand_present": 1, "resolution": 2.1, "filepath": "x"},
        {"structure_id": 3, "pdb_id": "BBBA", "kinase_name": "KIN2", "kinase_family": "Fam", "kinase_group": "G", "species": "Human", "conformational_state": "inactive", "dfg_state": "na", "alphac_state": "out", "ligand_present": 0, "resolution": 2.3, "filepath": "x"},
        {"structure_id": 4, "pdb_id": "BBBB", "kinase_name": "KIN2", "kinase_family": "Fam", "kinase_group": "G", "species": "Human", "conformational_state": "active", "dfg_state": "in", "alphac_state": "in", "ligand_present": 0, "resolution": 2.2, "filepath": "x"},
    ]
    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_dir / "kinase_labels.csv", index=False)

    train = metadata.iloc[[0, 1]].copy()
    val = metadata.iloc[[2]].copy()
    test = metadata.iloc[[3]].copy()
    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)

    def save_tensor(pdb_id: str, coords):
        td = processed_dir / pdb_id
        td.mkdir(parents=True)
        coords = torch.tensor(coords, dtype=torch.float32)
        dist = torch.cdist(coords, coords)
        torch.save(coords, td / "ca_coords.pt")
        torch.save(dist, td / "distance_matrix.pt")
        torch.save({"label": 1 if pdb_id in {"AAAB", "BBBB"} else 0, "kinase_name": "KIN1"}, td / "metadata.pt")

    save_tensor("AAAA", [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    save_tensor("AAAB", [[0, 0, 0], [1, 1, 0], [2, 0, 1]])
    save_tensor("BBBA", [[0, 0, 0], [0, 1, 0], [0, 2, 0]])
    save_tensor("BBBB", [[0, 0, 0], [0, 1, 1], [0, 2, 1]])

    return {
        "data_root": data_root,
        "metadata": metadata,
        "splits_dir": splits_dir,
        "processed_dir": processed_dir,
    }
