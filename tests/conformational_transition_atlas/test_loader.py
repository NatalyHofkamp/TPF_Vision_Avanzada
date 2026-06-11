from __future__ import annotations

from experiments.conformational_transition_atlas.dataset_loader import TransitionAtlasLoader


def test_loader_loads_metadata_and_tensors(atlas_data):
    loader = TransitionAtlasLoader(
        metadata_csv=atlas_data["data_root"] / "metadata" / "kinase_labels.csv",
        splits_dir=atlas_data["splits_dir"],
        processed_dir=atlas_data["processed_dir"],
    )
    records = list(loader.iter_records())
    assert len(records) == 4
    assert records[0].coords.shape[1] == 3
    assert records[0].distance_matrix.shape[0] == records[0].coords.shape[0]


def test_loader_does_not_modify_source_files(atlas_data):
    metadata_path = atlas_data["data_root"] / "metadata" / "kinase_labels.csv"
    before = metadata_path.read_bytes()
    loader = TransitionAtlasLoader(
        metadata_csv=metadata_path,
        splits_dir=atlas_data["splits_dir"],
        processed_dir=atlas_data["processed_dir"],
    )
    _ = list(loader.iter_records())
    after = metadata_path.read_bytes()
    assert before == after
