from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.conformational_transition_atlas.dataset_loader import TransitionAtlasLoader
from experiments.conformational_transition_atlas.diversity_analysis import diversity_coverage_novelty
from experiments.conformational_transition_atlas.kinase_analysis import kinase_state_distribution, per_kinase_summary
from experiments.conformational_transition_atlas.run_transition_atlas import run_atlas
from experiments.conformational_transition_atlas.transition_paths import evaluate_transition_pairs, pair_active_inactive


def test_preserves_granular_states(atlas_data):
    loader = TransitionAtlasLoader(
        metadata_csv=atlas_data["data_root"] / "metadata" / "kinase_labels.csv",
        splits_dir=atlas_data["splits_dir"],
        processed_dir=atlas_data["processed_dir"],
    )
    atlas = loader.records_dataframe()
    assert "out-like" in set(atlas["dfg_state"])
    assert "na" in set(atlas["dfg_state"])


def test_distribution_and_summary(atlas_data):
    loader = TransitionAtlasLoader(
        metadata_csv=atlas_data["data_root"] / "metadata" / "kinase_labels.csv",
        splits_dir=atlas_data["splits_dir"],
        processed_dir=atlas_data["processed_dir"],
    )
    atlas = loader.records_dataframe()
    dist = kinase_state_distribution(atlas)
    summary = per_kinase_summary(atlas)
    diversity = diversity_coverage_novelty(atlas, list(loader.iter_records()))
    assert not dist.empty
    assert not summary.empty
    assert not diversity.empty


def test_transition_pairs_and_run_outputs(atlas_data, tmp_path):
    loader = TransitionAtlasLoader(
        metadata_csv=atlas_data["data_root"] / "metadata" / "kinase_labels.csv",
        splits_dir=atlas_data["splits_dir"],
        processed_dir=atlas_data["processed_dir"],
    )
    atlas = loader.records_dataframe()
    records = list(loader.iter_records())
    pairs = pair_active_inactive(atlas)
    transitions = evaluate_transition_pairs(atlas, records)
    assert len(pairs) >= 2
    assert not transitions.empty

    results_root = tmp_path / "results"
    result = run_atlas(atlas_data["data_root"], results_root)
    assert result["status"] == "completed"
    assert (results_root / "state_atlas.csv").exists()
    assert (results_root / "summary.md").exists()


def test_graphics_are_saved(atlas_data, tmp_path):
    results_root = tmp_path / "results"
    run_atlas(atlas_data["data_root"], results_root)
    figures = results_root / "figures"
    assert figures.exists()
    assert any(figures.glob("*.png"))
