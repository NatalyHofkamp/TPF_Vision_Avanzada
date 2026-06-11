from __future__ import annotations

from experiments.conformational_transition_atlas.run_transition_atlas import run_atlas


def test_outputs_generated(atlas_data, tmp_path):
    result = run_atlas(atlas_data["data_root"], tmp_path / "results")
    assert result["status"] == "completed"
    expected = [
        "state_atlas.csv",
        "kinase_state_distribution.csv",
        "per_kinase_summary.csv",
        "transition_paths.csv",
        "diversity_coverage_novelty.csv",
        "summary.md",
    ]
    for name in expected:
        assert (tmp_path / "results" / name).exists()
