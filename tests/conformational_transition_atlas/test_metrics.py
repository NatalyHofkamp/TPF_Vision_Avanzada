from __future__ import annotations

import numpy as np

from experiments.conformational_transition_atlas.manifold_analysis import cluster_landscape, compute_pca
from experiments.conformational_transition_atlas.order_parameters import compute_order_parameters
from experiments.conformational_transition_atlas.structural_metrics import aligned_rmsd, contact_map_similarity, drmsd, pairwise_diversity


def test_structural_metrics_run():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    b = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 1]], dtype=float)
    da = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    db = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=-1)
    assert aligned_rmsd(a, b) >= 0
    assert drmsd(da, db) >= 0
    assert 0 <= contact_map_similarity(a, b) <= 1
    assert pairwise_diversity([a, b]) >= 0
    geom = compute_order_parameters(a, da)
    assert "radius_of_gyration" in geom


def test_pca_and_clustering_run(atlas_data):
    import pandas as pd

    df = pd.read_csv(atlas_data["data_root"] / "metadata" / "kinase_labels.csv")
    df["num_residues"] = [3, 3, 3, 3]
    df["radius_of_gyration"] = [0.1, 0.2, 0.3, 0.4]
    df["mean_pair_distance"] = [1, 1, 1, 1]
    df["median_pair_distance"] = [1, 1, 1, 1]
    df["pair_distance_std"] = [0.1, 0.1, 0.1, 0.1]
    df["pair_distance_min"] = [0.1, 0.1, 0.1, 0.1]
    df["pair_distance_max"] = [2, 2, 2, 2]
    df["contact_density"] = [0.2, 0.3, 0.4, 0.5]
    df["local_contact_fraction"] = [0.2, 0.3, 0.4, 0.5]
    df["global_contact_fraction"] = [0.2, 0.3, 0.4, 0.5]
    df["compactness"] = [0.9, 0.8, 0.7, 0.6]
    pca = compute_pca(df)
    clustered = cluster_landscape(df)
    assert {"pca_1", "pca_2"}.issubset(pca.columns)
    assert "cluster" in clustered.columns
