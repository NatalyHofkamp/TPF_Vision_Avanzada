"""Quality-control plots and kinase-level ESM similarity analysis."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def _configure_plotting() -> None:
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/kinase-matplotlib"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kinase-xdg-cache")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/kinase-numba-cache")


def _load_umap_class():
    # umap-learn imports optional TensorFlow and coverage integrations at package
    # import time. They are not needed for standard non-parametric UMAP and can
    # pull incompatible binary dependencies into otherwise valid environments.
    blocked_modules = []
    for module_name in ("tensorflow", "coverage"):
        if module_name not in sys.modules:
            sys.modules[module_name] = None
            blocked_modules.append(module_name)
    try:
        from umap import UMAP
    except (ImportError, AttributeError) as exc:
        raise ImportError("UMAP plots require a working 'umap-learn' package") from exc
    finally:
        for module_name in blocked_modules:
            sys.modules.pop(module_name, None)
    return UMAP


def load_pooled_embeddings(
    manifest: pd.DataFrame, project_root: Path | str = Path(".")
) -> np.ndarray:
    project_root = Path(project_root)
    by_hash: dict[str, np.ndarray] = {}
    embeddings = []
    for row in manifest.itertuples(index=False):
        if row.sequence_hash not in by_hash:
            path = project_root / row.embedding_file
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            by_hash[row.sequence_hash] = payload["pooled_embedding"].float().numpy()
        embeddings.append(by_hash[row.sequence_hash])
    return np.stack(embeddings)


def generate_esm_projections(
    manifest: pd.DataFrame,
    output_dir: Path | str = Path("figures/esm"),
    project_root: Path | str = Path("."),
    random_state: int = 42,
) -> None:
    _configure_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = load_pooled_embeddings(manifest, project_root)
    unique_mask = ~manifest["sequence_hash"].duplicated()
    unique_manifest = manifest.loc[unique_mask].reset_index(drop=True)
    unique_embeddings = embeddings[unique_mask.to_numpy()]
    pca_coordinates = PCA(n_components=2, random_state=random_state).fit_transform(
        unique_embeddings
    )
    projection_frame = unique_manifest[
        ["kinase", "pdb_id", "conformational_state", "sequence_hash"]
    ].copy()
    projection_frame["x"] = pca_coordinates[:, 0]
    projection_frame["y"] = pca_coordinates[:, 1]
    _scatter_projection(
        projection_frame, "kinase", "ESM2 PCA by kinase", output_dir / "pca_by_kinase.png"
    )
    _scatter_projection(
        projection_frame,
        "conformational_state",
        "ESM2 PCA by conformational state",
        output_dir / "pca_by_state.png",
    )

    UMAP = _load_umap_class()
    neighbors = min(15, max(2, len(unique_manifest) - 1))
    umap_coordinates = UMAP(
        n_components=2,
        n_neighbors=neighbors,
        metric="cosine",
        random_state=random_state,
        transform_seed=random_state,
    ).fit_transform(unique_embeddings)
    projection_frame["x"] = umap_coordinates[:, 0]
    projection_frame["y"] = umap_coordinates[:, 1]
    _scatter_projection(
        projection_frame,
        "kinase",
        "ESM2 UMAP by kinase",
        output_dir / "umap_by_kinase.png",
    )
    _scatter_projection(
        projection_frame,
        "conformational_state",
        "ESM2 UMAP by conformational state",
        output_dir / "umap_by_state.png",
    )
    plt.close("all")


def _scatter_projection(
    frame: pd.DataFrame, hue: str, title: str, output_path: Path
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    figure, axis = plt.subplots(figsize=(11, 8))
    sns.scatterplot(data=frame, x="x", y="y", hue=hue, alpha=0.75, s=40, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_loko_figures(
    manifest: pd.DataFrame,
    loko_statistics: dict[str, Any],
    output_dir: Path | str = Path("figures/loko"),
) -> None:
    _configure_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kinase_counts = manifest["kinase"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(11, 6))
    kinase_counts.plot.bar(ax=axis, color="#4472c4")
    axis.set_title("Structures per kinase")
    axis.set_ylabel("Structures")
    figure.tight_layout()
    figure.savefig(output_dir / "structures_per_kinase.png", dpi=200)
    plt.close(figure)

    state_counts = (
        manifest.groupby(["kinase", "conformational_state"]).size().unstack(fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    state_counts.plot.bar(stacked=True, ax=axis, color=["#2ca02c", "#d62728"])
    axis.set_title("Active vs inactive structures")
    axis.set_ylabel("Structures")
    figure.tight_layout()
    figure.savefig(output_dir / "active_inactive_distribution.png", dpi=200)
    plt.close(figure)

    composition_rows = []
    assignment = np.zeros((len(loko_statistics["folds"]), len(kinase_counts)), dtype=int)
    kinase_index = {kinase: index for index, kinase in enumerate(kinase_counts.index)}
    for fold in loko_statistics["folds"]:
        for split_index, split in enumerate(("train", "validation", "test")):
            composition_rows.append(
                {
                    "fold": f"fold_{fold['fold_id']:02d}",
                    "split": split,
                    "structures": fold[split]["structures"],
                }
            )
        for kinase in fold["kinase_assignments"]["train"]:
            assignment[fold["fold_id"] - 1, kinase_index[kinase]] = 0
        assignment[
            fold["fold_id"] - 1,
            kinase_index[fold["kinase_assignments"]["validation"]],
        ] = 1
        assignment[
            fold["fold_id"] - 1, kinase_index[fold["kinase_assignments"]["test"]]
        ] = 2

    composition = pd.DataFrame(composition_rows)
    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=composition, x="fold", y="structures", hue="split", ax=axis)
    axis.set_title("LOKO fold composition")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / "fold_composition.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        assignment,
        cmap=sns.color_palette(["#d9d9d9", "#ffbf00", "#c00000"], as_cmap=True),
        cbar=False,
        linewidths=0.5,
        xticklabels=kinase_counts.index,
        yticklabels=[f"fold_{i:02d}" for i in range(1, len(assignment) + 1)],
        ax=axis,
    )
    axis.set_title("Kinase assignment heatmap (train=gray, validation=amber, test=red)")
    figure.tight_layout()
    figure.savefig(output_dir / "kinase_assignment_heatmap.png", dpi=200)
    plt.close(figure)


def analyze_kinase_similarity(
    manifest: pd.DataFrame,
    output_dir: Path | str = Path("reports/kinase_similarity"),
    project_root: Path | str = Path("."),
) -> dict[str, Any]:
    _configure_plotting()
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = load_pooled_embeddings(manifest, project_root)
    embedding_frame = pd.DataFrame(embeddings)
    embedding_frame["kinase"] = manifest["kinase"].to_numpy()
    kinase_embeddings = embedding_frame.groupby("kinase").mean().sort_index()
    kinases = kinase_embeddings.index.tolist()
    matrix = cosine_similarity(kinase_embeddings.to_numpy())
    matrix_frame = pd.DataFrame(matrix, index=kinases, columns=kinases)
    matrix_frame.to_csv(output_dir / "cosine_similarity_matrix.csv")

    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix_frame, cmap="viridis", annot=True, fmt=".3f", ax=axis)
    axis.set_title("Mean kinase ESM2 cosine similarity")
    figure.tight_layout()
    figure.savefig(output_dir / "cosine_similarity_heatmap.png", dpi=200)
    plt.close(figure)

    nearest_neighbors = {}
    rows = []
    for index, kinase in enumerate(kinases):
        order = np.argsort(matrix[index])[::-1]
        neighbors = [
            {"kinase": kinases[j], "cosine_similarity": float(matrix[index, j])}
            for j in order
            if j != index
        ]
        nearest_neighbors[kinase] = neighbors
        for rank, neighbor in enumerate(neighbors, start=1):
            rows.append(
                {
                    "kinase": kinase,
                    "rank": rank,
                    "neighbor": neighbor["kinase"],
                    "cosine_similarity": neighbor["cosine_similarity"],
                }
            )
    with (output_dir / "nearest_neighbors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["kinase", "rank", "neighbor", "cosine_similarity"]
        )
        writer.writeheader()
        writer.writerows(rows)

    expected_pairs = [("EGFR", "ERBB2"), ("FGFR1", "PDGFRA"), ("CDK4", "CDK6")]
    pair_evaluation = []
    for first, second in expected_pairs:
        if first not in kinases or second not in kinases:
            pair_evaluation.append({"pair": [first, second], "available": False})
            continue
        first_index = kinases.index(first)
        second_index = kinases.index(second)
        first_rank = next(
            row["rank"]
            for row in rows
            if row["kinase"] == first and row["neighbor"] == second
        )
        second_rank = next(
            row["rank"]
            for row in rows
            if row["kinase"] == second and row["neighbor"] == first
        )
        pair_evaluation.append(
            {
                "pair": [first, second],
                "available": True,
                "cosine_similarity": float(matrix[first_index, second_index]),
                f"{first}_neighbor_rank": first_rank,
                f"{second}_neighbor_rank": second_rank,
                "mutual_top_3": first_rank <= 3 and second_rank <= 3,
            }
        )
    report = {
        "aggregation": "mean pooled ESM embedding over manifest structures per kinase",
        "nearest_neighbors": nearest_neighbors,
        "expected_pair_evaluation": pair_evaluation,
    }
    (output_dir / "nearest_neighbor_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
