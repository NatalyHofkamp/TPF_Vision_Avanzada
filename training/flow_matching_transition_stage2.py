#!/usr/bin/env python3
"""Stage-2 Flow Matching + CFG experiment with path export and visual analysis.

This script builds on the latent-transition diagnostic experiment, but focuses on
one stronger training stage for the activation direction and on explicit
visualization of the learned conformational path.

It keeps intermediates latent:
- t=0.00: observed source endpoint
- t=0.25/0.50/0.75: generated hypotheses
- t=1.00: generated endpoint prediction evaluated against the observed target

Outputs are written under ``results/flow_matching_cfg_transition_stage2`` by
default, with exported trajectories under
``outputs/transition_paths/[kinase]/[pair_id]/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/kinase_stage2_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kinase_stage2_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import flow_matching_transition_diagnostics as diag


LOG = logging.getLogger("flow_matching_stage2")
GUIDANCE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
LR_GRID = (1e-4, 7.5e-5, 5e-5)
PATH_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
STATE_LABELS = {"active": "Activa", "inactive": "Inactiva"}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def pyplot():
    plt, sns = diag.pyplot()
    sns.set_theme(style="whitegrid")
    return plt, sns


def write_pdb(coords: np.ndarray, path: Path, chain_id: str = "A") -> None:
    lines: List[str] = []
    for index, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  ALA {chain_id:1s}{index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def local_rmsd(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    squared = (prediction - target).square().sum(-1)
    value = (squared * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return torch.where(mask.sum(-1) > 0, value.sqrt() * scale, torch.full_like(value, float("nan")))


def local_distance_error(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pred_dist = torch.cdist(prediction, prediction)
    target_dist = torch.cdist(target, target)
    pair_mask = mask[:, :, None] & mask[:, None, :]
    error = ((pred_dist - target_dist).abs() * pair_mask).sum((1, 2))
    error = error / pair_mask.sum((1, 2)).clamp_min(1)
    return error * scale


def marked_distance(
    coords: torch.Tensor,
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    values = []
    for index in range(len(coords)):
        first = torch.where(first_mask[index])[0]
        second = torch.where(second_mask[index])[0]
        if len(first) == 0 or len(second) == 0:
            values.append(float("nan"))
        else:
            distance = torch.linalg.vector_norm(coords[index, first[0]] - coords[index, second[0]])
            values.append(float(distance * scale[index]))
    return torch.tensor(values)


def trajectory_feature(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    return centered.reshape(-1)


def path_projection(trajectories: Sequence[Dict[str, Any]], output: Path, name: str) -> None:
    records: List[Dict[str, Any]] = []
    features: List[np.ndarray] = []
    for trajectory_index, item in enumerate(trajectories):
        features.extend([trajectory_feature(item["x0"]), trajectory_feature(item["x1"])])
        records.extend(
            [
                {
                    "kind": f"real_{item['source_state']}",
                    "time": 0.0,
                    "trajectory": trajectory_index,
                },
                {
                    "kind": f"real_{item['target_state']}",
                    "time": 1.0,
                    "trajectory": trajectory_index,
                },
            ]
        )
        for time in sorted(item["path"]):
            features.append(trajectory_feature(item["path"][time]))
            records.append({"kind": "generated", "time": float(time), "trajectory": trajectory_index})

    matrix = np.stack(features)
    metadata = pd.DataFrame(records)
    projections: Dict[str, np.ndarray] = {"pca": PCA(n_components=2).fit_transform(matrix)}
    metadata["pca1"], metadata["pca2"] = projections["pca"][:, 0], projections["pca"][:, 1]
    metadata.to_csv(output / "metrics" / f"{name}_projection_points.csv", index=False)

    plt, sns = pyplot()
    fig, ax = plt.subplots(figsize=(9, 7))
    real = metadata[metadata["kind"] != "generated"]
    sns.scatterplot(data=real, x="pca1", y="pca2", hue="kind", style="kind", s=90, ax=ax)
    generated = metadata[metadata["kind"] == "generated"]
    for _, group in generated.groupby("trajectory"):
        group = group.sort_values("time")
        ax.plot(group["pca1"], group["pca2"], "-o", alpha=0.7, color="tab:purple")
        for left, right in zip(group.iloc[:-1].itertuples(), group.iloc[1:].itertuples()):
            ax.annotate("", xy=(right.pca1, right.pca2), xytext=(left.pca1, left.pca2), arrowprops={"arrowstyle": "->", "color": "tab:purple", "alpha": 0.45})
    ax.set_title("Campo conformacional proyectado (PCA)")
    fig.tight_layout()
    fig.savefig(output / "plots" / f"{name}_conformational_field_pca.png", dpi=200)
    plt.close(fig)

    if len(matrix) >= 10:
        perplexity = min(30, max(2, len(matrix) // 4), len(matrix) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(matrix)
        metadata["tsne1"], metadata["tsne2"] = tsne[:, 0], tsne[:, 1]
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(data=metadata, x="tsne1", y="tsne2", hue="kind", style="kind", size="time", ax=ax)
        ax.set_title("Campo conformacional proyectado (t-SNE)")
        fig.tight_layout()
        fig.savefig(output / "plots" / f"{name}_conformational_field_tsne.png", dpi=200)
        plt.close(fig)

    try:
        import umap

        embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(matrix)
        metadata["umap1"], metadata["umap2"] = embedding[:, 0], embedding[:, 1]
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.scatterplot(data=metadata, x="umap1", y="umap2", hue="kind", style="kind", size="time", ax=ax)
        ax.set_title("Campo conformacional proyectado (UMAP)")
        fig.tight_layout()
        fig.savefig(output / "plots" / f"{name}_conformational_field_umap.png", dpi=200)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        LOG.warning("UMAP unavailable: %s", exc)

    metadata.to_csv(output / "metrics" / f"{name}_projection_points.csv", index=False)


def plot_trajectory_bundle(
    item: Dict[str, Any],
    per_point: pd.DataFrame,
    output_dir: Path,
    name: str,
) -> None:
    plt, sns = pyplot()
    times = sorted(item["path"])

    fig = plt.figure(figsize=(4 * len(times), 4))
    for index, time in enumerate(times, start=1):
        axis = fig.add_subplot(1, len(times), index, projection="3d")
        coords = item["path"][time] * item["scale"]
        axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=1.2, color="tab:purple")
        axis.set_title(f"t={time:.2f}")
        axis.set_axis_off()
    fig.suptitle(f"{item['kinase']}: {item['source_state']} -> {item['target_state']}")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / f"{name}_trajectory_3d.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.lineplot(data=per_point, x="t", y="distance_to_target", ax=axes[0], marker="o")
    axes[0].set_title("Distancia al target")
    sns.lineplot(data=per_point, x="t", y="distance_to_source", ax=axes[1], marker="o")
    axes[1].set_title("Distancia al source")
    sns.lineplot(data=per_point, x="t", y="distance_matrix_consistency_error", ax=axes[2], marker="o")
    axes[2].set_title("Error de geometría interna")
    for axis in axes:
        axis.set_xlabel("t")
        axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / f"{name}_trajectory_metrics.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=per_point, x="t", y="distance_to_active", ax=axes[0], marker="o")
    axes[0].set_title("Distancia al estado activo")
    sns.lineplot(data=per_point, x="t", y="distance_to_inactive", ax=axes[1], marker="o")
    axes[1].set_title("Distancia al estado inactivo")
    for axis in axes:
        axis.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / f"{name}_active_inactive_distance.png", dpi=200)
    plt.close(fig)


def export_pdb_bundle(
    item: Dict[str, Any],
    output_root: Path,
    pair_index: int,
) -> Path:
    kinase_dir = output_root / "outputs" / "transition_paths" / slugify(item["kinase"])
    pair_dir = kinase_dir / f"pair_{pair_index:03d}_{slugify(item['inactive_pdb'])}_to_{slugify(item['active_pdb'])}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    scale = float(item["scale"])
    paths = item["path"]
    write_pdb(item["x0"] * scale, pair_dir / "inactive_start.pdb")
    write_pdb(item["x1"] * scale, pair_dir / "active_target.pdb")
    write_pdb(paths[0.25] * scale, pair_dir / "generated_t025.pdb")
    write_pdb(paths[0.50] * scale, pair_dir / "generated_t050.pdb")
    write_pdb(paths[0.75] * scale, pair_dir / "generated_t075.pdb")
    write_pdb(paths[1.00] * scale, pair_dir / "generated_t100.pdb")
    return pair_dir


def pairwise_rmsd(coords_list: Sequence[np.ndarray]) -> float:
    if len(coords_list) < 2:
        return 0.0
    values = []
    for i in range(len(coords_list)):
        for j in range(i + 1, len(coords_list)):
            a = coords_list[i]
            b = coords_list[j]
            n = min(len(a), len(b))
            values.append(diag.rmsd_aligned(a[:n], b[:n]))
    return float(np.nanmean(values)) if values else 0.0


def evaluate_with_guidance(
    model: diag.DiagnosticFlowField,
    dataset: diag.AlignedDataset,
    config: diag.DiagnosticConfig,
    guidance_scale: float,
    max_samples: Optional[int],
    output: Path,
    name: str,
    export_paths: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    eval_config = diag.DiagnosticConfig(**{**asdict(config), "guidance_scale": guidance_scale})
    device = diag.device_name()
    data = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    rows: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []
    seen = 0
    trajectory_limit = max_samples if max_samples is not None else 20
    model.eval()
    for raw in data:
        batch = diag.move_batch(raw, device)
        paths = diag.integrate(model, batch, eval_config)
        generated = paths[1.0]
        x0 = raw["x0"]
        x1 = raw["x1"]
        mask = raw["mask"]
        scale = raw["scale"]
        target_is_active = torch.tensor(
            [str(value) == "active" for value in raw["target_state_name"]],
            device=x0.device,
            dtype=torch.bool,
        )
        active_ref = torch.where(target_is_active[:, None, None], x1, x0)
        inactive_ref = torch.where(target_is_active[:, None, None], x0, x1)
        generated_to_target = diag.local_rmsd(generated, x1, mask, scale)
        initial_to_target = diag.local_rmsd(x0, x1, mask, scale)
        generated_to_source = diag.local_rmsd(generated, x0, mask, scale)
        generated_to_active = diag.local_rmsd(generated, active_ref, mask, scale)
        generated_to_inactive = diag.local_rmsd(generated, inactive_ref, mask, scale)
        dfg_rmsd = diag.local_rmsd(generated, x1, raw["dfg_mask"], scale)
        alphac_rmsd = diag.local_rmsd(generated, x1, raw["alphac_mask"], scale)
        hrd_rmsd = diag.local_rmsd(generated, x1, raw["hrd_mask"], scale)
        activation_loop_rmsd = diag.local_rmsd(generated, x1, raw["activation_loop_mask"], scale)
        local_dist = local_distance_error(generated, x1, mask, scale)
        generated_lys_glu = marked_distance(generated, raw["lys_mask"], raw["glu_mask"], scale)
        target_lys_glu = marked_distance(x1, raw["lys_mask"], raw["glu_mask"], scale)
        generated_dfg_distance = marked_distance(generated, raw["dfg_asp_mask"], raw["dfg_phe_mask"], scale)
        target_dfg_distance = marked_distance(x1, raw["dfg_asp_mask"], raw["dfg_phe_mask"], scale)
        path_stack = torch.stack([paths[t] for t in sorted(paths)], dim=1)
        increments = path_stack[:, 1:] - path_stack[:, :-1]
        path_length = increments.square().sum(dim=-1).mean(dim=-1).sqrt().sum(dim=-1) * scale
        diversity = path_stack.std(dim=1).square().sum(dim=-1).mean(dim=-1).sqrt() * scale
        for index in range(generated.shape[0]):
            source_state = raw["source_state_name"][index]
            target_state = raw["target_state_name"][index]
            direction = "activation" if source_state == "inactive" else "deactivation"
            source_pdb = raw["source_pdb"][index]
            target_pdb = raw["target_pdb"][index]
            inactive_pdb = source_pdb if source_state == "inactive" else target_pdb
            active_pdb = target_pdb if target_state == "active" else source_pdb
            baseline = float(initial_to_target[index])
            final = float(generated_to_target[index])
            source_curve = []
            target_curve = []
            geometry_curve = []
            time_points = sorted(paths)
            for time in time_points:
                state = paths[time][index : index + 1]
                state_mask = mask[index : index + 1]
                state_scale = scale[index : index + 1]
                source_value = float(diag.local_rmsd(state, x0[index : index + 1], state_mask, state_scale)[0])
                target_value = float(diag.local_rmsd(state, x1[index : index + 1], state_mask, state_scale)[0])
                active_value = float(diag.local_rmsd(state, active_ref[index : index + 1], state_mask, state_scale)[0])
                inactive_value = float(diag.local_rmsd(state, inactive_ref[index : index + 1], state_mask, state_scale)[0])
                expected = (1 - time) * torch.cdist(x0[index], x0[index]) + time * torch.cdist(x1[index], x1[index])
                state_distance = torch.cdist(state[0], state[0])
                pair_mask = state_mask[0, :, None] & state_mask[0, None, :]
                geometry_value = float(((state_distance - expected).abs() * pair_mask).sum() / pair_mask.sum().clamp_min(1) * state_scale[0])
                source_curve.append(source_value)
                target_curve.append(target_value)
                geometry_curve.append(geometry_value)
                if time == 0.0:
                    supervision_status = "observed source endpoint"
                    observed = True
                elif time == 1.0:
                    supervision_status = "generated endpoint prediction supervised against observed target"
                    observed = False
                else:
                    supervision_status = "latent generated hypothesis"
                    observed = False
                point_rows.append(
                    {
                        "model": name,
                        "kinase": raw["kinase_name"][index],
                        "source_pdb": raw["source_pdb"][index],
                        "target_pdb": raw["target_pdb"][index],
                        "t": time,
                        "distance_to_source": source_value,
                        "distance_to_target": target_value,
                        "distance_to_active": active_value,
                        "distance_to_inactive": inactive_value,
                        "distance_matrix_consistency_error": geometry_value,
                        "is_observed_structure": observed,
                        "supervision_status": supervision_status,
                    }
                )
            monotonic = [
                target_curve[pos] <= target_curve[pos - 1] + 1e-6
                and source_curve[pos] >= source_curve[pos - 1] - 1e-6
                for pos in range(1, len(time_points))
            ]
            rows.append(
                {
                    "model": name,
                    "guidance_scale": guidance_scale,
                    "direction": direction,
                    "source_state": source_state,
                    "target_state": target_state,
                    "kinase": raw["kinase_name"][index],
                    "inactive_pdb": inactive_pdb,
                    "active_pdb": active_pdb,
                    "global_ca_rmsd": final,
                    "initial_target_rmsd": baseline,
                    "generated_to_source_rmsd": float(generated_to_source[index]),
                    "generated_to_active_rmsd": float(generated_to_active[index]),
                    "generated_to_inactive_rmsd": float(generated_to_inactive[index]),
                    "relative_improvement": (baseline - final) / max(baseline, 1e-8),
                    "closer_to_target_than_initial": final < baseline,
                    "closer_to_target_than_source": final < float(generated_to_source[index]),
                    "closer_to_active": float(generated_to_active[index]) < float(generated_to_inactive[index]),
                    "closer_to_inactive": float(generated_to_inactive[index]) < float(generated_to_active[index]),
                    "distance_matrix_mae": float(local_dist[index]),
                    "dfg_rmsd": float(dfg_rmsd[index]),
                    "alphac_rmsd": float(alphac_rmsd[index]),
                    "hrd_rmsd": float(hrd_rmsd[index]),
                    "activation_loop_rmsd": float(activation_loop_rmsd[index]),
                    "lys_glu_distance_error": float(abs(generated_lys_glu[index] - target_lys_glu[index])),
                    "dfg_distance_error": float(abs(generated_dfg_distance[index] - target_dfg_distance[index])),
                    "trajectory_smoothness": float(
                        (path_stack[:, 1:] - path_stack[:, :-1]).square().sum(dim=-1).mean(dim=(1, 2)).sqrt()[index]
                    )
                    if path_stack.shape[1] > 1
                    else 0.0,
                    "directionality_fraction": float(np.mean(monotonic)),
                    "mean_path_geometry_error": float(np.mean(geometry_curve)),
                    "path_length": float(path_length[index]),
                    "conformational_diversity": float(diversity[index]),
                    "cycle_rmsd": float(
                        diag.local_rmsd(
                            diag.integrate(
                                model,
                                diag.reverse_batch(batch, start=generated),
                                eval_config,
                            )[1.0],
                            x0,
                            mask,
                            scale,
                        )[index]
                    ),
                }
            )
            if len(trajectories) < trajectory_limit:
                item = {
                    "kinase": raw["kinase_name"][index],
                    "inactive_pdb": inactive_pdb,
                    "active_pdb": active_pdb,
                    "source_state": source_state,
                    "target_state": target_state,
                    "direction": direction,
                    "x0": x0[index].numpy(),
                    "x1": x1[index].numpy(),
                    "path": {time: value[index].numpy() for time, value in paths.items()},
                    "scale": float(scale[index]),
                }
                if export_paths and direction == "activation":
                    pair_dir = export_pdb_bundle(item, output, len(trajectories))
                    item["pair_dir"] = str(pair_dir)
                trajectories.append(item)
            seen += 1
            if max_samples and seen >= max_samples:
                break
        if max_samples and seen >= max_samples:
            break

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(output / "metrics" / f"{name}_per_pair.csv", index=False)
        per_kinase = (
            frame.groupby("kinase")
            .agg(
                pairs=("global_ca_rmsd", "size"),
                global_ca_rmsd=("global_ca_rmsd", "mean"),
                improvement=("relative_improvement", "mean"),
                dfg_rmsd=("dfg_rmsd", "mean"),
                alphac_rmsd=("alphac_rmsd", "mean"),
                hrd_rmsd=("hrd_rmsd", "mean"),
                activation_loop_rmsd=("activation_loop_rmsd", "mean"),
                distance_matrix_mae=("distance_matrix_mae", "mean"),
                directionality=("directionality_fraction", "mean"),
                smoothness=("trajectory_smoothness", "mean"),
            )
            .reset_index()
        )
    else:
        per_kinase = pd.DataFrame()
    if point_rows:
        pd.DataFrame(point_rows).to_csv(output / "metrics" / f"{name}_latent_hypotheses.csv", index=False)
    return frame, trajectories, per_kinase


def select_guidance_scale(
    model: diag.DiagnosticFlowField,
    dataset: diag.AlignedDataset,
    config: diag.DiagnosticConfig,
    output: Path,
    name: str,
    max_samples: int = 8,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    for scale in GUIDANCE_GRID:
        frame, _, _ = evaluate_with_guidance(model, dataset, config, scale, max_samples, output, f"{name}_guidance_{scale:.1f}")
        rows.append(
            {
                "guidance_scale": scale,
                "global_ca_rmsd": float(frame["global_ca_rmsd"].mean()) if not frame.empty else float("inf"),
                "directionality": float(frame["directionality_fraction"].mean()) if not frame.empty else float("nan"),
                "distance_matrix_mae": float(frame["distance_matrix_mae"].mean()) if not frame.empty else float("nan"),
                "improvement": float(frame["relative_improvement"].mean()) if not frame.empty else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["global_ca_rmsd", "distance_matrix_mae"], ascending=[True, True])
    table.to_csv(output / "metrics" / f"{name}_guidance_sweep.csv", index=False)
    return float(table.iloc[0]["guidance_scale"]), table


def compare_models(
    base: diag.DiagnosticConfig,
    direct_pairs: Mapping[str, Sequence[diag.AlignedPair]],
    inverse_pairs: Mapping[str, Sequence[diag.AlignedPair]],
    lookup: Mapping[str, diag.StructureRecord],
    kinase_vocab: Mapping[str, int],
    output: Path,
    selected_lr: float,
    smoke: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]]]:
    config_map = {
        "aligned_full_conditioning": diag.DiagnosticConfig(
            **{
                **asdict(base),
                "epochs": 120 if not smoke else 2,
                "patience": 20 if not smoke else 1,
                "learning_rate": selected_lr,
                "guidance_scale": 2.0,
                "use_ligand_condition": True,
                "use_resolution_condition": True,
                "self_consistency_weight": 0.0,
            }
        ),
        "bidirectional_self_consistent": diag.DiagnosticConfig(
            **{
                **asdict(base),
                "epochs": 120 if not smoke else 2,
                "patience": 20 if not smoke else 1,
                "learning_rate": selected_lr,
                "guidance_scale": 2.0,
                "bidirectional_training": True,
                "use_ligand_condition": True,
                "use_resolution_condition": True,
            }
        ),
        "inverse_deactivation": diag.DiagnosticConfig(
            **{
                **asdict(base),
                "epochs": 120 if not smoke else 2,
                "patience": 20 if not smoke else 1,
                "learning_rate": selected_lr,
                "guidance_scale": 2.0,
                "use_ligand_condition": True,
                "use_resolution_condition": True,
            }
        ),
        "no_conditioning": diag.DiagnosticConfig(
            **{
                **asdict(base),
                "epochs": 120 if not smoke else 2,
                "patience": 20 if not smoke else 1,
                "learning_rate": selected_lr,
                "guidance_scale": 0.0,
                "condition_dropout": 1.0,
                "use_kinase_condition": False,
                "use_ligand_condition": False,
                "use_resolution_condition": False,
            }
        ),
    }

    summary_rows = []
    trajectory_index: Dict[str, List[Dict[str, Any]]] = {}
    for name, cfg in config_map.items():
        if name == "inverse_deactivation":
            train_dataset = diag.AlignedDataset(inverse_pairs["train"], lookup, cfg, kinase_vocab)
            val_dataset = diag.AlignedDataset(inverse_pairs["val"], lookup, cfg, kinase_vocab)
        elif name == "bidirectional_self_consistent":
            mirrored = diag.mirror_pairs(direct_pairs)
            merged = {split: list(direct_pairs[split]) + list(mirrored[split]) for split in diag.SPLITS}
            train_dataset = diag.AlignedDataset(merged["train"], lookup, cfg, kinase_vocab)
            val_dataset = diag.AlignedDataset(merged["val"], lookup, cfg, kinase_vocab)
        else:
            train_dataset = diag.AlignedDataset(direct_pairs["train"], lookup, cfg, kinase_vocab)
            val_dataset = diag.AlignedDataset(direct_pairs["val"], lookup, cfg, kinase_vocab)
        model, _ = diag.train_variant(cfg, train_dataset, val_dataset, output, name)

        val_scale, sweep = select_guidance_scale(model, val_dataset, cfg, output, name)
        test_source = inverse_pairs if name == "inverse_deactivation" else direct_pairs
        test_dataset = diag.AlignedDataset(test_source["test"], lookup, cfg, kinase_vocab)
        frame, trajectories, per_kinase = evaluate_with_guidance(
            model,
            test_dataset,
            cfg,
            val_scale,
            max_samples=None if not smoke else 4,
            output=output,
            name=name,
            export_paths=(name == "aligned_full_conditioning"),
        )
        frame["best_guidance_scale"] = val_scale
        frame.to_csv(output / "metrics" / f"{name}_per_pair.csv", index=False)
        if not per_kinase.empty:
            per_kinase.to_csv(output / "metrics" / f"{name}_per_kinase.csv", index=False)
        if name == "aligned_full_conditioning" and trajectories:
            trajectory_index[name] = trajectories
            path_projection(trajectories, output, name)
            if trajectories:
                example = trajectories[0]
                per_point = pd.read_csv(output / "metrics" / f"{name}_latent_hypotheses.csv")
                first_pair = per_point[
                    (per_point["kinase"] == example["kinase"])
                    & (per_point["source_pdb"] == example["inactive_pdb"])
                    & (per_point["target_pdb"] == example["active_pdb"])
                ].sort_values("t")
                if not first_pair.empty:
                    plot_trajectory_bundle(example, first_pair, output, name)
        summary_rows.append(
            {
                "model": name,
                "guidance_scale": val_scale,
                "pairs": len(frame),
                "global_ca_rmsd": float(frame["global_ca_rmsd"].mean()) if not frame.empty else float("nan"),
                "initial_target_rmsd": float(frame["initial_target_rmsd"].mean()) if not frame.empty else float("nan"),
                "relative_improvement": float(frame["relative_improvement"].mean()) if not frame.empty else float("nan"),
                "distance_matrix_mae": float(frame["distance_matrix_mae"].mean()) if not frame.empty else float("nan"),
                "dfg_rmsd": float(frame["dfg_rmsd"].mean()) if not frame.empty else float("nan"),
                "alphac_rmsd": float(frame["alphac_rmsd"].mean()) if not frame.empty else float("nan"),
                "hrd_rmsd": float(frame["hrd_rmsd"].mean()) if not frame.empty else float("nan"),
                "activation_loop_rmsd": float(frame["activation_loop_rmsd"].mean()) if not frame.empty else float("nan"),
                "trajectory_smoothness": float(frame["trajectory_smoothness"].mean()) if not frame.empty else float("nan"),
                "directionality": float(frame["directionality_fraction"].mean()) if not frame.empty else float("nan"),
                "cycle_rmsd": float(frame["cycle_rmsd"].mean()) if not frame.empty else float("nan"),
                "closer_to_target_rate": float(frame["closer_to_target_than_initial"].mean()) if not frame.empty else float("nan"),
                "crossing_rate": float(frame["closer_to_target_than_source"].mean()) if not frame.empty else float("nan"),
                "closer_to_active_rate": float(frame["closer_to_active"].mean()) if not frame.empty else float("nan"),
                "closer_to_inactive_rate": float(frame["closer_to_inactive"].mean()) if not frame.empty else float("nan"),
            }
        )

    if direct_pairs["test"]:
        oracle_dataset = diag.AlignedDataset(direct_pairs["test"], lookup, base, kinase_vocab)
        oracle_frame = diag.oracle_linear_baseline(oracle_dataset, output)
        summary_rows.append(
            {
                "model": "linear_interpolation_oracle",
                "guidance_scale": 1.0,
                "pairs": len(oracle_frame),
                "global_ca_rmsd": float(oracle_frame["global_ca_rmsd"].mean()),
                "initial_target_rmsd": float(oracle_frame["initial_target_rmsd"].mean()),
                "relative_improvement": float(oracle_frame["relative_improvement"].mean()),
                "distance_matrix_mae": float(oracle_frame["local_distance_matrix_mae"].mean()),
                "dfg_rmsd": float(oracle_frame["dfg_loop_rmsd"].mean()),
                "alphac_rmsd": float(oracle_frame["alphac_rmsd"].mean()),
                "hrd_rmsd": float(oracle_frame["hrd_rmsd"].mean()),
                "activation_loop_rmsd": float(oracle_frame["activation_loop_rmsd"].mean()),
                "trajectory_smoothness": float(oracle_frame["trajectory_smoothness"].mean()),
                "directionality": float(oracle_frame["directionality_fraction"].mean()),
                "cycle_rmsd": float(oracle_frame["cycle_rmsd"].mean()),
                "closer_to_target_rate": float(oracle_frame["closer_to_target_than_initial"].mean()),
                "crossing_rate": float(oracle_frame["crosses_to_target"].mean()),
                "closer_to_active_rate": 1.0,
                "closer_to_inactive_rate": 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("global_ca_rmsd")
    summary.to_csv(output / "comparison_table.csv", index=False)
    return summary, trajectory_index


def write_report(summary: pd.DataFrame, output: Path, selected_name: str) -> None:
    learned = summary[summary["model"] != "linear_interpolation_oracle"]
    direct = learned[learned["model"] == "aligned_full_conditioning"].iloc[0] if not learned[learned["model"] == "aligned_full_conditioning"].empty else None
    bidir = learned[learned["model"] == "bidirectional_self_consistent"].iloc[0] if not learned[learned["model"] == "bidirectional_self_consistent"].empty else None
    inverse = learned[learned["model"] == "inverse_deactivation"].iloc[0] if not learned[learned["model"] == "inverse_deactivation"].empty else None
    no_cond = learned[learned["model"] == "no_conditioning"].iloc[0] if not learned[learned["model"] == "no_conditioning"].empty else None
    best = learned.sort_values("global_ca_rmsd").iloc[0] if not learned.empty else None
    direct_dir = output / "outputs" / "transition_paths"
    sample_dirs = sorted(direct_dir.rglob("generated_t100.pdb")) if direct_dir.exists() else []
    path_section = []
    for idx, path in enumerate(sample_dirs[:3], start=1):
        pair_dir = path.parent
        path_section.append(f"- [{pair_dir.name}]({pair_dir.resolve()})")
    best_model = str(best["model"]) if best is not None else "n/a"
    best_guidance = f"{best['guidance_scale']:.2f}" if best is not None else "n/a"
    best_rmsd = f"{best['global_ca_rmsd']:.3f} A" if best is not None else "n/a"
    best_improvement = f"{100 * best['relative_improvement']:.2f}%" if best is not None else "n/a"
    text = f"""# Stage-2 transition learning report

## Main result

- Best learned model: **{best_model}**.
- Best guidance scale: **{best_guidance}**.
- Best global RMSD: **{best_rmsd}**.
- Relative improvement: **{best_improvement}**.
- The model still cannot be claimed as a biological mechanism without external validation.

## Comparison summary

| Model | Guidance | RMSD final | Improvement | Directionality | Smoothness | Cycle RMSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for _, row in summary.iterrows():
        text += (
            f"| {row['model']} | {row['guidance_scale']:.2f} | {row['global_ca_rmsd']:.3f} | "
            f"{100 * row['relative_improvement']:.2f}% | {row['directionality']:.3f} | "
            f"{row['trajectory_smoothness']:.3f} | {row['cycle_rmsd']:.3f} |\n"
        )
    text += f"""
## Camino conformacional generado por el modelo

- Modelo de referencia para el camino exportado: `{selected_name}`.
- Puntos exportados: `t=0.00`, `0.25`, `0.50`, `0.75`, `1.00`.
- Los puntos intermedios son hipótesis latentes generadas por el modelo.
- Los archivos PDB se guardan bajo `outputs/transition_paths/`.

### Archivos exportados

{chr(10).join(path_section) if path_section else '- Sin trayectorias exportadas.'}

### Interpretación

La etapa 2 mejora el experimento si baja RMSD, aumenta la direccionalidad,
reduce el error geométrico interno y mantiene suavidad en la trayectoria. La
validez biológica sigue sin poder afirmarse solo con estas métricas; hace falta
validación experimental o simulación molecular adicional.
"""
    (output / "stage2_report.md").write_text(text, encoding="utf-8")


def prepare(root: Path, output: Path, seed: int, epochs: int, patience: int, num_points: int, max_pairs_per_kinase: int) -> Tuple[diag.DiagnosticConfig, Dict[str, List[diag.StructureRecord]], Dict[str, List[diag.AlignedPair]], Dict[str, diag.StructureRecord], Dict[str, int], Path]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    config = diag.DiagnosticConfig(
        seed=seed,
        epochs=epochs,
        patience=patience,
        num_points=num_points,
        max_pairs_per_kinase=max_pairs_per_kinase,
        learning_rate=1e-4,
        guidance_scale=2.0,
        use_ligand_condition=True,
        use_resolution_condition=True,
    )
    records, audit = diag.load_records(root)
    audit.to_csv(output / "structure_annotation_audit.csv", index=False)
    direct_pairs, _ = diag.build_aligned_pairs(records, ["inactive"], ["active"], config)
    direct_pairs, holdout_note = diag.diagnostic_holdout(direct_pairs)
    inverse_pairs, _ = diag.build_aligned_pairs(records, ["active"], ["inactive"], config)
    inverse_pairs, _ = diag.diagnostic_holdout(inverse_pairs)
    diag.json_dump(asdict(config), output / "stage2_config.json")
    diag.json_dump({"holdout_note": holdout_note}, output / "evaluation_split_notes.json")
    lookup = diag.record_lookup(records)
    kinase_vocab = {
        kinase: index for index, kinase in enumerate(sorted({record.kinase for record in records["train"]}), start=0)
    }
    return config, records, {"direct": direct_pairs["train"], "val": direct_pairs["val"], "test": direct_pairs["test"], "inverse_train": inverse_pairs["train"], "inverse_val": inverse_pairs["val"], "inverse_test": inverse_pairs["test"]}, lookup, kinase_vocab, output


def run(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve() if args.root else diag.discover_root(Path.cwd())
    output = Path(args.output).resolve() if args.output else root / "results" / "flow_matching_cfg_transition_stage2"
    config, records, pair_bundle, lookup, kinase_vocab, output = prepare(
        root,
        output,
        args.seed,
        args.epochs,
        args.patience,
        args.num_points,
        args.max_pairs_per_kinase,
    )
    direct_pairs = {
        "train": pair_bundle["direct"],
        "val": pair_bundle["val"],
        "test": pair_bundle["test"],
    }
    inverse_pairs = {
        "train": pair_bundle["inverse_train"],
        "val": pair_bundle["inverse_val"],
        "test": pair_bundle["inverse_test"],
    }
    diag.LOG.info(
        "records=%s direct_pairs=%s inverse_pairs=%s",
        {split: len(values) for split, values in records.items()},
        {split: len(values) for split, values in direct_pairs.items()},
        {split: len(values) for split, values in inverse_pairs.items()},
    )

    base_variants = diag.variant_configs(config)
    selected_cfg = base_variants["aligned_full_conditioning"]
    sweep_rows = []
    best_lr = None
    best_loss = float("inf")
    best_run_name = None
    for lr in LR_GRID:
        cfg = diag.DiagnosticConfig(**{**asdict(selected_cfg), "learning_rate": lr, "epochs": args.epochs, "patience": args.patience})
        train_dataset = diag.AlignedDataset(direct_pairs["train"], lookup, cfg, kinase_vocab)
        val_dataset = diag.AlignedDataset(direct_pairs["val"], lookup, cfg, kinase_vocab)
        model, history = diag.train_variant(cfg, train_dataset, val_dataset, output, f"aligned_full_conditioning_lr_{lr:.0e}")
        loss = float(history["validation_loss"].min()) if not history.empty else float("inf")
        sweep_rows.append({"learning_rate": lr, "best_validation_loss": loss})
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            best_run_name = f"aligned_full_conditioning_lr_{lr:.0e}"
    pd.DataFrame(sweep_rows).to_csv(output / "training_sweep_summary.csv", index=False)
    if best_lr is None:
        raise RuntimeError("Could not select a learning rate.")
    if best_run_name:
        best_checkpoint = output / "models" / best_run_name / "best_model.pt"
        if best_checkpoint.exists():
            target = output / "best_aligned_full_conditioning_checkpoint.pt"
            target.write_bytes(best_checkpoint.read_bytes())
        diag.json_dump(
            {"best_run": best_run_name, "best_learning_rate": best_lr, "best_validation_loss": best_loss},
            output / "best_training_selection.json",
        )

    # Train comparison models at the selected learning rate.
    selected_base = diag.DiagnosticConfig(**{**asdict(selected_cfg), "learning_rate": best_lr, "epochs": args.epochs, "patience": args.patience})
    summary, _ = compare_models(
        selected_base,
        direct_pairs,
        inverse_pairs,
        lookup,
        kinase_vocab,
        output,
        selected_lr=best_lr,
        smoke=args.command == "smoke",
    )
    write_report(summary, output, "aligned_full_conditioning")

    if args.command == "smoke":
        return 0
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "run"])
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-points", type=int, default=192)
    parser.add_argument("--max-pairs-per-kinase", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    args = parse_args()
    if args.command == "smoke":
        args.epochs = min(args.epochs, 2)
        args.patience = 1
        args.num_points = min(args.num_points, 64)
        args.max_pairs_per_kinase = min(args.max_pairs_per_kinase, 4)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
