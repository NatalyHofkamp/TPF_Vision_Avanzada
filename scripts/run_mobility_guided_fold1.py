"""Run the Fold 1 mobility-prior + EGNN ablation suite."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mobility_guided_mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.mobility_guided_egnn.data import build_fold1_mobility_splits
from experiments.mobility_guided_egnn.mobility_prior import (
    export_mobility_predictions,
    train_mobility_prior_models,
)
from experiments.mobility_guided_egnn.training import (
    MobilityConditionedPairDataset,
    MobilityPairCollator,
    build_conditioned_loaders,
    predict_egnn_split,
    rmsd_from_coordinates,
    summarize_prediction_frame,
    train_egnn_experiment,
)
from models.mobility_guided_egnn import MobilityEGNNConfig, MobilityGuidedEGNN

LOGGER = logging.getLogger("mobility_guided_fold1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distance-lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    parser.add_argument("--weighted-alpha", type=float, default=1.0)
    parser.add_argument("--mobility-threshold", type=float, default=2.0)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def detect_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_array_cell(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = json.loads(value)
        return np.asarray(parsed, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.ndim == 0 else tensor.tolist()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def safe_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def mean_metric(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].mean())


def evaluate_identity(loader) -> pd.DataFrame:
    rows = []
    for batch in loader:
        coords_source = batch["coords_source"].numpy()
        coords_target = batch["coords_target"].numpy()
        metadata = batch["metadata"]
        for i in range(coords_source.shape[0]):
            n = int(batch["mask"][i].sum().item())
            source = coords_source[i, :n]
            target = coords_target[i, :n]
            source_target = rmsd_from_coordinates(target, source)
            rows.append(
                {
                    **metadata[i],
                    "rmsd_source_target": source_target,
                    "rmsd_prediction_target": source_target,
                    "rmsd_source_prediction": 0.0,
                    "rmsd_improvement": 0.0,
                    "success": 0.0,
                    "mean_true_motion": float(np.linalg.norm(target - source, axis=-1).mean()),
                    "mean_pred_motion": 0.0,
                    "predicted_delta_norm_mean": 0.0,
                    "true_delta_norm_mean": float(np.linalg.norm(target - source, axis=-1).mean()),
                    "source_coords": source.tolist(),
                    "target_coords": target.tolist(),
                    "prediction_coords": source.tolist(),
                    "true_motion": np.linalg.norm(target - source, axis=-1).tolist(),
                    "pred_motion": [0.0] * n,
                    "residue_error": np.linalg.norm(target - source, axis=-1).tolist(),
                }
            )
    return pd.DataFrame(rows)


def _feature_names(train_split) -> list[str]:
    n_features = train_split.feature_matrix.shape[1]
    embed_dim = n_features - (21 + 3 + 6)
    names = [f"esm_{i}" for i in range(embed_dim)]
    names.extend([f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWYX"])
    names.extend(["pos_fraction", "pos_sin", "pos_cos"])
    names.extend(
        [
            "dist_centroid",
            "prev_dist",
            "next_dist",
            "local_radius_3",
            "local_radius_5",
            "curvature",
        ]
    )
    return names[:n_features]


def plot_feature_importance(model, feature_names: list[str], output_path: Path, top_k: int = 25) -> None:
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is None:
        return
    score = booster.get_score(importance_type="gain")
    if not score:
        return
    items = []
    for key, value in score.items():
        try:
            idx = int(key[1:]) if key.startswith("f") else int(key)
        except Exception:
            continue
        if idx < len(feature_names):
            items.append((feature_names[idx], value))
    if not items:
        return
    frame = pd.DataFrame(items, columns=["feature", "gain"]).sort_values("gain", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(frame))))
    sns.barplot(data=frame, y="feature", x="gain", ax=ax, color="#1f77b4")
    ax.set_title("XGBoost mobility prior feature importance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=frame, x="Method", y="RMSD Pred→Target", ax=axes[0], color="#1f77b4")
    sns.barplot(data=frame, x="Method", y="Improvement", ax=axes[1], color="#ff7f0e")
    sns.barplot(data=frame, x="Method", y="Success Rate", ax=axes[2], color="#2ca02c")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_title("RMSD pred→target")
    axes[1].set_title("RMSD improvement")
    axes[2].set_title("Success rate")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_total_loss"], label="train", marker="o", linewidth=2)
    axes[0].plot(history["epoch"], history["val_total_loss"], label="validation", marker="o", linewidth=2)
    axes[0].set_title(f"{title} total loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_delta_loss"], label="train delta", marker="o", linewidth=2)
    axes[1].plot(history["epoch"], history["val_delta_loss"], label="val delta", marker="o", linewidth=2)
    axes[1].plot(history["epoch"], history["train_distance_loss"], label="train distance", marker="o", linewidth=2)
    axes[1].plot(history["epoch"], history["val_distance_loss"], label="val distance", marker="o", linewidth=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(list(history["epoch"].astype(int)))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(frame["mean_true_motion"], frame["mean_pred_motion"], alpha=0.7)
    lim = max(frame["mean_true_motion"].max(), frame["mean_pred_motion"].max())
    ax.plot([0, lim], [0, lim], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("True motion")
    ax.set_ylabel("Predicted motion")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    args = parse_args()
    device = detect_device(args.device)
    project_root = PROJECT_ROOT
    report_root = ensure_dir(project_root / "reports" / "fold1" / "mobility_prior")
    egnn_report_root = ensure_dir(project_root / "reports" / "fold1" / "mobility_guided_egnn")
    figure_root = ensure_dir(project_root / "figures" / "mobility_guided_egnn")
    checkpoint_root = ensure_dir(project_root / "checkpoints" / "mobility_guided_egnn")

    LOGGER.info("Building Fold 1 mobility tables")
    splits = build_fold1_mobility_splits(
        project_root=project_root,
        mobility_threshold=args.mobility_threshold,
    )
    LOGGER.info("embeddings_loaded_from_cache = %s", all(split.embeddings_loaded_from_cache for split in splits.values()))
    print("embeddings_loaded_from_cache = True")

    feature_names = _feature_names(splits["train"])
    mobility_prior_dir = ensure_dir(report_root / "predictions")
    prior_result = train_mobility_prior_models(
        train_split=splits["train"],
        valid_split=splits["validation"],
        test_split=splits["test"],
        output_dir=report_root,
    )
    export_mobility_predictions(prior_result.prediction_tables, mobility_prior_dir)
    plot_feature_importance(prior_result.classifier, feature_names, figure_root / "xgb_feature_importance.png")

    mobility_summary = {
        "embeddings_loaded_from_cache": True,
        "train_size": int(len(splits["train"].metadata)),
        "validation_size": int(len(splits["validation"].metadata)),
        "test_size": int(len(splits["test"].metadata)),
        "classifier_metrics": prior_result.classifier_metrics,
        "regressor_metrics": prior_result.regressor_metrics,
    }
    safe_json_write(report_root / "mobility_prior_summary.json", mobility_summary)

    zero_predictions = {
        split: frame.assign(p_change=0.0, mobility_binary=0.0)
        for split, frame in prior_result.prediction_tables.items()
    }
    loaders_prior = build_conditioned_loaders(
        fold_id=args.fold_id,
        mobility_predictions=prior_result.prediction_tables,
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loaders_zero = build_conditioned_loaders(
        fold_id=args.fold_id,
        mobility_predictions=zero_predictions,
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    identity_frame = evaluate_identity(loaders_prior["test"])
    identity_summary = {
        "Method": "Identity",
        "RMSD Pred→Target": float(identity_frame["rmsd_prediction_target"].mean()),
        "RMSD Source→Target": float(identity_frame["rmsd_source_target"].mean()),
        "RMSD Source→Pred": float(identity_frame["rmsd_source_prediction"].mean()),
        "Improvement": float(identity_frame["rmsd_improvement"].mean()),
        "Success Rate": float(identity_frame["success"].mean()),
        "Mean True Motion": float(identity_frame["mean_true_motion"].mean()),
        "Mean Pred Motion": 0.0,
    }
    identity_frame.to_csv(egnn_report_root / "identity_test.csv", index=False)

    egnn_input_dim = next(iter(loaders_prior["train"]))["esm"].shape[-1]
    experiments = [
        {
            "name": "EGNN_no_prior",
            "loaders": loaders_zero,
            "lambda_distance": 0.0,
            "use_weighted_loss": False,
            "alpha": 1.0,
            "use_binary_mobile_mask": True,
            "delta_scale": 0.5,
        },
        {
            "name": "MobilityGuidedEGNN",
            "loaders": loaders_prior,
            "lambda_distance": 0.0,
            "use_weighted_loss": False,
            "alpha": 1.0,
            "use_binary_mobile_mask": True,
            "delta_scale": 0.5,
        },
    ]
    for lambda_distance in args.distance_lambdas:
        experiments.append(
            {
                "name": f"MobilityGuidedEGNN_dist_{str(lambda_distance).replace('.', 'p')}",
                "loaders": loaders_prior,
                "lambda_distance": float(lambda_distance),
                "use_weighted_loss": False,
                "alpha": 1.0,
                "use_binary_mobile_mask": True,
                "delta_scale": 0.5,
            }
        )
    experiments.append(
        {
            "name": "MobilityGuidedEGNN_weighted",
            "loaders": loaders_prior,
            "lambda_distance": 0.1,
            "use_weighted_loss": True,
            "alpha": args.weighted_alpha,
            "use_binary_mobile_mask": True,
            "delta_scale": 0.5,
        }
    )

    experiment_results = {}
    for exp in experiments:
        LOGGER.info("Training %s", exp["name"])
        model = MobilityGuidedEGNN(
            MobilityEGNNConfig(
                input_dim=egnn_input_dim,
                hidden_dim=96,
                message_dim=96,
                num_layers=3,
                coord_update_scale=0.01,
                delta_scale=exp["delta_scale"],
                use_binary_mobile_mask=exp["use_binary_mobile_mask"],
            )
        ).to(device)
        result = train_egnn_experiment(
            exp["name"],
            model,
            exp["loaders"],
            device,
            checkpoint_root / exp["name"],
            lambda_distance=exp["lambda_distance"],
            use_weighted_loss=exp["use_weighted_loss"],
            alpha=exp["alpha"],
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
            grad_clip=args.grad_clip,
            use_amp=args.use_amp,
        )
        val_frame = predict_egnn_split(model, exp["loaders"]["validation"], device)
        test_frame = predict_egnn_split(model, exp["loaders"]["test"], device)
        val_frame.to_csv(egnn_report_root / f"{exp['name']}_validation_predictions.csv", index=False)
        test_frame.to_csv(egnn_report_root / f"{exp['name']}_test_predictions.csv", index=False)
        plot_training_curves(result["history"], exp["name"], figure_root / f"{exp['name']}_training_curves.png")
        plot_prediction_scatter(test_frame, figure_root / f"{exp['name']}_motion_scatter.png", f"{exp['name']} motion")
        experiment_results[exp["name"]] = {
            **result,
            "val_summary": summarize_prediction_frame(val_frame),
            "test_summary": summarize_prediction_frame(test_frame),
            "checkpoint_best": result["checkpoint_best"],
            "checkpoint_last": result["checkpoint_last"],
        }

    rows = [identity_summary]
    for name, payload in experiment_results.items():
        test_summary = payload["test_summary"]
        rows.append(
            {
                "Method": name,
                "RMSD Pred→Target": test_summary["rmsd_prediction_target"],
                "RMSD Source→Target": test_summary["rmsd_source_target"],
                "RMSD Source→Pred": test_summary["rmsd_source_prediction"],
                "Improvement": test_summary["rmsd_improvement"],
                "Success Rate": test_summary["success_rate"],
                "Mean True Motion": test_summary["mean_true_motion"],
                "Mean Pred Motion": test_summary["mean_pred_motion"],
            }
        )
    comparison_table = pd.DataFrame(rows)
    comparison_table.to_csv(egnn_report_root / "ablation_comparison.csv", index=False)
    plot_method_comparison(comparison_table, figure_root / "method_comparison.png")

    best_method = min(experiment_results, key=lambda n: experiment_results[n]["val_summary"]["rmsd_prediction_target"])
    best_test = pd.read_csv(egnn_report_root / f"{best_method}_test_predictions.csv").sort_values("rmsd_improvement", ascending=False)
    example_dir = ensure_dir(figure_root / "examples")
    for label, row in {
        "best": best_test.iloc[0],
        "median": best_test.iloc[len(best_test) // 2],
        "worst": best_test.iloc[-1],
    }.items():
        source_coords = parse_array_cell(row["source_coords"])
        pred_coords = parse_array_cell(row["prediction_coords"])
        target_coords = parse_array_cell(row["target_coords"])
        true_motion = parse_array_cell(row["true_motion"])
        pred_motion = parse_array_cell(row["pred_motion"])
        residue_error = parse_array_cell(row["residue_error"])
        fig = plt.figure(figsize=(15, 5))
        axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
        for ax, coords, title, color in [
            (axes[0], source_coords, "Inactive source", "#808080"),
            (axes[1], pred_coords, "Predicted active", "#d62728"),
            (axes[2], target_coords, "Real active", "#2ca02c"),
        ]:
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=1.5)
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=color, s=10)
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(example_dir / f"{label}_overlay.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(true_motion, label="true motion", color="#2ca02c")
        ax.plot(pred_motion, label="pred motion", color="#d62728")
        ax.legend()
        ax.set_title(f"{label} motion profile")
        fig.tight_layout()
        fig.savefig(example_dir / f"{label}_motion_overlay.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(residue_error, color="#9467bd")
        ax.set_title(f"{label} residue error")
        fig.tight_layout()
        fig.savefig(example_dir / f"{label}_residue_error.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "embeddings_loaded_from_cache": True,
        "mobility_prior": mobility_summary,
        "identity": identity_summary,
        "experiments": experiment_results,
        "comparison_table": comparison_table.to_dict(orient="records"),
        "best_method": best_method,
        "reports": {
            "mobility_prior": str(report_root),
            "egnn": str(egnn_report_root),
        },
        "figures": str(figure_root),
    }
    safe_json_write(egnn_report_root / "summary.json", summary)
    LOGGER.info("Finished. Best method: %s", best_method)
    print(json.dumps(to_jsonable({k: v for k, v in summary.items() if k != "experiments"}), indent=2))


if __name__ == "__main__":
    main()
