"""XGBoost-based mobility prior models for residue mobility prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from .data import MobilityResidueSplit


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    if np.std(x_rank) < 1e-12 or np.std(y_rank) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))),
        "pearson": _pearsonr(y_true, y_pred),
        "spearman": _spearmanr(y_true, y_pred),
    }


def _safe_fit(model, x_train, y_train, x_val, y_val, eval_metric: str) -> Any:
    fit_kwargs = {
        "eval_set": [(x_val, y_val)],
        "verbose": False,
    }
    try:
        return model.fit(x_train, y_train, eval_metric=eval_metric, early_stopping_rounds=50, **fit_kwargs)
    except TypeError:
        return model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)


@dataclass(frozen=True)
class MobilityPriorResult:
    classifier: XGBClassifier
    regressor: XGBRegressor
    classifier_metrics: dict[str, dict[str, float]]
    regressor_metrics: dict[str, dict[str, float]]
    prediction_tables: dict[str, pd.DataFrame]
    embeddings_loaded_from_cache: bool = True


def _fit_classifier(train: MobilityResidueSplit, valid: MobilityResidueSplit, seed: int) -> XGBClassifier:
    y_train = train.mobility_binary
    scale_pos_weight = float((len(y_train) - y_train.sum()) / max(1.0, y_train.sum()))
    model = XGBClassifier(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=2.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    _safe_fit(model, train.feature_matrix, train.mobility_binary, valid.feature_matrix, valid.mobility_binary, "aucpr")
    return model


def _fit_regressor(train: MobilityResidueSplit, valid: MobilityResidueSplit, seed: int) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=1200,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=2.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    _safe_fit(model, train.feature_matrix, train.mobility_regression, valid.feature_matrix, valid.mobility_regression, "rmse")
    return model


def _build_prediction_table(
    split: MobilityResidueSplit,
    classifier: XGBClassifier,
    regressor: XGBRegressor,
) -> pd.DataFrame:
    prob = classifier.predict_proba(split.feature_matrix)[:, 1]
    reg = regressor.predict(split.feature_matrix)
    frame = split.metadata.copy()
    frame["p_change"] = prob.astype(np.float32)
    frame["mobility_regression_pred"] = reg.astype(np.float32)
    frame["mobility_binary"] = split.mobility_binary.astype(int)
    frame["movement"] = split.mobility_regression.astype(np.float32)
    frame["probability_threshold_0.5"] = (prob >= 0.5).astype(int)
    return frame


def train_mobility_prior_models(
    train_split: MobilityResidueSplit,
    valid_split: MobilityResidueSplit,
    test_split: MobilityResidueSplit,
    output_dir: Path | str,
    seed: int = 7,
) -> MobilityPriorResult:
    """Fit XGBoost mobility classifiers/regressors and evaluate on all splits."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = _fit_classifier(train_split, valid_split, seed=seed)
    regressor = _fit_regressor(train_split, valid_split, seed=seed)

    prediction_tables = {
        "train": _build_prediction_table(train_split, classifier, regressor),
        "validation": _build_prediction_table(valid_split, classifier, regressor),
        "test": _build_prediction_table(test_split, classifier, regressor),
    }

    classifier_metrics = {
        split: _binary_metrics(
            table["mobility_binary"].to_numpy(),
            table["probability_threshold_0.5"].to_numpy(),
            table["p_change"].to_numpy(),
        )
        for split, table in prediction_tables.items()
    }
    regressor_metrics = {
        split: _regression_metrics(
            table["movement"].to_numpy(),
            table["mobility_regression_pred"].to_numpy(),
        )
        for split, table in prediction_tables.items()
    }

    for split, table in prediction_tables.items():
        table.to_csv(output_dir / f"{split}_mobility_predictions.csv", index=False)

    payload = {
        "embeddings_loaded_from_cache": True,
        "classifier_metrics": classifier_metrics,
        "regressor_metrics": regressor_metrics,
        "train_samples": int(len(train_split.metadata)),
        "validation_samples": int(len(valid_split.metadata)),
        "test_samples": int(len(test_split.metadata)),
        "feature_dimension": int(train_split.feature_matrix.shape[1]),
    }
    (output_dir / "mobility_prior_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return MobilityPriorResult(
        classifier=classifier,
        regressor=regressor,
        classifier_metrics=classifier_metrics,
        regressor_metrics=regressor_metrics,
        prediction_tables=prediction_tables,
    )


def export_mobility_predictions(
    prediction_tables: dict[str, pd.DataFrame],
    output_dir: Path | str,
) -> None:
    """Persist per-residue p(change) exports."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in prediction_tables.items():
        columns = [
            "kinase",
            "sample_index",
            "residue_index",
            "sequence_position",
            "source_pdb_id",
            "target_pdb_id",
            "residue_name",
            "p_change",
            "mobility_regression_pred",
            "movement",
            "mobility_binary",
        ]
        frame[columns].to_csv(output_dir / f"{split}_p_change.csv", index=False)

