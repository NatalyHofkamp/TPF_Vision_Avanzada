#!/usr/bin/env python3
"""Train and evaluate Fold 1 of the kinase translation benchmark."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fold1_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fold1_xdg_cache")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import torch

from training.fold1_experiment import (
    Fold1TrainingConfig,
    build_model_and_loaders,
    build_optimizer,
    load_checkpoint,
    plot_prediction_comparison,
    plot_trajectory_snapshots,
    run_epoch,
    run_example_prediction,
    save_checkpoint,
    save_training_curves,
    select_example_sample,
    validation_cfg_diagnostics,
    write_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--cfg-dropout-probability", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--reverse-steps", type=int, default=6)
    parser.add_argument("--reverse-min-t", type=float, default=0.0)
    parser.add_argument("--official-checkpoint", type=Path, default=Path("checkpoints/foldflow/foldflow-sfm.pth"))
    parser.add_argument("--source-root", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(42)
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    config = Fold1TrainingConfig(
        fold_id=args.fold_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        cfg_dropout_probability=args.cfg_dropout_probability,
        guidance_scale=args.guidance_scale,
        unfreeze_backbone=args.unfreeze_backbone,
        num_workers=args.num_workers,
        reverse_steps=args.reverse_steps,
        reverse_min_t=args.reverse_min_t,
        official_checkpoint=args.official_checkpoint,
        source_root=args.source_root,
    )

    model, train_loader, validation_loader, test_loader, fold_sizes = build_model_and_loaders(
        config, device
    )
    optimizer = build_optimizer(model, config)
    checkpoint_manager = Path(config.checkpoint_root)
    checkpoint_manager.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_manager / "best_validation.pt"
    last_path = checkpoint_manager / "last.pt"
    if args.resume and last_path.exists():
        payload = load_checkpoint(last_path, model, optimizer)
        start_epoch = int(payload["epoch"]) + 1
        best_validation_loss = float(payload["best_validation_loss"])
    else:
        start_epoch = 1
        best_validation_loss = float("inf")

    metrics_rows: list[dict[str, float]] = []
    no_improve = 0
    best_epoch = 0
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            config.cfg_dropout_probability,
            optimizer=optimizer,
            grad_clip=config.grad_clip,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            config.cfg_dropout_probability,
            optimizer=None,
        )
        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_rotation_loss": train_metrics["rotation_loss"],
            "train_translation_loss": train_metrics["translation_loss"],
            "validation_total_loss": validation_metrics["total_loss"],
            "validation_rotation_loss": validation_metrics["rotation_loss"],
            "validation_translation_loss": validation_metrics["translation_loss"],
        }
        metrics_rows.append(row)
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            best_validation_loss,
            config,
            metrics=row,
            extra={"fold_sizes": fold_sizes},
        )
        if validation_metrics["total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["total_loss"]
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                best_validation_loss,
                config,
                metrics=row,
                extra={"fold_sizes": fold_sizes},
            )
        else:
            no_improve += 1
        if no_improve >= config.patience:
            break

    metrics_path = write_metrics_csv(metrics_rows, config.report_root / "training_metrics.csv")
    save_training_curves(metrics_path, config.figure_root)

    cfg_diagnostics = validation_cfg_diagnostics(
        model,
        validation_loader,
        device,
        guidance_scales=(0.0, 1.0, 3.0, 5.0),
    )
    cfg_report_path = config.report_root / "cfg_diagnostics.json"
    cfg_report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_report_path.write_text(json.dumps(cfg_diagnostics, indent=2) + "\n", encoding="utf-8")

    example_sample = select_example_sample(test_loader, index=0)["sample"]
    example_prediction = run_example_prediction(
        model,
        example_sample,
        device,
        guidance_scale=config.guidance_scale,
        num_steps=config.reverse_steps,
        min_t=config.reverse_min_t,
    )
    example_prediction_path = config.report_root / "example_prediction.pt"
    example_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(example_prediction, example_prediction_path)
    plot_trajectory_snapshots(example_prediction, config.figure_root / "trajectory")
    plot_prediction_comparison(
        example_prediction,
        config.figure_root / "prediction_comparison.png",
    )

    summary = {
        "fold_id": config.fold_id,
        "train_size": fold_sizes["train"],
        "validation_size": fold_sizes["validation"],
        "test_size": fold_sizes["test"],
        "best_validation_loss": best_validation_loss,
        "final_training_loss": metrics_rows[-1]["train_total_loss"],
        "total_epochs": metrics_rows[-1]["epoch"],
        "conditioning_parameter_count": model.conditioning_parameter_count,
        "FoldFlow_parameter_count": model.official_parameter_count,
        "checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
        "metrics_csv": str(metrics_path),
        "cfg_diagnostics_path": str(cfg_report_path),
        "example_prediction_path": str(example_prediction_path),
        "best_epoch": best_epoch,
    }
    summary_path = config.report_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    duration = time.perf_counter() - start_time
    gpu_memory = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"number of training samples: {fold_sizes['train']}")
    print(f"number of validation samples: {fold_sizes['validation']}")
    print(f"number of test samples: {fold_sizes['test']}")
    print(f"number of trainable parameters: {trainable_parameters}")
    print(f"GPU memory usage: {gpu_memory:.2f} MB")
    print(f"training duration: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
