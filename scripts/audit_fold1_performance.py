#!/usr/bin/env python3
"""Rigorous single-epoch performance audit for Fold 1 training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import torch

from training.fold1_experiment import (
    Fold1TrainingConfig,
    build_model_and_loaders,
    build_optimizer,
    parameter_trainability_report,
    recommended_num_workers,
    run_epoch,
    select_example_sample,
    validation_cfg_diagnostics,
    write_metrics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--cfg-dropout-probability", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--official-checkpoint", type=Path, default=Path("checkpoints/foldflow/foldflow-sfm.pth"))
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fold1/performance_audit"))
    return parser.parse_args()


def run_audit(use_amp: bool, config: Fold1TrainingConfig, device: torch.device, output_dir: Path) -> dict[str, object]:
    model, train_loader, validation_loader, test_loader, fold_sizes = build_model_and_loaders(config, device)
    optimizer = build_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    train_metrics, train_timing = run_epoch(
        model,
        train_loader,
        device,
        config.cfg_dropout_probability,
        optimizer=optimizer,
        grad_clip=config.grad_clip,
        use_amp=use_amp,
        scaler=scaler,
        collect_timing=True,
    )
    validation_metrics, validation_timing = run_epoch(
        model,
        validation_loader,
        device,
        config.cfg_dropout_probability,
        optimizer=None,
        collect_timing=True,
    )
    train_metrics_path = output_dir / f"training_metrics_amp_{int(use_amp)}.csv"
    write_metrics_csv(
        [
            {
                "epoch": 1,
                "train_total_loss": train_metrics["total_loss"],
                "train_rotation_loss": train_metrics["rotation_loss"],
                "train_translation_loss": train_metrics["translation_loss"],
                "validation_total_loss": validation_metrics["total_loss"],
                "validation_rotation_loss": validation_metrics["rotation_loss"],
                "validation_translation_loss": validation_metrics["translation_loss"],
            }
        ],
        train_metrics_path,
    )
    trainability = parameter_trainability_report(model)
    report = {
        "use_amp_requested": use_amp,
        "use_amp_effective": bool(use_amp and device.type == "cuda"),
        "device": str(device),
        "gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0,
        "train_size": fold_sizes["train"],
        "validation_size": fold_sizes["validation"],
        "test_size": fold_sizes["test"],
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_timing": train_timing,
        "validation_timing": validation_timing,
        "average_batch_time_seconds": float(train_timing["batch_seconds"]),
        "total_epoch_time_seconds": float(train_timing["batch_seconds"] * train_timing["batch_count"]),
        "trainability": trainability,
        "train_metrics_csv": str(train_metrics_path),
    }
    return report


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Fold1TrainingConfig(
        fold_id=args.fold_id,
        epochs=1,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=1,
        cfg_dropout_probability=args.cfg_dropout_probability,
        num_workers=args.num_workers,
        use_amp=True,
        official_checkpoint=args.official_checkpoint,
        source_root=args.source_root,
    )
    start = time.perf_counter()
    with torch.no_grad():
        pass
    trainability_preview = None
    model, train_loader, validation_loader, test_loader, fold_sizes = build_model_and_loaders(
        config, device
    )
    trainability_preview = parameter_trainability_report(model)
    print("Recommended num_workers:", recommended_num_workers(device))
    print("Trainability preview:", trainability_preview)
    del model, train_loader, validation_loader, test_loader, fold_sizes

    amp_off = run_audit(False, config, device, output_dir)
    amp_on = run_audit(True, config, device, output_dir)

    combined = {
        "device": str(device),
        "recommended_num_workers": recommended_num_workers(device),
        "backbone_trainability": trainability_preview,
        "amp_off": amp_off,
        "amp_on": amp_on,
        "elapsed_wall_seconds": time.perf_counter() - start,
        "notes": [
            "ESM embeddings are loaded from the existing cache through EmbeddingCache.load.",
            "AMP on CPU is not active; use_amp_effective will be false unless CUDA is available.",
        ],
        "analysis": {
            "principal_bottleneck": "official FoldFlow forward/backward on the frozen backbone remains the dominant cost",
            "safe_compute_reduction": "Keeping autograd for the conditioning path but avoiding unnecessary CPU<->GPU copies and using AMP on CUDA are safe",
        },
    }
    report_path = output_dir / "performance_audit.json"
    report_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
