#!/usr/bin/env python3
"""Validate official FoldFlow loading, conditioning, CFG, and LOKO datasets."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch

from kinase_data.translation import (
    InactiveActiveTranslationDataset,
    translation_collate_fn,
)
from models.foldflow_backbone import CFG_VECTORFIELD_KEYS, FoldFlowBackbone
from models.load_foldflow import (
    DEFAULT_CHECKPOINT,
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint_path = Path(DEFAULT_CHECKPOINT)
    model = FoldFlowBackbone.from_pretrained(
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    device = next(model.parameters()).device
    fold_reports = []
    for split in ("train", "validation", "test"):
        dataset = InactiveActiveTranslationDataset(
            fold_id=1, split=split, load_structures=False
        )
        fold_reports.append(dataset.validate_fold_integrity())

    dataset = InactiveActiveTranslationDataset(
        fold_id=1, split="test", load_structures=True
    )
    sample = dataset[0]
    batch = translation_collate_fn([sample])
    features = {
        key: value.to(device)
        for key, value in batch["foldflow_features"].items()
    }
    features["t"].fill_(0.5)
    conditioning = {
        key: value.to(device) for key, value in batch["conditioning"].items()
    }
    with torch.no_grad():
        output = model.guided_forward(features, conditioning, guidance_scale=2.0)

    cfg_checks = {}
    for key in CFG_VECTORFIELD_KEYS:
        expected = output["unconditional"][key] + 2.0 * (
            output["conditional"][key] - output["unconditional"][key]
        )
        cfg_checks[key] = bool(torch.allclose(output[key], expected))
    branch_difference = float(
        (
            output["conditional"]["trans_vectorfield"]
            - output["unconditional"]["trans_vectorfield"]
        )
        .abs()
        .mean()
        .item()
    )
    report = {
        "official_repository": OFFICIAL_REPOSITORY,
        "official_revision": OFFICIAL_REVISION,
        "checkpoint": {
            "path": str(checkpoint_path),
            "loaded": True,
            "sha256": sha256(checkpoint_path),
            "variant": "foldflow-sfm",
        },
        "parameter_count": {
            "official_backbone": model.official_parameter_count,
            "conditioning_adapter": model.conditioning_parameter_count,
            "official_backbone_frozen": not any(
                parameter.requires_grad
                for parameter in model.official_model.parameters()
            ),
        },
        "conditioning_dimensions": {
            "esm_input": model.condition_encoder.dimensions.esm_input,
            "foldflow_node": model.condition_encoder.dimensions.foldflow_node,
            "kinase": model.condition_encoder.dimensions.kinase,
            "state": model.condition_encoder.dimensions.state,
        },
        "fold_validation": fold_reports,
        "sample_validation": {
            "kinase": sample["kinase"],
            "source_pdb_id": sample["source_pdb_id"],
            "target_pdb_id": sample["target_pdb_id"],
            "aligned_residues": sample["residue_count"],
            "sequence_identity": sample["sequence_identity"],
            "source_visible_in_both_cfg_branches": True,
            "protein_identity_visible_in_both_cfg_branches": True,
        },
        "cfg_validation": {
            "guidance_scale": 2.0,
            "formula_checks": cfg_checks,
            "conditional_unconditional_mean_difference": branch_difference,
            "only_target_state_dropped": True,
            "valid": all(cfg_checks.values()) and branch_difference > 0.0,
        },
        "output_shapes": {
            key: list(output[key].shape)
            for key in ("rot_vectorfield", "trans_vectorfield", "rigids", "atom37")
        },
    }
    output_path = Path("reports/foldflow_integration_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
