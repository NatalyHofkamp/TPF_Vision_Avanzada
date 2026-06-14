"""Leave-One-Kinase-Out fold generation and leakage validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

FOLD_COLUMNS = [
    "kinase",
    "pdb_id",
    "chain",
    "conformational_state",
    "sequence_hash",
    "embedding_file",
]

DEFAULT_KINASE_ORDER = [
    "EGFR",
    "BRAF",
    "ABL1",
    "FGFR1",
    "MET",
    "KIT",
    "ALK",
    "PIK3CA",
    "AKT1",
    "CDK4",
    "CDK6",
    "PDGFRA",
    "ERBB2",
]


class LeaveOneKinaseOutSplitter:
    """Generate deterministic kinase-level train/validation/test folds."""

    def __init__(
        self,
        manifest: pd.DataFrame | Path | str = Path("data/esm_manifest.csv"),
        output_dir: Path | str = Path("data/folds"),
        kinase_order: Sequence[str] | None = None,
    ):
        self.manifest = (
            pd.read_csv(manifest) if isinstance(manifest, (str, Path)) else manifest.copy()
        )
        required = set(FOLD_COLUMNS)
        missing = required - set(self.manifest.columns)
        if missing:
            raise ValueError(f"ESM manifest is missing columns: {sorted(missing)}")
        self.output_dir = Path(output_dir)
        available = list(dict.fromkeys(self.manifest["kinase"].astype(str)))
        if kinase_order is None:
            canonical = [kinase for kinase in DEFAULT_KINASE_ORDER if kinase in available]
            extras = sorted(set(available) - set(canonical))
            self.kinases = canonical + extras
        else:
            requested = list(kinase_order)
            if set(requested) != set(available) or len(requested) != len(set(requested)):
                raise ValueError("kinase_order must contain every manifest kinase exactly once")
            self.kinases = requested
        if len(self.kinases) < 3:
            raise ValueError("LOKO requires at least three kinases")

    def assignments(self) -> list[dict[str, Any]]:
        assignments = []
        for index, test_kinase in enumerate(self.kinases):
            validation_kinase = self.kinases[(index + 1) % len(self.kinases)]
            train_kinases = [
                kinase
                for kinase in self.kinases
                if kinase not in {test_kinase, validation_kinase}
            ]
            assignments.append(
                {
                    "fold_id": index + 1,
                    "test_kinase": test_kinase,
                    "validation_kinase": validation_kinase,
                    "train_kinases": train_kinases,
                }
            )
        return assignments

    @staticmethod
    def validate_fold(
        train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
    ) -> dict[str, Any]:
        groups = {
            "train": set(train["kinase"].unique()),
            "validation": set(validation["kinase"].unique()),
            "test": set(test["kinase"].unique()),
        }
        overlaps = {
            "train_validation": sorted(groups["train"] & groups["validation"]),
            "train_test": sorted(groups["train"] & groups["test"]),
            "validation_test": sorted(groups["validation"] & groups["test"]),
        }
        leaking = {name: values for name, values in overlaps.items() if values}
        if leaking:
            raise ValueError(f"Kinase leakage detected: {leaking}")
        return {
            "valid": True,
            "train_kinases": sorted(groups["train"]),
            "validation_kinases": sorted(groups["validation"]),
            "test_kinases": sorted(groups["test"]),
            "overlaps": overlaps,
        }

    @staticmethod
    def _split_statistics(frame: pd.DataFrame) -> dict[str, Any]:
        states = frame["conformational_state"].value_counts()
        return {
            "structures": int(len(frame)),
            "active": int(states.get("active", 0)),
            "inactive": int(states.get("inactive", 0)),
            "kinases": sorted(frame["kinase"].unique().tolist()),
        }

    def generate(
        self,
        statistics_path: Path | str = Path("reports/loko_statistics.json"),
        validation_path: Path | str = Path("reports/loko_validation.json"),
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fold_statistics = []
        validation_reports = []
        for assignment in self.assignments():
            fold_id = assignment["fold_id"]
            test_kinase = assignment["test_kinase"]
            validation_kinase = assignment["validation_kinase"]
            train = self.manifest[
                self.manifest["kinase"].isin(assignment["train_kinases"])
            ][FOLD_COLUMNS].reset_index(drop=True)
            validation = self.manifest[
                self.manifest["kinase"] == validation_kinase
            ][FOLD_COLUMNS].reset_index(drop=True)
            test = self.manifest[self.manifest["kinase"] == test_kinase][
                FOLD_COLUMNS
            ].reset_index(drop=True)
            validation_report = self.validate_fold(train, validation, test)
            validation_report["fold_id"] = fold_id
            validation_reports.append(validation_report)

            fold_dir = self.output_dir / f"fold_{fold_id:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            train.to_csv(fold_dir / "train.csv", index=False)
            validation.to_csv(fold_dir / "validation.csv", index=False)
            test.to_csv(fold_dir / "test.csv", index=False)
            fold_statistics.append(
                {
                    "fold_id": fold_id,
                    "kinase_assignments": {
                        "train": assignment["train_kinases"],
                        "validation": validation_kinase,
                        "test": test_kinase,
                    },
                    "train": self._split_statistics(train),
                    "validation": self._split_statistics(validation),
                    "test": self._split_statistics(test),
                }
            )

        statistics = {"number_of_folds": len(fold_statistics), "folds": fold_statistics}
        statistics_path = Path(statistics_path)
        statistics_path.parent.mkdir(parents=True, exist_ok=True)
        statistics_path.write_text(
            json.dumps(statistics, indent=2) + "\n", encoding="utf-8"
        )
        validation_payload = {
            "valid": True,
            "number_of_folds": len(validation_reports),
            "folds": validation_reports,
        }
        validation_path = Path(validation_path)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8"
        )
        return statistics
