#!/usr/bin/env python3
"""Render the fold-1 inactive/source, predicted, and target structures."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fold1_mpl")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.fold1_experiment import plot_prediction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example-prediction",
        type=Path,
        default=Path("reports/fold1/example_prediction.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/fold1/prediction_comparison.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.example_prediction, map_location="cpu", weights_only=False)
    plot_prediction_comparison(payload, args.output)
    print(str(args.output))


if __name__ == "__main__":
    main()
