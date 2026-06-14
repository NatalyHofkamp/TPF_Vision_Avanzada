#!/usr/bin/env python3
"""Build cached ESM embeddings, LOKO folds, reports, and figures."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinase_data.analysis import (
    analyze_kinase_similarity,
    generate_esm_projections,
    generate_loko_figures,
)
from kinase_data.esm import DEFAULT_ESM_MODEL, ESMEmbedder
from kinase_data.loko import LeaveOneKinaseOutSplitter
from kinase_data.metadata import ESMManifestBuilder

LOGGER = logging.getLogger("build_esm_loko")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, default=Path("data/metadata/kinase_labels.csv")
    )
    parser.add_argument("--model-name", default=DEFAULT_ESM_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Extract sequences and folds without running ESM or embedding-dependent reports.",
    )
    parser.add_argument(
        "--skip-figures", action="store_true", help="Do not generate projection/QC figures."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    project_root = PROJECT_ROOT
    os.chdir(project_root)
    builder = ESMManifestBuilder(
        metadata_csv=args.metadata,
        manifest_csv=Path("data/esm_manifest.csv"),
        cache_dir=Path("cache/esm_embeddings"),
        project_root=project_root,
    )
    embedder = None
    if not args.manifest_only:
        embedder = ESMEmbedder(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            chunk_overlap=args.chunk_overlap,
        )
    LOGGER.info("Building ESM manifest from %s", args.metadata)
    manifest = builder.build(
        embedder=embedder, embedding_batch_size=args.embedding_batch_size
    )
    builder.write_statistics(manifest)

    splitter = LeaveOneKinaseOutSplitter(manifest)
    LOGGER.info("Exporting %d LOKO folds", len(splitter.kinases))
    loko_statistics = splitter.generate()
    if args.manifest_only:
        builder.validate_embeddings(manifest, raise_on_error=False)
        LOGGER.info("Manifest-only build complete; embeddings were not generated")
        return

    builder.validate_embeddings(manifest)
    analyze_kinase_similarity(manifest, project_root=project_root)
    if not args.skip_figures:
        generate_esm_projections(manifest, project_root=project_root)
        generate_loko_figures(manifest, loko_statistics)
    LOGGER.info("ESM and LOKO preprocessing complete")


if __name__ == "__main__":
    main()
