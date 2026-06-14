# Official FoldFlow Migration Report

Date: 2026-06-14

## Upstream

- Official repository: `https://github.com/DreamFold/FoldFlow`
- Pinned revision inspected: `9d2c260813da3c9a2bc944973953f63ea6d71203`
- Upstream release used by this integration: `0.1.0`
- License: Creative Commons Attribution-NonCommercial 4.0
- Paper models: FoldFlow-Base, FoldFlow-OT, and FoldFlow-SFM

The integration must import the upstream `foldflow` and `openfold` Python
packages. No FoldFlow network, invariant point attention block, SE(3) flow
matcher, or reverse integration method is reimplemented locally.

## Installation Constraints

Upstream documents Python 3.9.15, PyTorch 1.13.1, and CUDA 11.6.1. Its full
environment additionally uses Hydra 1.2, Geoopt, POT, TorchSDE, TorchDiffEq,
TorchDyn, PyTorch Lightning, and a custom Geomstats fork.

The repository contains both `FoldFlow/` and `foldflow/`, which collide on
case-insensitive filesystems. The installer therefore exports the required
lowercase `foldflow/` source plus `openfold/` from the pinned Git revision
instead of checking out both paths into the worktree.

The upstream custom Geomstats fork is not Python 3.12 compatible. The local
runtime uses Geomstats 2.8.0 with its PyTorch backend, preserving the SE(3) API
used by FoldFlow while supporting the repository's current interpreter.

## Official Checkpoints

Release `0.1.0` provides:

- `foldflow-base.pth`
- `foldflow-ot.pth`
- `foldflow-sfm.pth`

Release `0.2.0` additionally provides FoldFlow-2 checkpoints:

- `ff2_base.pth`
- `ff2_reft.pth`

The FoldFlow-1 checkpoint is a Torch-serialized dictionary. The official loader
uses:

- `checkpoint["conf"].model` for architecture configuration when available
- `checkpoint["model"]` for parameters
- removal of an optional `module.` prefix
- migration of legacy `score_model.` keys to `vectorfield.`

The published `foldflow-sfm.pth` asset contains `checkpoint["model"]` but omits
`checkpoint["conf"]`. This integration therefore reads the exact upstream
`runner/config/model/benchmark.yaml` and
`runner/config/flow_matcher/default.yaml` files from the pinned revision.
Upstream does not publish release-asset checksums; the downloader verifies the
known SHA-256 computed for the selected SFM asset.

## Selected Backbone

This phase integrates official **FoldFlow-SFM** from release `0.1.0`.

FoldFlow-2 is not selected because its pretrained architecture runs an internal
frozen ESM model and consumes both all-layer residue representations and
attention-derived pair representations. The existing cache contains final
Hugging Face ESM2 residue embeddings and pooled embeddings, not FoldFlow-2's
checkpoint-specific internal ESM tensors. Replacing those internals would make
the official FoldFlow-2 checkpoint incompatible.

FoldFlow-1 is the correct official structural backbone for adding an external,
trainable conditioning adapter while reusing the existing ESM cache.

## Model Inputs

The official FoldFlow-1 `VectorFieldNetwork.forward` accepts a feature
dictionary containing:

- `rigids_t`: `[B, N, 7]` quaternion and translation frames
- `res_mask`: `[B, N]`
- `fixed_mask`: `[B, N]`
- `seq_idx`: `[B, N]`
- `t`: `[B]`
- `sc_ca_t`: `[B, N, 3]`
- `torsion_angles_sin_cos`: `[B, N, 7, 2]`

The translation dataset must construct source and target backbone frames from
the KLIFS PDB files using upstream FoldFlow/OpenFold parsing and frame
transforms.

## Model Outputs

The official FoldFlow-1 model returns:

- `rot_vectorfield`: `[B, N, 3, 3]`
- `trans_vectorfield`: `[B, N, 3]`
- `rigids`: `[B, N, 7]`
- `psi`: `[B, N, 2]`
- `atom37`: `[B, N, 37, 3]`
- `atom14`: `[B, N, 14, 3]`

Official reverse sampling uses `SE3FlowMatcher.reverse` over these vector
fields.

## Conditioning Attachment

The checkpoint-compatible attachment point is the output of the official
FoldFlow-1 `Embedder`, immediately before the official IPA vector-field
network. A local adapter projects:

- cached residue-level ESM2 embeddings
- trainable kinase identity embeddings
- trainable target-state embeddings

to the official node width (`256` for released FoldFlow-1 checkpoints). The
adapter output is added to the official node embedding. No official parameter
shape or state-dict key is changed.

Source structure remains visible through `rigids_t` in both CFG branches.
Protein identity remains visible through ESM and kinase conditioning in both
branches. Only the target-state contribution is removed in the unconditional
branch.

## CFG Contract

For vector-field keys, inference combines official FoldFlow outputs as:

`unconditional + guidance_scale * (conditional - unconditional)`

Conditional:

- source structure
- ESM condition
- kinase condition
- target-state condition

Unconditional:

- source structure
- ESM condition
- kinase condition
- zero target-state condition

Training-time CFG dropout applies only to the target-state condition.

## Migration Steps

1. Install the pinned official source as an external dependency.
2. Download and verify the official FoldFlow-SFM release checkpoint.
3. Load the official checkpoint and instantiate its exact architecture/config.
4. Add external conditioning encoders and a node-conditioning adapter.
5. Build inactive-to-active pairs exclusively within existing LOKO split CSVs.
6. Parse structures through upstream FoldFlow/OpenFold utilities.
7. Validate checkpoint loading, parameter counts, conditioning dimensions,
   CFG algebra, and kinase-level fold integrity.

## Primary Sources

- Official repository: <https://github.com/DreamFold/FoldFlow>
- Official releases: <https://github.com/DreamFold/FoldFlow/releases>
- FoldFlow paper: <https://arxiv.org/abs/2310.02391>
- FoldFlow-2 paper: <https://arxiv.org/abs/2405.20313>
