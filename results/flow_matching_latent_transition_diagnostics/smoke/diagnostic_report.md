# Automatic transition-learning diagnosis

## Main findings

- Endpoint availability problem: **not dominant**.
  Known inactive endpoints: 413; known active endpoints: 450.
- Pairing problem: **controlled after filtering**.
- Alignment problem: **substantially reduced**.
  Mean selected-pair coverage: 0.953; identity: 1.000.
- Architecture/local supervision: **auxiliary/local supervision improves the same architecture**.
- Intermediate supervision: **not a labeling problem: intermediates are intentionally latent**.
- DFG/alphaC descriptor combinations observed: 5;
  these are analysis descriptors, not supervised intermediate labels.
- No variant learned a target-directed transition in this run. The lowest RMSD was aligned_full_conditioning at 11.800 A, but its improvement remained negative (-2056.56%).
- Among directly comparable variants, aligned_full_conditioning gives the largest RMSD reduction versus index matching: 20.075 A.

## Why the original model failed

1. Normalized-index resampling paired residues that were not guaranteed to be
   homologous, corrupting the velocity target and local geometry.
2. Random same-kinase pairing ignored resolution, ligand status, sequence
   mismatches and missing residues.
3. C-alpha-only tensors removed chain IDs, residue IDs, sequence and side-chain
   atoms needed for motif-aware supervision.
4. Flow loss alone did not explicitly enforce endpoint geometry, local motifs,
   directionality, path smoothness or cycle consistency.
5. The generated intermediate conformations have no experimental ground truth;
   they must be evaluated as latent hypotheses rather than classified as correct
   or incorrect structures.
6. Distance-matrix targets along the path are endpoint-derived geometric
   regularizers, not experimentally observed intermediate geometries.

## Available annotation quality

- DFG motif detection: 96.2%.
- HRD motif detection: 83.1%.
- VAIK Lys detection: 99.1%.
- alphaC Glu proxy detection: 98.6%.
- KLIFS pocket-position mappings are not present locally; sequence alignment is
  used as the best available fallback.
- Explicit mutation annotations are absent. Sequence mismatches are reported as
  possible mutations or construct differences.
- Lys-Glu generated-state distance can only be approximated with C-alpha
  coordinates unless an all-atom model is introduced.

## Recommended next decision

Use `diagnostic_variant_comparison.csv` to select the change with the largest
reduction in endpoint/local RMSD and the best smoothness, directionality,
geometry and cycle metrics. Do not claim that a generated path is the real
activation mechanism without molecular simulation or experimental validation.

## Epistemic status of intermediate conformations

Only t=0 and t=1 are observed structures. States generated at t=0.25, 0.50 and
0.75 are model hypotheses constrained by endpoint data and geometric losses.
They are not labels, experimentally observed intermediates or ground truth.
For learned trajectories, the t=1 row is a generated endpoint prediction
evaluated against the observed target; it is not itself an observed structure.
Success means coherent geometry, stable motifs, monotonic progress and endpoint
recovery; it does not establish the biological mechanism.
