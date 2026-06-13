# ALK Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `3lcs`, resolution 1.95 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `5iug`, resolution 1.93 A, ligand=1, DFG=out, alphaC=in.
- Alignment coverage: 0.911; sequence identity: 0.996.
- Compared residues: 277.

## Global comparison

- Global C-alpha RMSD: **1.513 A**.
- Distance-matrix error: **0.919 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 5.001 | 3.931 | 8.578 | 1.006 | 0.811 |
| alphaC | 15.000 | 1.284 | 1.253 | 1.896 | 0.171 | 0.999 |
| activation_loop | 13.000 | 3.680 | 2.316 | 8.578 | 1.929 | 0.947 |
| HRD | 7.000 | 0.423 | 0.399 | 0.620 | 0.044 | 1.000 |
| Lys-Glu | 2.000 | 0.944 | 0.931 | 1.086 | 0.032 | 0.993 |

## Direct answers

- DFG change: 5.001 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 1.284 A local RMSD.
- Activation-loop change: 3.680 A local RMSD.
- Lys-Glu: active=2.661 A, inactive=3.074 A, difference=-0.413 A (NZ-OE side-chain).
- Largest local change: **DFG**.
- Most stable requested component: **HRD**.

## Model training

- R

- Training/validation: 690 / 173.
- Epochs executed: 64; best epoch: 44.
- Best validation accuracy: 0.942.
- Best validation balanced accuracy: 0.942.
- Best validation ROC AUC: 0.969.

The model contributes a state-consistency score to representative selection.
All reported structural differences are computed directly from the selected
real PDB endpoints after sequence alignment and rigid-body superposition.

## Files

- [Active PDB](active_selected.pdb)
- [Inactive PDB](inactive_selected.pdb)
- [Aligned two-model PDB](active_inactive_aligned.pdb)
- [PyMOL script](view_active_inactive.pml)
- [ChimeraX script](view_active_inactive.cxc)
- [Figures](figures/)

## Mapping limitations

No explicit KLIFS-position table is present in the current repository.
Components were mapped from sequence motifs and residue identifiers: DFG and
HRD by exact motif, alphaC from the conserved Glu upstream of VAIK, activation
loop from DFG onward, and Lys-Glu from the detected VAIK Lys/alphaC Glu pair.
Missing components are reported rather than imputed as biological truth.
