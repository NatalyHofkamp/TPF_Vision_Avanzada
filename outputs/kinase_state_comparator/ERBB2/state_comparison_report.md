# ERBB2 Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `3pp0`, resolution 2.25 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `3rcd`, resolution 3.21 A, ligand=1, DFG=in, alphaC=out.
- Alignment coverage: 0.913; sequence identity: 1.000.
- Compared residues: 261.

## Global comparison

- Global C-alpha RMSD: **1.996 A**.
- Distance-matrix error: **1.547 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 5.000 | 2.197 | 1.862 | 3.996 | 0.168 | 0.939 |
| alphaC | 15.000 | 2.284 | 2.137 | 3.473 | 1.172 | 0.985 |
| activation_loop | 9.000 | 2.045 | 1.636 | 3.996 | 0.982 | 0.989 |
| HRD | 7.000 | 0.672 | 0.654 | 0.864 | 0.447 | 0.999 |
| Lys-Glu | 2.000 | 1.658 | 1.620 | 1.970 | 1.412 | 0.999 |

## Direct answers

- DFG change: 2.197 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 2.284 A local RMSD.
- Activation-loop change: 2.045 A local RMSD.
- Lys-Glu: active=6.735 A, inactive=12.026 A, difference=-5.291 A (NZ-OE side-chain).
- Largest local change: **alphaC**.
- Most stable requested component: **HRD**.

## Model training

- Real structures used: 863.
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
