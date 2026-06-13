# FGFR1 Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `8jmz`, resolution 1.99 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `4v01`, resolution 2.33 A, ligand=1, DFG=out, alphaC=in.
- Alignment coverage: 0.899; sequence identity: 1.000.
- Compared residues: 275.

## Global comparison

- Global C-alpha RMSD: **1.169 A**.
- Distance-matrix error: **0.833 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 4.000 | 4.283 | 2.575 | 8.451 | 1.338 | 0.728 |
| alphaC | 15.000 | 0.500 | 0.462 | 0.920 | 0.091 | 0.998 |
| activation_loop | 17.000 | 2.921 | 2.170 | 8.451 | 1.414 | 0.679 |
| HRD | 7.000 | 0.251 | 0.227 | 0.471 | 0.119 | 1.000 |
| Lys-Glu | 2.000 | 0.460 | 0.457 | 0.502 | 0.051 | 0.999 |

## Direct answers

- DFG change: 4.283 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 0.500 A local RMSD.
- Activation-loop change: 2.921 A local RMSD.
- Lys-Glu: active=2.836 A, inactive=2.807 A, difference=+0.029 A (NZ-OE side-chain).
- Largest local change: **DFG**.
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
