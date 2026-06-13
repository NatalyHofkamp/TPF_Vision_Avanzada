# MET Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `3q6w`, resolution 1.75 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `8ouv`, resolution 1.78 A, ligand=1, DFG=out, alphaC=out.
- Alignment coverage: 0.970; sequence identity: 0.997.
- Compared residues: 288.

## Global comparison

- Global C-alpha RMSD: **2.964 A**.
- Distance-matrix error: **1.923 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 5.125 | 4.202 | 8.489 | 1.469 | 0.855 |
| alphaC | 15.000 | 0.575 | 0.560 | 0.856 | 0.051 | 0.999 |
| activation_loop | 25.000 | 7.894 | 6.229 | 16.644 | 3.502 | 0.586 |
| HRD | 7.000 | 0.449 | 0.413 | 0.776 | 0.181 | 0.999 |
| Lys-Glu | 2.000 | 0.728 | 0.710 | 0.874 | 0.087 | 0.999 |

## Direct answers

- DFG change: 5.125 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 0.575 A local RMSD.
- Activation-loop change: 7.894 A local RMSD.
- Lys-Glu: active=11.560 A, inactive=11.433 A, difference=+0.127 A (NZ-OE side-chain).
- Largest local change: **activation_loop**.
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
