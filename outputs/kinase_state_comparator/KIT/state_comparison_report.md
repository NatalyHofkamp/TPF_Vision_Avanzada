# KIT Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `8pqd`, resolution 1.50 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `6gql`, resolution 2.01 A, ligand=1, DFG=out, alphaC=in.
- Alignment coverage: 0.990; sequence identity: 1.000.
- Compared residues: 295.

## Global comparison

- Global C-alpha RMSD: **5.248 A**.
- Distance-matrix error: **2.998 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 6.285 | 4.836 | 13.056 | 1.216 | 0.677 |
| alphaC | 15.000 | 1.708 | 1.702 | 1.935 | 0.224 | 0.996 |
| activation_loop | 25.000 | 16.277 | 12.748 | 32.876 | 5.426 | 0.103 |
| HRD | 7.000 | 1.737 | 1.730 | 1.985 | 0.143 | 0.998 |
| Lys-Glu | 2.000 | 1.404 | 1.401 | 1.489 | 0.167 | 0.998 |

## Direct answers

- DFG change: 6.285 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 1.708 A local RMSD.
- Activation-loop change: 16.277 A local RMSD.
- Lys-Glu: active=2.724 A, inactive=2.702 A, difference=+0.022 A (NZ-OE side-chain).
- Largest local change: **activation_loop**.
- Most stable requested component: **Lys-Glu**.

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
