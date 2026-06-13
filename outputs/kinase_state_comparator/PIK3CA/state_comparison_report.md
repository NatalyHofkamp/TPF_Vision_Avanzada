# PIK3CA Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `4jps`, resolution 2.20 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `7myn`, resolution 2.79 A, ligand=0, DFG=out-like, alphaC=in.
- Alignment coverage: 0.968; sequence identity: 1.000.
- Compared residues: 967.

## Global comparison

- Global C-alpha RMSD: **1.290 A**.
- Distance-matrix error: **0.915 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 0.629 | 0.606 | 0.983 | 0.318 | 0.998 |
| alphaC | 15.000 | 2.959 | 2.865 | 3.670 | 0.327 | 0.996 |
| activation_loop | 16.000 | 1.057 | 0.970 | 1.950 | 0.599 | 0.991 |
| Lys-Glu | 2.000 | 2.939 | 2.920 | 3.253 | 0.097 | 0.973 |

## Direct answers

- DFG change: 0.629 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 2.959 A local RMSD.
- Activation-loop change: 1.057 A local RMSD.
- Lys-Glu: active=5.798 A, inactive=8.972 A, difference=-3.174 A (NZ-OE side-chain).
- Largest local change: **alphaC**.
- Most stable requested component: **DFG**.

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
