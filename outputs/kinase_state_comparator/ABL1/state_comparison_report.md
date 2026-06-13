# ABL1 Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `2f4j`, resolution 1.91 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `2g1t`, resolution 1.80 A, ligand=0, DFG=in, alphaC=out.
- Alignment coverage: 0.958; sequence identity: 1.000.
- Compared residues: 275.

## Global comparison

- Global C-alpha RMSD: **4.402 A**.
- Distance-matrix error: **3.010 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 3.643 | 3.289 | 5.124 | 1.314 | 0.987 |
| alphaC | 15.000 | 6.702 | 6.143 | 10.148 | 0.374 | 0.872 |
| activation_loop | 24.000 | 10.712 | 7.948 | 22.779 | 3.704 | 0.361 |
| HRD | 7.000 | 1.281 | 1.230 | 2.024 | 0.208 | 0.998 |
| Lys-Glu | 2.000 | 4.966 | 4.086 | 6.909 | 4.025 | 0.948 |

## Direct answers

- DFG change: 3.643 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 6.702 A local RMSD.
- Activation-loop change: 10.712 A local RMSD.
- Lys-Glu: active=2.582 A, inactive=14.479 A, difference=-11.896 A (NZ-OE side-chain).
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
