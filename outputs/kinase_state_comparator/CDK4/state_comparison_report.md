# CDK4 Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `7sj3`, resolution 2.51 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `3g33`, resolution 3.00 A, ligand=0, DFG=in, alphaC=out.
- Alignment coverage: 0.928; sequence identity: 1.000.
- Compared residues: 270.

## Global comparison

- Global C-alpha RMSD: **4.425 A**.
- Distance-matrix error: **2.820 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 3.134 | 2.606 | 5.492 | 1.215 | 0.988 |
| alphaC | 15.000 | 5.435 | 4.748 | 8.962 | 0.286 | 0.878 |
| activation_loop | 25.000 | 11.885 | 9.109 | 24.650 | 3.684 | 0.548 |
| HRD | 7.000 | 1.081 | 1.035 | 1.503 | 0.169 | 0.997 |
| Lys-Glu | 2.000 | 3.799 | 3.326 | 5.162 | 2.541 | 0.975 |

## Direct answers

- DFG change: 3.134 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 5.435 A local RMSD.
- Activation-loop change: 11.885 A local RMSD.
- Lys-Glu: active=2.962 A, inactive=12.755 A, difference=-9.793 A (NZ-OE side-chain).
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
