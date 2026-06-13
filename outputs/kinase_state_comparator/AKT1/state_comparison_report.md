# AKT1 Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `4gv1`, resolution 1.49 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `7nh5`, resolution 1.90 A, ligand=1, DFG=out, alphaC=na.
- Alignment coverage: 0.723; sequence identity: 1.000.
- Compared residues: 277.

## Global comparison

- Global C-alpha RMSD: **4.680 A**.
- Distance-matrix error: **2.688 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 5.752 | 4.698 | 10.421 | 1.959 | 0.827 |
| alphaC | 10.000 | 7.520 | 6.797 | 11.786 | 0.404 | 0.859 |
| activation_loop | 19.000 | 13.983 | 9.861 | 34.067 | 4.363 | 0.703 |
| Lys-Glu | 2.000 | 7.737 | 7.015 | 10.278 | 0.230 | 0.888 |

## Direct answers

- DFG change: 5.752 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 7.520 A local RMSD.
- Activation-loop change: 13.983 A local RMSD.
- Lys-Glu: active=32.049 A, inactive=28.426 A, difference=+3.623 A (NZ-OE side-chain).
- Largest local change: **activation_loop**.
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
