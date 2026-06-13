# EGFR Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `8d73`, resolution 2.17 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `5hg8`, resolution 1.42 A, ligand=1, DFG=out, alphaC=out.
- Alignment coverage: 0.961; sequence identity: 1.000.
- Compared residues: 297.

## Global comparison

- Global C-alpha RMSD: **3.303 A**.
- Distance-matrix error: **2.630 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 2.446 | 1.996 | 5.420 | 0.841 | 0.993 |
| alphaC | 15.000 | 7.534 | 6.565 | 12.455 | 0.162 | 0.830 |
| activation_loop | 25.000 | 2.266 | 2.061 | 5.420 | 0.754 | 0.994 |
| HRD | 7.000 | 1.239 | 1.232 | 1.477 | 0.139 | 0.997 |
| Lys-Glu | 2.000 | 5.329 | 4.484 | 7.364 | 2.920 | 0.906 |

## Direct answers

- DFG change: 2.446 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 7.534 A local RMSD.
- Activation-loop change: 2.266 A local RMSD.
- Lys-Glu: active=2.900 A, inactive=11.185 A, difference=-8.286 A (NZ-OE side-chain).
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
