# PDGFRA Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `8pqh`, resolution 2.50 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `8pqk`, resolution 2.00 A, ligand=0, DFG=out, alphaC=in.
- Alignment coverage: 0.969; sequence identity: 1.000.
- Compared residues: 312.

## Global comparison

- Global C-alpha RMSD: **6.601 A**.
- Distance-matrix error: **4.841 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 6.549 | 5.327 | 13.309 | 0.766 | 0.582 |
| alphaC | 15.000 | 2.862 | 2.750 | 4.614 | 0.209 | 0.985 |
| activation_loop | 25.000 | 16.008 | 13.137 | 28.004 | 5.814 | 0.094 |
| HRD | 7.000 | 2.117 | 2.110 | 2.336 | 0.119 | 0.993 |
| Lys-Glu | 2.000 | 2.184 | 2.181 | 2.298 | 0.015 | 0.996 |

## Direct answers

- DFG change: 6.549 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 2.862 A local RMSD.
- Activation-loop change: 16.008 A local RMSD.
- Lys-Glu: active=3.246 A, inactive=2.580 A, difference=+0.666 A (NZ-OE side-chain).
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
