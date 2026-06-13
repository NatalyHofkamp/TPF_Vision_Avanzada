# CDK6 Active/Inactive Structural Comparison

This is a comparison of static experimental endpoints. It is not a temporal
simulation and does not establish a biological activation mechanism.

## Selected structures

- Active: `2euf`, resolution 3.00 A, ligand=1, DFG=in, alphaC=in.
- Inactive: `1blx`, resolution 1.90 A, ligand=0, DFG=in, alphaC=out.
- Alignment coverage: 0.925; sequence identity: 1.000.
- Compared residues: 282.

## Global comparison

- Global C-alpha RMSD: **5.714 A**.
- Distance-matrix error: **3.156 A**.

## Functional components

| component | residue_count | local_rmsd | mean_displacement | max_displacement | internal_distance_change | orientation_axis_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| DFG | 7.000 | 3.279 | 2.634 | 5.823 | 1.838 | 0.984 |
| alphaC | 15.000 | 5.289 | 4.340 | 10.125 | 0.843 | 0.863 |
| activation_loop | 25.000 | 15.999 | 12.398 | 29.990 | 3.942 | 0.045 |
| HRD | 7.000 | 1.148 | 1.129 | 1.480 | 0.112 | 0.996 |
| Lys-Glu | 2.000 | 3.354 | 3.089 | 4.394 | 1.343 | 0.985 |

## Direct answers

- DFG change: 3.279 A local RMSD. This quantifies endpoint displacement, not a verified orientation mechanism.
- alphaC change: 5.289 A local RMSD.
- Activation-loop change: 15.999 A local RMSD.
- Lys-Glu: active=2.949 A, inactive=12.131 A, difference=-9.182 A (NZ-OE side-chain).
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
