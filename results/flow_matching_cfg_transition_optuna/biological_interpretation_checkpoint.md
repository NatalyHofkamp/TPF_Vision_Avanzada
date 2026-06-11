# Automatic biological interpretation: checkpoint

- The learned endpoint is closer to the real active state than the inactive
  starting structure in 25.0% of pairs.
- It is closer to the active than to the inactive state in 0.0% of pairs.
- Mean relative RMSD improvement is -0.4%.
- Best kinases by generated-to-active RMSD: PDGFRA.
- Worst kinases by generated-to-active RMSD: PDGFRA.

## Conclusion

The transition is not yet consistently active-directed under these structural metrics.
Intermediate states are geometrically continuous, but biological plausibility
cannot be established from C-alpha RMSD and distance matrices alone.

## Limitations and required validation

- Residues are matched by normalized chain index because residue identifiers and
  sequence alignments are absent from the processed tensors.
- Pairing crystal structures does not provide experimentally observed transition paths;
  intermediate-state supervision is unavailable.
- Ligand, mutation, missing-residue, crystallographic and protonation effects are not modeled.
- The linear baseline is an oracle geometric reference because it uses the real active endpoint.
- Validate Ramachandran geometry after all-atom reconstruction, steric clashes,
  conserved kinase motifs (DFG, HRD and alphaC-Glu/Lys), molecular-dynamics
  stability, free-energy profiles and agreement with experimental observables.

This approach can prioritize hypotheses about cancer-kinase activation, but it
does not by itself demonstrate a biological mechanism or a physically populated
transition pathway.
