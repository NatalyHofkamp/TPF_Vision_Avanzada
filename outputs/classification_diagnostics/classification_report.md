# Active/Inactive Classification Diagnostics

## Main findings

- Valid real structures: **863**.
- Active/inactive balance among valid structures: **450 / 413**.
- Best within-kinase balanced accuracy: **0.969** using `xgboost` + `local_geometry`.
- Best official OOD-kinase balanced accuracy: **0.846** using `logistic_regression` + `dfg_only`.
- Dedicated Transformer best validation balanced accuracy: **0.871**, but test balanced accuracy is **0.246** and test ROC AUC is **0.184**.
- Labels disagreeing with the strict joint DFG-in/alphaC-in rule: **6**.
- Sampled cross-state neighbors below 1 A aligned C-alpha RMSD: **132**.

## Answers

1. **Does the dataset contain signal?** Yes. Simple models separate states within kinases.
2. **Maximum observed accuracy:** balanced accuracy `0.969` in the within-kinase protocol. Best OOD balanced accuracy is `0.846`.
3. **Best representation:** `local_geometry` within kinases. Rotation-invariant local and distance features outperform raw canonicalized coordinates.
4. **Easy kinases:** AKT1, BRAF, CDK6.
5. **Difficult kinases:** KIT, MET, PIK3CA; several other kinases lack enough minority examples for a stable estimate.
6. **Do DFG/alphaC explain classification?** Yes, especially DFG. The leading features are `hrd_dfg_distance` (0.125), `DFG_std_distance` (0.116), `DFG_radius` (0.079), `DFG_mean_distance` (0.053), `lys_dfg_distance` (0.051), `glu_dfg_distance` (0.048), `mean_distance` (0.040), `radius_of_gyration` (0.040). AlphaC alone is informative but weaker.
7. **Label noise:** active labels are consistent with DFG-in; inactive is a heterogeneous class containing DFG-in/alphaC-out and DFG-out/alphaC-in states. The `6` strict-rule disagreements are mostly missing motif metadata, not clear inversions.
8. **Data or classifier?** Both, but the 0% result is primarily a training/protocol failure. Signal exists, while the original auxiliary head used a reduced paired subset, weak motif anchoring and an OOD split. The data still impose a ceiling because some opposite-label structures are nearly identical in C-alpha geometry.
9. **Classifier as guidance:** not yet as a universal biological energy. A local, rotation-invariant classifier can only guide kinases where it is independently calibrated.
10. **Pretraining recommendation:** pretrain on all real structures, validate both within kinase and on held-out kinases, calibrate probabilities, freeze the model, and reject guidance when balanced accuracy or ROC AUC misses a predefined threshold.

## Separability

PCA silhouette scores are weak: for distance matrices, state silhouette is
`0.055` and kinase
silhouette is `0.122`.
Therefore active/inactive does not form a clean universal unsupervised cluster.
Supervised local features nevertheless recover strong within-kinase signal.

## Root cause of the collapsed auxiliary classifier

1. It was trained inside the generative paired-data path instead of using all 863 valid real structures.
2. Fixed-length sampling does not consistently anchor homologous DFG/alphaC positions, and global pooling weakens local motif information.
3. The official split is kinase-disjoint and severely imbalanced: validation contains KIT/MET, while test contains PDGFRA/PIK3CA.
4. More capacity does not solve the shift. The Transformer learns validation kinases and fails catastrophically on test kinases.
5. C-alpha geometry is sufficient for many kinases, but not universally: `132` sampled cross-state neighbors are below 1 A and PIK3CA is close to chance.

## Representation diagnosis

`ca_coordinates` uses PCA-canonicalized C-alpha coordinates. `distance_matrix`
is rotation invariant. `local_geometry` explicitly measures DFG, alphaC, HRD,
activation loop, and Lys-Glu geometry. The advantage of local/distance features
shows that representation and motif anchoring matter more than raw capacity.

UMAP was attempted, but the installed `umap`/`coverage` combination is
incompatible. PCA and t-SNE results are available.
