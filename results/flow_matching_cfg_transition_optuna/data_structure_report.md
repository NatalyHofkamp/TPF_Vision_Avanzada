# Data structure report

- Project root: `/Users/josefinadehan/tp_vision_avanzada/TPF_Vision_Avanzada`
- Metadata: `/Users/josefinadehan/tp_vision_avanzada/TPF_Vision_Avanzada/data/metadata/kinase_labels.csv` (1777 rows)
- Processed tensor directories: 863
- Valid tensor sets: 863
- Invalid tensor sets: 0
- Missing metadata.pt: 0
- C-alpha lengths: {'min': 209, 'median': 292.0, 'max': 1205, 'unique_lengths': 192, 'most_common': [(300, 21), (292, 21), (297, 20), (287, 19), (280, 19), (291, 19), (288, 19), (293, 18), (264, 17), (299, 16)]}

## Splits

- train: 1392 rows, 615 unique PDBs, 777 duplicate rows, 9 kinases, states={'inactive': 722, 'active': 670}
- val: 234 rows, 136 unique PDBs, 98 duplicate rows, 2 kinases, states={'inactive': 207, 'active': 27}
- test: 151 rows, 112 unique PDBs, 39 duplicate rows, 2 kinases, states={'active': 135, 'inactive': 16}

## Leakage checks

```json
{
  "train_val": {
    "pdb_overlap": [],
    "kinase_overlap": []
  },
  "train_test": {
    "pdb_overlap": [],
    "kinase_overlap": []
  },
  "val_test": {
    "pdb_overlap": [],
    "kinase_overlap": []
  }
}
```

Variable lengths are handled by deterministic index-space linear resampling. The active structure is centered and Kabsch-aligned to the inactive structure. This is an approximation because the processed tensors do not contain residue IDs.