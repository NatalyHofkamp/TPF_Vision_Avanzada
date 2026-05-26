# Scripts - Technical Reference

**For comprehensive documentation, see [API_REFERENCE.md](../docs/API_REFERENCE.md)**

This directory contains the main pipeline scripts.

---

## Quick Reference

| Script | Purpose | Time |
|--------|---------|------|
| `download_klifs_dataset.py` | Download kinases from KLIFS API | 30-60 min |
| `explore_dataset.py` | Analyze dataset statistics | 2 min |
| `preprocess_dataset.py` | Convert PDB to PyTorch tensors | 15-30 min |
| `validate_setup.py` | Check dependencies | 1 min |

---

## Running the Pipeline

```bash
# Validate setup
python scripts/validate_setup.py

# Download dataset
python scripts/download_klifs_dataset.py

# Explore statistics
python scripts/explore_dataset.py

# Preprocess to tensors
python scripts/preprocess_dataset.py

# Or use Makefile
make pipeline
```

---

## Documentation

- **Main guide**: See [../../README.md](../README.md) for quick start
- **Detailed API**: See [docs/API_REFERENCE.md](../docs/API_REFERENCE.md) for all methods and parameters
- **Troubleshooting**: See [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) for errors
- **Data leakage**: See [docs/LEAKAGE_AND_SPLITS.md](../docs/LEAKAGE_AND_SPLITS.md) for methodology

---

## Key Concepts

### DFG-state & alphaC-state

Kinases are classified as active/inactive based on two structural markers:

- **DFG-state**: Asp-Phe-Gly loop position
  - "DFG-in" → Active site prepared
  - "DFG-out" → Active site blocked

- **alphaC-state**: Alpha-helix position
  - "in" / "alphaC-in" → Properly positioned
  - "out" / "alphaC-out" → Repressed

When both are "in" → **ACTIVE**  
Otherwise → **INACTIVE**

### Grouped Splits

This pipeline uses **grouped stratification**:
- All structures of one kinase go to ONE split (train/val/test)
- Prevents leakage and simulates real generalization
- See [docs/LEAKAGE_AND_SPLITS.md](../docs/LEAKAGE_AND_SPLITS.md) for details

---

## Output Structure

```
data/
├── metadata/
│   └── kinase_labels.csv          # All structures + labels
├── raw/pdbs/
│   └── *.pdb                       # Downloaded structures (~500)
├── splits/
│   ├── train.csv                  # 70% structures (grouped by kinase)
│   ├── val.csv                    # 15% structures (different kinases)
│   └── test.csv                   # 15% structures (new kinases)
└── processed/
    └── [PDB_ID]/
        ├── ca_coords.pt           # (N, 3) alpha carbon coordinates
        ├── distance_matrix.pt     # (N, N) pairwise distances
        └── metadata.pt            # Label and metadata dict

figures/
└── dataset_analysis.png           # Visualization

notebooks/
└── dataset_visualization.ipynb    # Interactive exploration
```

---

## Common Tasks

### Use only specific kinases

```yaml
# Edit config.yaml
kinases:
  - name: "EGFR"
  - name: "BRAF"
  # Remove others
```

### Reduce memory usage

```yaml
# Edit config.yaml
preprocessing:
  batch_size: 8  # Instead of 32
```

### Check data integrity

```bash
python scripts/validate_setup.py
```

### View statistics

```bash
python scripts/explore_dataset.py
```

### Debug download

```bash
python scripts/download_klifs_dataset.py 2>&1 | head -100
```

---

## Troubleshooting

**See [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)** for:
- Import errors
- KLIFS API errors
- Download issues
- Memory errors
- Recovery procedures

---

**Last updated**: 2026-05-25

