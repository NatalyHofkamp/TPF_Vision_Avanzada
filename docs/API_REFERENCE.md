# API Reference - Script Details

**Technical documentation for all scripts in the pipeline.**

---

## Scripts Overview

| Script | Purpose | Time | Input | Output |
|--------|---------|------|-------|--------|
| `download_klifs_dataset.py` | Download kinases from KLIFS API | 30-60 min | Internet connection | PDB files, metadata CSV, splits |
| `explore_dataset.py` | Analyze dataset statistics | 2 min | Metadata CSV | Statistics, plots |
| `preprocess_dataset.py` | PDB → PyTorch tensors | 15-30 min | PDB files, metadata | Tensor files |
| `validate_setup.py` | Check dependencies | 1 min | Environment | Validation report |

---

## download_klifs_dataset.py

**Orchestrates the entire dataset acquisition.**

### Class: KLIFSDownloader

```python
from scripts.download_klifs_dataset import KLIFSDownloader

downloader = KLIFSDownloader()
metadata_df = downloader.download_all()
downloader.create_train_val_test_splits(metadata_df)
```

### Key Methods

#### `download_all() → pd.DataFrame`

Downloads all structures for configured kinases.

```python
metadata_df = downloader.download_all()
# Returns DataFrame with all downloaded structures
```

**Outputs:**
- `data/raw/pdbs/*.pdb` - Crystal structure files
- `data/metadata/kinase_labels.csv` - Metadata CSV

**DataFrame columns:**
- `pdb_id`: Protein Data Bank ID
- `kinase_name`: Kinase name (EGFR, BRAF, etc.)
- `kinase_family`: Type (Tyrosine, Serine/Threonine, etc.)
- `species`: Organism (Homo sapiens, etc.)
- `conformational_state`: 'active' or 'inactive'
- `dfg_state`: 'DFG-in' or 'DFG-out'
- `alphac_state`: 'alphaC-in' or 'alphaC-out'
- `ligand_present`: 1 if small molecule bound, 0 otherwise
- `resolution`: Crystal resolution in Angstroms
- `filepath`: Path to PDB file

#### `classify_conformation(structure) → str`

Classifies structure based on DFG and alphaC states.

```python
state = downloader.classify_conformation(structure_dict)
# Returns: 'active' or 'inactive'
```

**Logic:**
```
IF (dfg_state IN ['DFG-in'] AND alphac_state IN ['in', 'alphaC-in']):
    RETURN 'active'
ELSE:
    RETURN 'inactive'
```

#### `create_train_val_test_splits(df, train_size=0.7, val_size=0.15, test_size=0.15, strategy="grouped")`

Splits data into train/val/test.

```python
train_df, val_df, test_df = downloader.create_train_val_test_splits(
    metadata_df,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    strategy="grouped"  # or "stratified"
)
```

**Strategies:**

- `"grouped"`: ✅ **Recommended**
  - Groups all structures of one kinase in one split
  - Prevents leakage
  - Better for generalization

- `"stratified"`: ⚠️ **Legacy**
  - Random split maintaining class balance
  - May cause kinase overlap (leakage)
  - For compatibility only

**Outputs:**
- `data/splits/train.csv` - Training split
- `data/splits/val.csv` - Validation split
- `data/splits/test.csv` - Test split

### Configuration (config.yaml)

```yaml
kinases:
  - name: "EGFR"
  - name: "BRAF"
  # ... more kinases

download:
  max_retries: 3
  retry_delay: 1.0  # seconds
  timeout: 30  # seconds
  rate_limit_delay: 0.1  # seconds between requests

splits:
  strategy: "grouped"  # or "stratified"
  group_by: "kinase_name"
  train: 0.70
  validation: 0.15
  test: 0.15
  random_state: 42
```

---

## explore_dataset.py

**Analyzes dataset and generates visualizations.**

### Class: DatasetExplorer

```python
from scripts.explore_dataset import DatasetExplorer

explorer = DatasetExplorer()
explorer.explore()
```

### Key Methods

#### `explore()`

Runs full analysis pipeline.

```python
explorer.explore()
```

**Outputs:**
- Terminal statistics
- `figures/dataset_analysis.png` - Visualization

**Statistics Shown:**
- Total structures and kinases
- Active/inactive distribution
- Top kinases
- Resolution statistics
- DFG and alphaC state distribution
- Split analysis (if available)

#### `print_statistics()`

Prints basic dataset statistics.

```python
explorer.print_statistics()
```

**Example output:**
```
===================
ESTADÍSTICAS BÁSICAS
===================
Total estructuras: 523
Activas: 343 (65.6%)
Inactivas: 180 (34.4%)
Kinasas únicas: 13
Resolución media: 2.15 Å
Ligandos: 512 (98.0%)
```

#### `plot_distributions()`

Creates visualization plots.

```python
explorer.plot_distributions()
```

**Generates:**
- Histograms of active/inactive
- Kinase distribution
- Resolution distribution
- DFG/alphaC state combinations

---

## preprocess_dataset.py

**Converts PDB structures to PyTorch tensors.**

### Class: ProteinPreprocessor

```python
from scripts.preprocess_dataset import ProteinPreprocessor

preprocessor = ProteinPreprocessor()
items = preprocessor.run(save_tensors=True)
```

### Key Methods

#### `extract_ca_coordinates(pdb_file) → np.ndarray`

Extracts alpha-carbon coordinates.

```python
coords = preprocessor.extract_ca_coordinates(Path('data/raw/pdbs/1abb.pdb'))
# Returns: (N, 3) array of Cα coordinates
```

**Cα (alpha-carbon):**
- Central atom of each amino acid
- Defines protein backbone 3D structure
- 1 per residue

#### `compute_distance_matrix(coords) → np.ndarray`

Computes pairwise distances between Cα atoms.

```python
distances = preprocessor.compute_distance_matrix(coords)
# Returns: (N, N) symmetric distance matrix
```

**Usage:**
- Captures spatial relationships
- Input for many ML models
- Rotation-invariant representation

#### `process_batch(batch_indices) → list`

Processes multiple structures (optimized for GPU).

```python
items = preprocessor.process_batch(range(0, 100))
# Returns: list of processed items with tensors
```

#### `run(save_tensors=True) → list`

Runs full preprocessing pipeline.

```python
all_items = preprocessor.run(save_tensors=True)
```

**Outputs:**
- Saves tensors: `data/processed/[PDB_ID]/*.pt`
- Returns: list of processed structures

**Generated Files:**
- `ca_coords.pt` - (N, 3) float32 tensor
- `distance_matrix.pt` - (N, N) float32 tensor
- `metadata.pt` - dict with label, kinase, etc.

### Configuration (config.yaml)

```yaml
preprocessing:
  extract_ca_only: true  # Use only alpha carbons
  compute_distance_matrix: true
  batch_size: 32
  save_tensors: true
  tensor_format: "pt"  # PyTorch format
```

---

## validate_setup.py

**Checks that all dependencies and directories are configured.**

### Functions

#### `validate_setup()`

Runs all checks.

```python
python scripts/validate_setup.py
```

**Checks:**
- Directories exist (data/, scripts/, etc.)
- Requirements installed
- KLIFS API accessible
- Disk space available

**Example output:**
```
📁 Checking directories...
  ✅ data/
  ✅ scripts/
  ✅ config.yaml

📦 Checking dependencies...
  ✅ requests
  ✅ pandas
  ✅ biopython

🌐 Checking KLIFS API...
  ✅ API is accessible

✅ Setup is valid!
```

---

## Data Formats

### Metadata CSV

**File:** `data/metadata/kinase_labels.csv`

```csv
pdb_id,kinase_name,conformational_state,dfg_state,alphac_state,resolution,filepath
1ABB,EGFR,active,DFG-in,alphaC-in,2.15,data/raw/pdbs/1abb.pdb
1ABC,EGFR,inactive,DFG-out,alphaC-out,2.30,data/raw/pdbs/1abc.pdb
```

**Columns:**
| Column | Type | Example | Note |
|--------|------|---------|------|
| pdb_id | string | "1ABB" | PDB ID |
| kinase_name | string | "EGFR" | Gene name |
| conformational_state | string | "active" | Classification |
| dfg_state | string | "DFG-in" | Asp-Phe-Gly loop |
| alphac_state | string | "alphaC-in" | Alpha-helix position |
| resolution | float | 2.15 | Angstroms |
| filepath | string | "data/raw/pdbs/1abb.pdb" | Local path |

### PyTorch Tensors

**Directory:** `data/processed/[PDB_ID]/`

```
1ABB/
├── ca_coords.pt         # (N, 3) coordinates
├── distance_matrix.pt   # (N, N) distances
└── metadata.pt          # dict metadata
```

**Loading:**

```python
import torch

# Load coordinates
ca_coords = torch.load('data/processed/1ABB/ca_coords.pt')
# shape: (N, 3)

# Load distance matrix
distances = torch.load('data/processed/1ABB/distance_matrix.pt')
# shape: (N, N)

# Load metadata
metadata = torch.load('data/processed/1ABB/metadata.pt')
# dict with keys: label, kinase_name, resolution, sequence_length
```

### Split CSVs

**Files:** `data/splits/{train,val,test}.csv`

Same format as metadata CSV, but filtered rows.

```python
import pandas as pd

train_df = pd.read_csv('data/splits/train.csv')
print(f"Train: {len(train_df)} structures")
print(f"Kinases: {train_df['kinase_name'].nunique()}")
print(f"Active: {(train_df['conformational_state'] == 'active').sum()}")
```

---

## Biological Concepts

### Conformational States

#### ACTIVE (Catalytically Competent)

```
Features:
- DFG-in: Asp-Phe-Gly loop inserted into active site
- alphaC-in: Alpha-helix properly positioned
- Active site ready for catalysis
- Kinase can phosphorylate substrates

Example: EGFR with DFG-in and alphaC-in
```

#### INACTIVE (Non-Catalytic)

```
Features:
- DFG-out: Asp-Phe-Gly loop ejected from active site
- alphaC-out: Alpha-helix repressed/mispositioned
- Active site blocked/distorted
- Kinase cannot efficiently phosphorylate
```

### DFG Loop

- Asp-Phe-Gly tripeptide in kinase catalytic domain
- Critical for catalysis
- "DFG-in": Loop points into active site (active)
- "DFG-out": Loop points away (inactive)

### AlphaC Helix

- Αlpha-helix in kinase domain
- Helps stabilize ATP binding
- "alphaC-in": Properly positioned (active)
- "alphaC-out": Rotated/moved (inactive)

---

## Usage Examples

### Example 1: Full Pipeline

```bash
# Download
python scripts/download_klifs_dataset.py

# Explore
python scripts/explore_dataset.py

# Preprocess
python scripts/preprocess_dataset.py

# Or all at once
make pipeline
```

### Example 2: Custom Kinases

```yaml
# Edit config.yaml
kinases:
  - name: "EGFR"
  - name: "BRAF"
  - name: "ABL1"

# Run download
python scripts/download_klifs_dataset.py
```

### Example 3: Access Preprocessed Data

```python
import torch
import pandas as pd

# Load split
train_df = pd.read_csv('data/splits/train.csv')

# Process one structure
for idx, row in train_df.head(5).iterrows():
    pdb_id = row['pdb_id']
    
    # Load tensors
    coords = torch.load(f'data/processed/{pdb_id}/ca_coords.pt')
    distances = torch.load(f'data/processed/{pdb_id}/distance_matrix.pt')
    metadata = torch.load(f'data/processed/{pdb_id}/metadata.pt')
    
    print(f"{pdb_id}: {row['kinase_name']} - {row['conformational_state']}")
    print(f"  Coordinates shape: {coords.shape}")
    print(f"  Label: {metadata['label']}")
```

### Example 4: Custom Dataset Class

```python
from torch.utils.data import Dataset, DataLoader

class KinaseDataset(Dataset):
    def __init__(self, split_df):
        self.df = split_df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = row['pdb_id']
        
        coords = torch.load(f'data/processed/{pdb_id}/ca_coords.pt')
        metadata = torch.load(f'data/processed/{pdb_id}/metadata.pt')
        
        return {
            'pdb_id': pdb_id,
            'coordinates': coords,
            'label': metadata['label'],
            'kinase': row['kinase_name']
        }

# Create dataloader
train_df = pd.read_csv('data/splits/train.csv')
dataset = KinaseDataset(train_df)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch['coordinates'].shape)  # (32, N, 3)
```

---

## Performance Notes

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Download | 30-60 min | 2-3 GB | Depends on internet, KLIFS load |
| Explore | 2 min | ~500 MB | Fast local analysis |
| Preprocess | 15-30 min | ~2 GB | Can use GPU |
| Load batch | <1s | ~100 MB | PyTorch DataLoader |

---

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for common issues and solutions.

---

**Last updated**: 2026-05-25
