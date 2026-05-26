# Troubleshooting & Common Issues

**Solutions to common problems in the KLIFS pipeline.**

---

## Installation Issues

### `ModuleNotFoundError: No module named 'requests'`

```bash
pip install -r requirements.txt
```

### `ImportError: cannot import name 'PDBParser'`

```bash
pip install biopython
```

### `ImportError: No module named 'torch'`

```bash
pip install torch
```

---

## KLIFS API Issues

### Connection Error: "Cannot reach KLIFS API"

**Symptoms:**
```
requests.exceptions.ConnectionError
requests.exceptions.Timeout
```

**Solutions:**

```bash
# 1. Check KLIFS is online
curl https://klifs.net

# 2. Verify your internet connection
ping google.com

# 3. Try again later (API might be down)

# 4. If behind proxy/firewall
# - Configure requests to use your proxy
# - Contact your sysadmin
```

### Rate Limiting: "429 Too Many Requests"

The pipeline already has delays (`rate_limit_delay: 0.1` in config.yaml).

**If still getting rate-limited:**

```yaml
# In config.yaml
download:
  rate_limit_delay: 0.5  # Increase from 0.1 to 0.5
  max_retries: 5         # More retries
  retry_delay: 2.0       # Wait longer between retries
```

---

## Dataset Download Issues

### Error: "No kinases found"

**Cause**: KLIFS API structure might have changed or no kinases match filter.

**Solution**:
```bash
# Check the API response manually
curl https://klifs.net/api/kinases_list

# Or test the download with verbose logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from scripts.download_klifs_dataset import KLIFSDownloader
downloader = KLIFSDownloader()
downloader.download_all()
"
```

### Download Interrupted (Incomplete PDB Files)

**Solution**:

```bash
# 1. Check how many PDBs were downloaded
ls data/raw/pdbs/ | wc -l

# 2. Re-run the download
# The script skips already-downloaded files, so it will resume
python scripts/download_klifs_dataset.py

# 3. Or start fresh (if many are corrupted)
rm -rf data/raw/pdbs/
python scripts/download_klifs_dataset.py
```

### Disk Space Issues

The full dataset requires ~2-3 GB for PDB files.

**Solutions**:

```bash
# Option 1: Download only specific kinases
# Edit config.yaml and include only kinases you need

# Option 2: Remove PDBs after preprocessing
# (can re-download later if needed)
rm -rf data/raw/pdbs/

# Option 3: Check disk space
df -h  # Linux/Mac
dir   # Windows
```

---

## Preprocessing Issues

### Out of Memory During Preprocessing

**Error:**
```
CUDA out of memory. Tried to allocate...
MemoryError: Unable to allocate...
```

**Solutions**:

```bash
# Option 1: Reduce batch size
# Edit config.yaml
preprocessing:
  batch_size: 8  # Instead of 32
```

```bash
# Option 2: Use CPU instead of GPU
export CUDA_VISIBLE_DEVICES=""
python scripts/preprocess_dataset.py
```

```bash
# Option 3: Process subset first
python -c "
import pandas as pd
df = pd.read_csv('data/metadata/kinase_labels.csv')
df.head(100).to_csv('data/metadata/kinase_labels_small.csv', index=False)
"
python scripts/preprocess_dataset.py  # Will read the small one
```

### "PDB file not found" During Preprocessing

**Cause**: Download didn't complete or file got deleted.

**Solution**:

```bash
# Verify download completed
ls data/raw/pdbs/ | wc -l
# Should be ~500

# If fewer, re-download
python scripts/download_klifs_dataset.py
```

### Missing Tensors After Preprocessing

**Check**:

```bash
ls data/processed/ | wc -l
# Should be ~500 (one directory per PDB)

# If fewer, rerun preprocessing
python scripts/preprocess_dataset.py
```

---

## Split Issues

### Leakage Validation Failed

**Error:**
```
ValueError: Leakage between TRAIN and VAL: {'EGFR'}
```

**This means**: The new grouped split strategy detected that you're using an OLD split CSV with overlapping kinases.

**Solution**:

```bash
# Regenerate splits with grouped strategy
rm data/splits/*.csv
python scripts/download_klifs_dataset.py
```

### Imbalanced ACTIVE/INACTIVE Ratio

With grouped splits, balance might not be perfect.

**Example**:
```
Train: 70% active, 30% inactive
Test:  50% active, 50% inactive  ← Different!
```

**This is OK.** Solution in training:

```python
# Use weighted loss
from torch.nn import BCEWithLogitsLoss

# Calculate weights
n_active = 343
n_inactive = 180
pos_weight = n_inactive / n_active

loss_fn = BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
```

---

## Training Issues

### Test Accuracy Much Lower Than Train Accuracy

**Example**:
```
Train Accuracy: 92%
Val Accuracy:   88%
Test Accuracy:  62%  ← Much lower!
```

**Is this bad?** No, this is EXPECTED and CORRECT.

**Why?**
- Train: Sees kinases EGFR, BRAF, ABL1, ...
- Test: Sees kinases PDGFRA, FGFR1, ALK, ... (completely new)
- Difference is real: model generalizes 62% to new kinases

**This is NOT a bug**, it's **honest measurement**.

### Model Not Converging

**Symptoms**: Loss stays high, doesn't decrease

**Solutions**:

```python
# 1. Check learning rate (might be too high)
learning_rate = 1e-5  # Try smaller value

# 2. Check batch size (might be too large)
batch_size = 16  # Try smaller

# 3. Check for data issues
# Run: python scripts/explore_dataset.py

# 4. Add regularization
model = YourModel(dropout=0.5)  # Higher dropout

# 5. Reduce dataset to test
# Use data/splits/train_small.csv for quick debugging
```

---

## Data Validation

### Validate Dataset Integrity

```python
from pathlib import Path
import pandas as pd

def validate_dataset():
    """Check everything is OK."""
    
    # 1. Metadata exists
    metadata_path = Path('data/metadata/kinase_labels.csv')
    assert metadata_path.exists(), "No metadata CSV"
    
    df = pd.read_csv(metadata_path)
    assert len(df) > 0, "Metadata is empty"
    assert not df.isnull().any().any(), "Metadata has NULL values"
    
    # 2. PDB files exist
    for idx, row in df.iterrows():
        path = Path(row['filepath'])
        assert path.exists(), f"PDB not found: {path}"
    
    # 3. Splits exist
    for split in ['train', 'val', 'test']:
        split_path = Path(f'data/splits/{split}.csv')
        assert split_path.exists(), f"Split not found: {split}"
        
        split_df = pd.read_csv(split_path)
        assert len(split_df) > 0, f"Split is empty: {split}"
        
        # Check no leakage
        train_kinases = set(pd.read_csv('data/splits/train.csv')['kinase_name'].unique())
        test_kinases = set(pd.read_csv('data/splits/test.csv')['kinase_name'].unique())
        overlap = train_kinases & test_kinases
        assert not overlap, f"Leakage detected: {overlap}"
    
    # 4. Tensors exist
    processed_dir = Path('data/processed')
    assert processed_dir.exists(), "No processed directory"
    
    num_tensors = len(list(processed_dir.glob('*/ca_coords.pt')))
    print(f"✅ Found {num_tensors} tensor files")
    
    print("✅ Dataset validation passed!")

# Run it
validate_dataset()
```

---

## Performance Optimization

### Speed Up Preprocessing

```python
# In preprocess_dataset.py, increase batch size (if memory allows)
batch_size = 64  # Instead of 32

# Or use more workers
DataLoader(dataset, num_workers=4, batch_size=32)
```

### Speed Up Training

```python
# Use mixed precision (if GPU supports it)
from torch.cuda.amp import autocast

with autocast():
    loss = model(x)

# Or use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(loader):
    loss = model(batch)
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Monitoring & Debugging

### Enable Verbose Logging

```python
import logging

# Set to DEBUG level
logging.basicConfig(level=logging.DEBUG)

# Now all logs will be printed
python scripts/download_klifs_dataset.py
```

### Save Logs to File

```python
import logging

handler = logging.FileHandler('pipeline.log')
logger.addHandler(handler)

# Logs will be saved in pipeline.log
```

### Monitor Disk Usage

```bash
# Check data folder sizes
du -sh data/raw/pdbs/      # PDB files
du -sh data/processed/     # Tensors
du -sh figures/            # Plots

# If low on space
rm -rf data/processed/     # Can regenerate
```

---

## Recovery Procedures

### If Download Was Interrupted

```bash
# 1. Check what was downloaded
ls data/raw/pdbs/ | wc -l

# 2. Find what's missing
python -c "
import pandas as pd
import os

all_df = pd.read_csv('data/metadata/kinase_labels.csv')
downloaded = set(f.replace('.pdb', '').upper() for f in os.listdir('data/raw/pdbs'))

missing_df = all_df[~all_df['pdb_id'].isin(downloaded)]
print(f'Missing: {len(missing_df)} PDBs')
missing_df.to_csv('data/missing_pdbs.csv', index=False)
"

# 3. Resume download (or start fresh)
python scripts/download_klifs_dataset.py
```

### If Preprocessing Was Interrupted

```bash
# 1. Check what was processed
ls data/processed/ | wc -l

# 2. Modify preprocessing to skip completed
# preprocess_dataset.py already does this

# 3. Resume
python scripts/preprocess_dataset.py
```

---

## FAQ

**Q: Can I use the pipeline with only a few kinases?**  
A: Yes! Edit `config.yaml` under `kinases:` section.

**Q: How long does download take?**  
A: 30-60 minutes depending on internet speed. KLIFS API is rate-limited.

**Q: Can I download only ACTIVE structures?**  
A: Edit `scripts/download_klifs_dataset.py` and filter by `conformational_state`.

**Q: Do I need GPU?**  
A: No for download/preprocess. Yes for training (but can use CPU, slower).

**Q: Can I use a different dataset instead of KLIFS?**  
A: Yes, adapt `scripts/download_klifs_dataset.py` to your data format.

---

## Getting Help

1. **Check this file first** - you're probably looking for the solution here
2. **Run validation**: `python scripts/validate_setup.py`
3. **Check logs**: Look at terminal output and any `.log` files
4. **Search KLIFS status**: https://klifs.net might have issues
5. **Try small dataset first**: Test with 1-2 kinases before full run

---

**Last updated**: 2026-05-25
