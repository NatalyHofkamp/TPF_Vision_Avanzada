# FoldFlow Integration & Conditioning

**How to use the KLIFS dataset with FoldFlow for conditioned protein generation.**

---

## Overview

```
KLIFS Dataset
    ↓
├─ Active structures (DFG-in, alphaC-in)
├─ Inactive structures (DFG-out, alphaC-out)
└─ Structural metadata
    ↓
Preprocessing
    ↓
├─ Cα coordinates (tensors)
├─ Distance matrices
└─ Labels (active/inactive)
    ↓
FoldFlow + Conditioning Module
    ↓
Conditional Protein Generation
```

---

## What is Conditioning?

**Conditioning** = telling the generative model what to generate.

### Example 1: Simple Conditioning

```python
# "Generate EGFR in active state"
condition = {'kinase': 'EGFR', 'state': 'active'}
generated = foldflow.generate(condition=condition)
```

### Example 2: Multi-Criteria Conditioning

```python
# "Generate inactive BRAF with high resolution quality"
condition = {
    'kinase': 'BRAF',
    'conformational_state': 'inactive',
    'dfg_state': 'DFG-out',
    'alphac_state': 'alphaC-out',
    'resolution': 2.0  # Å
}
generated = foldflow.generate(condition=condition)
```

### Example 3: Continuous Conditioning

```python
# "Interpolate between active (1.0) and inactive (0.0)"
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    condition = {'active_probability': t}
    generated = foldflow.generate(condition=condition)
    # See smooth transition from inactive → active
```

---

## Dataset Pipeline for FoldFlow

### Step 1: Prepare Data

```python
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# Load splits (already created by pipeline)
train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')
test_df = pd.read_csv('data/splits/test.csv')

print(f"Train: {len(train_df)} structures")
print(f"Val: {len(val_df)} structures")
print(f"Test: {len(test_df)} structures")
```

### Step 2: Custom Dataset Class

```python
import torch
from pathlib import Path

class ConditionedProteinDataset(Dataset):
    """Dataset for FoldFlow conditioning."""
    
    def __init__(self, split_df, processed_dir='data/processed'):
        self.split_df = split_df
        self.processed_dir = Path(processed_dir)
    
    def __len__(self):
        return len(self.split_df)
    
    def __getitem__(self, idx):
        row = self.split_df.iloc[idx]
        pdb_id = row['pdb_id']
        
        # Load tensors
        tensor_dir = self.processed_dir / pdb_id
        
        ca_coords = torch.load(tensor_dir / 'ca_coords.pt')  # (N, 3)
        distance_matrix = torch.load(tensor_dir / 'distance_matrix.pt')  # (N, N)
        metadata = torch.load(tensor_dir / 'metadata.pt')  # dict
        
        # Create condition
        condition = {
            'kinase_name': row['kinase_name'],
            'conformational_state': row['conformational_state'],  # 'active' or 'inactive'
            'dfg_state': row['dfg_state'],  # 'DFG-in' or 'DFG-out'
            'alphac_state': row['alphac_state'],  # 'alphaC-in' or 'alphaC-out'
            'resolution': float(row['resolution']) if pd.notna(row['resolution']) else 0.0,
        }
        
        return {
            'pdb_id': pdb_id,
            'ca_coords': ca_coords,
            'distance_matrix': distance_matrix,
            'condition': condition,
            'label': 1 if row['conformational_state'] == 'active' else 0
        }

# Create dataloaders
train_dataset = ConditionedProteinDataset(train_df)
val_dataset = ConditionedProteinDataset(val_df)
test_dataset = ConditionedProteinDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

### Step 3: Conditioning Module

```python
import torch.nn as nn

class ProteinConditioning(nn.Module):
    """Converts condition dictionary to embedding."""
    
    def __init__(self, embedding_dim=256, num_kinases=50):
        super().__init__()
        
        # Embeddings for each condition type
        self.kinase_embedding = nn.Embedding(num_kinases, embedding_dim)
        self.state_embedding = nn.Embedding(2, embedding_dim)  # active/inactive
        self.dfg_embedding = nn.Embedding(2, embedding_dim)  # in/out
        self.alphac_embedding = nn.Embedding(2, embedding_dim)  # in/out
        
        # Combine embeddings
        self.combine = nn.Linear(embedding_dim * 4, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, condition_dict, kinase_index_map):
        """
        Args:
            condition_dict: dict with 'kinase_name', 'conformational_state', etc.
            kinase_index_map: dict mapping kinase names to indices
        
        Returns:
            Embedding (embedding_dim,)
        """
        kinase_idx = kinase_index_map[condition_dict['kinase_name']]
        state_idx = 1 if condition_dict['conformational_state'] == 'active' else 0
        dfg_idx = 1 if condition_dict['dfg_state'] == 'DFG-out' else 0
        alphac_idx = 1 if condition_dict['alphac_state'] == 'alphaC-out' else 0
        
        # Get embeddings
        emb_kinase = self.kinase_embedding(torch.tensor(kinase_idx))
        emb_state = self.state_embedding(torch.tensor(state_idx))
        emb_dfg = self.dfg_embedding(torch.tensor(dfg_idx))
        emb_alphac = self.alphac_embedding(torch.tensor(alphac_idx))
        
        # Combine
        combined = torch.cat([emb_kinase, emb_state, emb_dfg, emb_alphac])
        output = self.combine(combined)
        output = self.norm(output)
        
        return output

# Create index map for kinases
kinase_names = sorted(train_df['kinase_name'].unique())
kinase_index_map = {name: idx for idx, name in enumerate(kinase_names)}

# Create conditioning module
conditioning = ProteinConditioning(embedding_dim=256, num_kinases=len(kinase_names))
```

### Step 4: Training Loop with FoldFlow

```python
import torch.optim as optim
from torch.nn import MSELoss

def train_foldflow_conditional(
    foldflow_model,
    conditioning_module,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=1e-4,
    device='cuda'
):
    """Train FoldFlow with conditioning."""
    
    # Move models to device
    foldflow_model = foldflow_model.to(device)
    conditioning_module = conditioning_module.to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        list(foldflow_model.parameters()) + list(conditioning_module.parameters()),
        lr=learning_rate
    )
    
    # Loss function
    loss_fn = MSELoss()
    
    for epoch in range(num_epochs):
        # Training
        foldflow_model.train()
        conditioning_module.train()
        
        total_loss = 0
        for batch in train_loader:
            # Move to device
            ca_coords = batch['ca_coords'].to(device)  # (batch, N, 3)
            
            # Get condition embeddings
            condition_embeddings = []
            for i in range(len(batch['pdb_id'])):
                cond = {
                    'kinase_name': batch['condition']['kinase_name'][i],
                    'conformational_state': batch['condition']['conformational_state'][i],
                    'dfg_state': batch['condition']['dfg_state'][i],
                    'alphac_state': batch['condition']['alphac_state'][i],
                }
                emb = conditioning_module(cond, kinase_index_map)
                condition_embeddings.append(emb)
            
            condition_embeddings = torch.stack(condition_embeddings)  # (batch, embedding_dim)
            
            # Forward pass through FoldFlow
            output = foldflow_model(ca_coords, condition=condition_embeddings)
            
            # Compute loss (example: reconstruction loss)
            loss = loss_fn(output, ca_coords)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        foldflow_model.eval()
        conditioning_module.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ca_coords = batch['ca_coords'].to(device)
                # ... same process as training ...
                output = foldflow_model(ca_coords, condition=condition_embeddings)
                val_loss += loss_fn(output, ca_coords).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss / len(val_loader):.4f}")
    
    return foldflow_model, conditioning_module

# Train
foldflow_model, conditioning_module = train_foldflow_conditional(
    foldflow_model=foldflow_model,  # Your FoldFlow model
    conditioning_module=conditioning,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=1e-4
)
```

---

## Inference: Generating Conditioned Structures

### Generate for Specific Condition

```python
def generate_conditioned_structure(
    foldflow_model,
    conditioning_module,
    condition_dict,
    kinase_index_map,
    device='cuda'
):
    """Generate a protein structure for given condition."""
    
    foldflow_model.eval()
    conditioning_module.eval()
    
    with torch.no_grad():
        # Get condition embedding
        condition_emb = conditioning_module(condition_dict, kinase_index_map)
        condition_emb = condition_emb.unsqueeze(0).to(device)  # Add batch dim
        
        # Generate (implementation depends on FoldFlow)
        # This is pseudocode - adjust to actual FoldFlow API
        generated_structure = foldflow_model.generate(
            condition=condition_emb,
            num_steps=100,
            noise_scale=1.0
        )
    
    return generated_structure

# Example: Generate active EGFR
condition = {
    'kinase_name': 'EGFR',
    'conformational_state': 'active',
    'dfg_state': 'DFG-in',
    'alphac_state': 'alphaC-in'
}

generated = generate_conditioned_structure(
    foldflow_model,
    conditioning_module,
    condition,
    kinase_index_map
)

# Save generated structure
import numpy as np
np.save('generated_egfr_active.npy', generated.cpu().numpy())
```

### Generate Interpolations

```python
def generate_interpolation(
    foldflow_model,
    conditioning_module,
    kinase_name,
    num_steps=5,
    kinase_index_map=None,
    device='cuda'
):
    """Generate smooth interpolation from inactive → active."""
    
    foldflow_model.eval()
    conditioning_module.eval()
    
    generated_structures = []
    
    for t in np.linspace(0, 1, num_steps):
        with torch.no_grad():
            # Create interpolated condition
            condition = {
                'kinase_name': kinase_name,
                'conformational_state': 'active' if t > 0.5 else 'inactive',
                'dfg_state': 'DFG-out' if t < 0.5 else 'DFG-in',
                'alphac_state': 'alphaC-out' if t < 0.5 else 'alphaC-in'
            }
            
            condition_emb = conditioning_module(condition, kinase_index_map)
            condition_emb = condition_emb.unsqueeze(0).to(device)
            
            structure = foldflow_model.generate(condition=condition_emb)
            generated_structures.append(structure)
    
    return generated_structures

# Generate interpolation for BRAF
structures = generate_interpolation(
    foldflow_model,
    conditioning_module,
    kinase_name='BRAF',
    num_steps=10
)
```

---

## Advanced: Multi-Level Conditioning

### Hierarchical Conditioning

```python
class HierarchicalConditioning(nn.Module):
    """Multi-level conditioning: protein → family → kinase → state."""
    
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # Level 1: Protein family (tyrosine, serine/threonine, etc.)
        self.family_embedding = nn.Embedding(10, embedding_dim)
        
        # Level 2: Kinase name
        self.kinase_embedding = nn.Embedding(50, embedding_dim)
        
        # Level 3: Conformational state
        self.state_embedding = nn.Embedding(2, embedding_dim)
        
        # Combine with attention
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.combine = nn.Linear(embedding_dim * 3, embedding_dim)
    
    def forward(self, family_idx, kinase_idx, state_idx):
        emb_family = self.family_embedding(family_idx)
        emb_kinase = self.kinase_embedding(kinase_idx)
        emb_state = self.state_embedding(state_idx)
        
        # Stack embeddings
        embeddings = torch.stack([emb_family, emb_kinase, emb_state])  # (3, batch, dim)
        
        # Apply attention
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        attended = attended.mean(dim=0)  # (batch, dim)
        
        # Combine
        combined = torch.cat([emb_family, emb_kinase, emb_state], dim=-1)
        output = self.combine(combined)
        
        return output
```

---

## Integration Checklist

- [ ] Dataset prepared with `make pipeline`
- [ ] Splits created (grouped by kinase)
- [ ] Tensors preprocessed
- [ ] Custom Dataset class created
- [ ] Conditioning module implemented
- [ ] FoldFlow model loaded
- [ ] Training loop ready
- [ ] Validation metrics defined
- [ ] Inference code tested
- [ ] Results saved and visualized

---

## Performance Tips

1. **Use mixed precision** for faster training
2. **Gradient accumulation** if large batch size causes OOM
3. **Learning rate scheduling** (decay over time)
4. **Validation early stopping** to avoid overfitting
5. **Data augmentation** (rotation, noise) for robustness

---

## References

- FoldFlow documentation
- KLIFS paper: https://klifs.net
- Flow Matching for Protein Generation

---

**Last updated**: 2026-05-25
