# Integración con FoldFlow y Conditioning

## ¿Cómo usar este dataset con FoldFlow?

Este documento explica cómo integrar el dataset KLIFS con el pipeline de generación condicionada de proteínas usando Flow Matching/FoldFlow.

## Visión General

```
Dataset KLIFS
    ↓
├─ Estructuras activas (DFG-in, alphaC-in)
├─ Estructuras inactivas (DFG-out, alphaC-out)
└─ Metadata estructural
    ↓
Preprocesamiento
    ↓
├─ Tensores Cα
├─ Matrices de distancia
└─ Labels (activo/inactivo)
    ↓
FoldFlow + Conditioning
    ↓
Generación de conformaciones
```

## Conceptos de Conditioning

El **conditioning** significa que le dices al modelo de generación qué quieres que genere.

### Ejemplo 1: Conditioning Simple

```python
# "Dame la estructura EGFR en estado ACTIVO"
condition = {
    'kinase_name': 'EGFR',
    'conformational_state': 'active'
}

generated_structure = foldflow.generate(condition=condition)
```

### Ejemplo 2: Conditioning Multi-criterio

```python
# "Dame BRAF inactivo con alta resolución"
condition = {
    'kinase_name': 'BRAF',
    'conformational_state': 'inactive',
    'dfg_state': 'DFG-out',
    'alphac_state': 'alphaC-out',
}

generated_structure = foldflow.generate(condition=condition)
```

### Ejemplo 3: Conditioning Continuo

```python
# Interpolar entre dos conformaciones
condition = {
    'conformational_state': 0.5  # 50% activa, 50% inactiva
}

intermediate_structure = foldflow.generate(condition=condition)
```

## Arquitectura Propuesta

### 1. DataLoader para FoldFlow

```python
from torch.utils.data import DataLoader

# Cargar splits
train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')
test_df = pd.read_csv('data/splits/test.csv')

# Crear dataloaders
train_loader = create_protein_dataloader(
    train_df,
    batch_size=32,
    shuffle=True
)

val_loader = create_protein_dataloader(
    val_df,
    batch_size=32,
    shuffle=False
)
```

### 2. Conditioning Module

```python
class ConformationConditioning(nn.Module):
    """Embedding de condiciones conformacionales."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        
        # Embeddings para cada tipo de condición
        self.kinase_embedding = nn.Embedding(50, embedding_dim)
        self.conformation_embedding = nn.Embedding(2, embedding_dim)  # active/inactive
        self.dfg_embedding = nn.Embedding(2, embedding_dim)  # in/out
        self.alphac_embedding = nn.Embedding(2, embedding_dim)  # in/out
        
        # Combinador
        self.proj = nn.Linear(embedding_dim * 4, embedding_dim)

    def forward(self, kinase_idx, conformation_idx, dfg_idx, alphac_idx):
        """
        Combina múltiples condiciones.
        
        Args:
            kinase_idx: índice de la kinasa
            conformation_idx: 0=activa, 1=inactiva
            dfg_idx: 0=in, 1=out
            alphac_idx: 0=in, 1=out
        
        Returns:
            Embedding condicionado (batch_size, embedding_dim)
        """
        emb_kinase = self.kinase_embedding(kinase_idx)
        emb_conf = self.conformation_embedding(conformation_idx)
        emb_dfg = self.dfg_embedding(dfg_idx)
        emb_alphac = self.alphac_embedding(alphac_idx)
        
        combined = torch.cat([emb_kinase, emb_conf, emb_dfg, emb_alphac], dim=-1)
        return self.proj(combined)
```

### 3. Loop de Entrenamiento

```python
def train_with_conditioning(
    model,
    conditioning_module,
    train_loader,
    num_epochs=100,
    learning_rate=1e-4
):
    """Entrena FoldFlow con conditioning."""
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(conditioning_module.parameters()),
        lr=learning_rate
    )
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Extraer datos
            ca_coords = batch['ca_coords']  # (batch, seq_len, 3)
            labels = batch['label']  # (batch,) - activo/inactivo
            kinase_names = batch['kinase_name']
            
            # Generar condiciones
            condition_emb = conditioning_module(
                kinase_idx=encode_kinase_names(kinase_names),
                conformation_idx=labels,
                dfg_idx=encode_dfg_state(batch['dfg_state']),
                alphac_idx=encode_alphac_state(batch['alphac_state'])
            )
            
            # Forward pass
            loss = model(ca_coords, condition=condition_emb)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
```

## Usos Prácticos

### Uso 1: Fine-tuning de FoldFlow

```python
from models.conditioning import ConformationConditioning

# 1. Cargar FoldFlow pre-entrenado
foldflow = load_foldflow_checkpoint('checkpoints/foldflow_pretrained.pt')

# 2. Agregar módulo de conditioning
conditioning = ConformationConditioning(embedding_dim=256)

# 3. Entrenar
train_with_conditioning(
    model=foldflow,
    conditioning_module=conditioning,
    train_loader=train_loader,
    num_epochs=50
)

# 4. Guardar
torch.save(conditioning.state_dict(), 'checkpoints/conditioning.pt')
```

### Uso 2: Generación Condicionada

```python
# Generar conformación activa de EGFR
egfr_active = foldflow.generate(
    condition={
        'kinase': 'EGFR',
        'state': 'active'
    },
    num_samples=10,
    temperature=1.0
)

# Generar conformación inactiva de BRAF
braf_inactive = foldflow.generate(
    condition={
        'kinase': 'BRAF',
        'state': 'inactive'
    },
    num_samples=5,
    temperature=0.8  # Menos variabilidad
)
```

### Uso 3: Interpolación entre Estados

```python
# Interpolar de activa a inactiva
n_steps = 10
for t in np.linspace(0, 1, n_steps):
    structure = foldflow.generate(
        condition={
            'interpolation': t,  # 0=activa, 1=inactiva
            'kinase': 'EGFR'
        }
    )
    # Guardar estructura de tránsito
    save_pdb(structure, f'interpolation_step_{t:.2f}.pdb')
```

## Estructura de Datos para FoldFlow

El dataset preprocesado contiene:

```
data/processed/[PDB_ID]/
├── ca_coords.pt          # Coordenadas Cα (seq_len, 3)
├── distance_matrix.pt    # Matriz de distancias (seq_len, seq_len)
└── metadata.pt           # {label, kinase, resolution, dfg, alphac}
```

### Cómo Cargar en PyTorch

```python
class KinaseConformationDataset(Dataset):
    def __init__(self, csv_path, processed_dir):
        self.df = pd.read_csv(csv_path)
        self.processed_dir = Path(processed_dir)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_id = row['pdb_id']
        
        pdb_dir = self.processed_dir / pdb_id
        
        # Cargar tensores
        ca_coords = torch.load(pdb_dir / 'ca_coords.pt')
        distance_matrix = torch.load(pdb_dir / 'distance_matrix.pt')
        metadata = torch.load(pdb_dir / 'metadata.pt')
        
        return {
            'pdb_id': pdb_id,
            'ca_coords': ca_coords,
            'distance_matrix': distance_matrix,
            'label': metadata['label'],  # 1=activa, 0=inactiva
            'kinase': row['kinase_name'],
            'family': row['kinase_family'],
            'resolution': metadata['resolution'],
            'dfg_state': row['dfg_state'],
            'alphac_state': row['alphac_state'],
        }

# Uso
dataset = KinaseConformationDataset(
    csv_path='data/splits/train.csv',
    processed_dir='data/processed'
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    print(f"Batch size: {len(batch['ca_coords'])}")
    print(f"Sequence lengths: {batch['ca_coords'].shape}")
    print(f"Labels: {batch['label']}")
```

## Evaluación

### Métricas de Distancia

```python
from scipy.spatial.distance import cdist

def compute_rmsd(predicted_coords, reference_coords):
    """Root Mean Square Deviation entre estructuras."""
    # Alinear
    centered_pred = predicted_coords - predicted_coords.mean(axis=0)
    centered_ref = reference_coords - reference_coords.mean(axis=0)
    
    # Calcular RMSD
    rmsd = np.sqrt(np.mean((centered_pred - centered_ref) ** 2))
    return rmsd

# Usar en validación
for batch in val_loader:
    generated = model.generate(condition=batch['condition'])
    rmsd = compute_rmsd(generated.cpu().numpy(), batch['ca_coords'].numpy())
    print(f"RMSD: {rmsd:.2f} Å")
```

### Métricas de Conformación

```python
def evaluate_conformational_accuracy(predicted_labels, true_labels):
    """¿Qué tan bien predice la conformación?"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
```

## Próximos Pasos

1. **Integrar dataset** con tu código de FoldFlow
2. **Definir módulo de conditioning** específico para tu arquitectura
3. **Entrenar** modelo con multitask learning (reconstrucción + predicción de conformación)
4. **Evaluar** generación en ambas conformaciones
5. **Publicar** resultados de generación condicionada

## Referencias Útiles

- Flow Matching: https://arxiv.org/abs/2210.02747
- Protein Structure Prediction: https://github.com/facebookresearch/foldflow
- PyTorch Dataloader: https://pytorch.org/docs/stable/data.html

---

Para más detalles técnicos sobre el dataset, ver `scripts/README.md`.
