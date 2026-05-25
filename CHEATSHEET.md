# 📋 Cheat Sheet - Comandos Rápidos

## Flujo Básico (Copy-Paste Ready)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Validar que todo está OK
make validate

# 3. Descargar dataset KLIFS
make download-dataset

# 4. Ver estadísticas
make explore-dataset

# 5. Procesar a tensores
make preprocess

# 6. Abrir notebook (en navegador)
jupyter notebook notebooks/dataset_visualization.ipynb
```

## One-Liner (Ejecutar Todo)

```bash
make pipeline
```

---

## Comandos Individuales

### Descarga

```bash
# Descargar KLIFS dataset
python scripts/download_klifs_dataset.py

# O con Make
make download-dataset
```

### Exploración

```bash
# Análisis estadístico
python scripts/explore_dataset.py

# O con Make
make explore-dataset
```

### Preprocesamiento

```bash
# Convertir a tensores PyTorch
python scripts/preprocess_dataset.py

# O con Make
make preprocess
```

### Validación

```bash
# Verificar setup
python scripts/validate_setup.py

# O con Make
make validate
```

---

## Acceso a Datos

### CSV de Metadata

```python
import pandas as pd

# Cargar dataset completo
df = pd.read_csv('data/metadata/kinase_labels.csv')

# Ver primeras estructuras
print(df.head(10))

# Estadísticas
print(df.describe())

# Estructuras activas
active = df[df['conformational_state'] == 'active']
print(f"Activas: {len(active)}")

# Top 5 kinasas
print(df['kinase_name'].value_counts().head(5))
```

### Splits

```python
# Train set
train = pd.read_csv('data/splits/train.csv')

# Val set
val = pd.read_csv('data/splits/val.csv')

# Test set
test = pd.read_csv('data/splits/test.csv')

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
```

### Tensores PyTorch

```python
import torch
from pathlib import Path

pdb_id = '1ABB'
pdb_dir = Path(f'data/processed/{pdb_id}')

# Cargar coordenadas Cα
ca_coords = torch.load(pdb_dir / 'ca_coords.pt')
print(f"Forma: {ca_coords.shape}")  # (seq_len, 3)

# Cargar matriz de distancias
dist_matrix = torch.load(pdb_dir / 'distance_matrix.pt')
print(f"Forma: {dist_matrix.shape}")  # (seq_len, seq_len)

# Cargar metadata
metadata = torch.load(pdb_dir / 'metadata.pt')
print(f"Label: {metadata['label']}")  # 1=activa, 0=inactiva
```

---

## Customización

### Cambiar Kinasas

Editar `scripts/download_klifs_dataset.py`:

```python
CANCER_KINASES = {
    'EGFR': 'tyrosine kinase',
    'BRAF': 'serine/threonine kinase',
    # Agregar más aquí
}
```

### Cambiar Splits

Editar en `config.yaml`:

```yaml
splits:
  train: 0.80      # 80% en lugar de 70%
  validation: 0.10 # 10% en lugar de 15%
  test: 0.10       # 10% en lugar de 15%
```

### Cambiar Batch Size

Editar `scripts/preprocess_dataset.py`:

```python
batch_size = 64  # en lugar de 32
```

---

## Debugging

### Ver logs

```bash
# Ejecutar con output completo
python scripts/download_klifs_dataset.py 2>&1 | tee download.log

# Ver errores
tail -50 download.log
```

### Verificar descarga

```bash
# Contar PDBs descargados
ls data/raw/pdbs/ | wc -l

# Ver tamaño
du -sh data/raw/pdbs/

# Ver primeros PDBs
ls data/raw/pdbs/ | head -5
```

### Verificar Metadata

```bash
# Contar filas
wc -l data/metadata/kinase_labels.csv

# Ver columnas
head -1 data/metadata/kinase_labels.csv | tr ',' '\n' | nl

# Verificar no hay nulos
python -c "import pandas as pd; df=pd.read_csv('data/metadata/kinase_labels.csv'); print(df.isnull().sum())"
```

### Verificar Tensores

```bash
# Contar archivos procesados
ls -d data/processed/*/ | wc -l

# Ver estructura
ls data/processed/1ABB/

# Verificar tensor
python -c "import torch; t=torch.load('data/processed/1ABB/ca_coords.pt'); print(t.shape)"
```

---

## Jupyter Notebooks

### Abrir Notebook

```bash
jupyter notebook notebooks/dataset_visualization.ipynb
```

### Ejecutar celda

- Presionar `Shift + Enter` o click en Run

### Comandos útiles en Notebook

```python
# Cargar datos
import pandas as pd
df = pd.read_csv('data/metadata/kinase_labels.csv')

# Ver datos
print(df.head())

# Plotear
import matplotlib.pyplot as plt
df['conformational_state'].value_counts().plot(kind='bar')
plt.show()
```

---

## Manejo de Errores Comunes

### "ModuleNotFoundError"

```bash
pip install -r requirements.txt
```

### "API Error"

```bash
# Verificar KLIFS
curl https://klifs.net

# O
python -c "import requests; print(requests.get('https://klifs.net').status_code)"
```

### "Out of Memory"

```bash
# Reducir batch size en preprocess_dataset.py
batch_size = 8  # en lugar de 32
```

### "File not found"

```bash
# Asegúrate de haber corrido descarga primero
ls data/raw/pdbs/

# Si está vacío
make download-dataset
```

---

## Archivos Clave

| Archivo | Descripción |
|---------|-------------|
| `data/metadata/kinase_labels.csv` | Metadata de TODAS las estructuras |
| `data/splits/train.csv` | Subset para entrenamiento (70%) |
| `data/splits/val.csv` | Subset para validación (15%) |
| `data/splits/test.csv` | Subset para testing (15%) |
| `data/raw/pdbs/*.pdb` | Archivos 3D descargados |
| `data/processed/[ID]/*.pt` | Tensores PyTorch |

---

## Tips Pro

### Procesar solo subset

```python
df = pd.read_csv('data/metadata/kinase_labels.csv')
df_small = df[df['kinase_name'].isin(['EGFR', 'BRAF'])].head(50)
df_small.to_csv('data/metadata/test_subset.csv', index=False)
```

### Filtrar por calidad

```python
df = pd.read_csv('data/metadata/kinase_labels.csv')
high_quality = df[df['resolution'] < 2.0]  # Mejor resolución
print(f"Estructuras de alta calidad: {len(high_quality)}")
```

### Balance de clases

```python
active = len(df[df['conformational_state'] == 'active'])
inactive = len(df[df['conformational_state'] == 'inactive'])
print(f"Active: {active} | Inactive: {inactive}")
print(f"Ratio: {active/inactive:.2f}")
```

### Verificar splits

```python
for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/splits/{split}.csv')
    active = len(df[df['conformational_state'] == 'active'])
    print(f"{split}: {len(df)} total ({active} activas, {len(df)-active} inactivas)")
```

---

## Atajos Útiles

```bash
# Ver todo en una línea
cat data/metadata/kinase_labels.csv | head -1 && wc -l data/metadata/kinase_labels.csv

# Contar por estado
grep "active" data/metadata/kinase_labels.csv | wc -l
grep "inactive" data/metadata/kinase_labels.csv | wc -l

# Ver kinasas únicas
cut -d',' -f2 data/metadata/kinase_labels.csv | sort | uniq -c | sort -rn
```

---

## Ver Este Cheat Sheet de Nuevo

```bash
cat CHEATSHEET.md
```

---

**Última actualización:** 2025-05-25
