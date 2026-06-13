# Tips y Troubleshooting

## Errores Comunes y Soluciones

### 1. Import Error: No module named 'requests'

**Error:**
```
ModuleNotFoundError: No module named 'requests'
```

**Solución:**
```bash
pip install requests
# o
pip install -r requirements.txt
```

### 2. Error: "No kinases found"

**Error:**
```
RuntimeError: No se encontraron kinasas relacionadas con cáncer
```

**Causas:**
- No hay conexión a internet
- KLIFS API está caída
- Cambió la estructura de la API

**Solución:**
```bash
# Verificar KLIFS está funcionando
curl https://klifs.net

# Verificar conectividad
python -c "import requests; requests.get('https://klifs.net')"

# Comprobar si hay errores en logs
python scripts/download_klifs_dataset.py 2>&1 | head -50
```

### 3. Error: "File not found: filepath"

**Problema:** El PDB descargado no existe cuando se intenta procesar.

**Solución:**
```bash
# Verificar que la descarga completó
ls data/raw/pdbs/ | wc -l

# Si está vacío, reiniciar descarga
make download-dataset
```

### 4. Out of Memory

**Error:**
```
CUDA out of memory. Tried to allocate ...
# o
MemoryError: Unable to allocate ...
```

**Soluciones:**
```bash
# Opción 1: Reducir batch size en preprocess_dataset.py
# batch_size = 8  # en lugar de 32

# Opción 2: Procesar solo subset del dataset
python -c "
import pandas as pd
df = pd.read_csv('data/splits/train.csv')
df.head(100).to_csv('data/splits/train_small.csv', index=False)
"

# Opción 3: Usar CPU en lugar de GPU
export CUDA_VISIBLE_DEVICES=""
```

### 5. Error de Conexión a KLIFS

**Error:**
```
requests.exceptions.ConnectionError
requests.exceptions.Timeout
```

**Soluciones:**
```bash
# Verificar firewall/proxy
ping klifs.net

# Aumentar timeout en download_klifs_dataset.py
downloader = KLIFSDownloader(retry_delay=2.0)

# Usar VPN si está bloqueado regionalmente
```

### 6. API Rate Limiting

**Problema:** KLIFS rechaza requests (429 Too Many Requests)

**Solución:**
```python
# En download_klifs_dataset.py
# Aumentar delay entre requests
time.sleep(0.5)  # en lugar de 0.1
```

## Optimizaciones

### 1. Acelerar Descarga

Si la descarga es lenta:

```bash
# Usar concurrencia (requiere concurrent.futures)
# O descargar solo kinasas específicas

# Editar download_klifs_dataset.py
CANCER_KINASES = {
    'EGFR': 'tyrosine kinase',
    'BRAF': 'serine/threonine kinase',
}
```

### 2. Acelerar Preprocessing

Si el preprocessing es lento:

```bash
# Aumentar batch size
batch_size = 64  # (si tienes memoria)

# Usar más workers en DataLoader
DataLoader(dataset, num_workers=4)

# Usar GPU si disponible
# PyTorch detecta automáticamente
```

### 3. Reducir Tamaño del Dataset

```python
import pandas as pd

# Cargar dataset completo
df = pd.read_csv('data/metadata/kinase_labels.csv')

# Filtrar solo kinasas de interés
df_filtered = df[df['kinase_name'].isin(['EGFR', 'BRAF', 'ABL1'])]

# Guardar subset
df_filtered.to_csv('data/metadata/kinase_labels_small.csv', index=False)

# Crear splits del subset
# ...
```

## Debugging

### 1. Modo Verbose

```python
# En download_klifs_dataset.py
logging.basicConfig(level=logging.DEBUG)

# Verá TODOS los logs, incluyendo requests details
```

### 2. Guardar Logs

```python
# Agregar a requirements.txt
pip install python-json-logger

# En scripts, guardar logs:
handler = logging.FileHandler('pipeline.log')
logger.addHandler(handler)
```

### 3. Inspeccionar Datos

```python
# Script para inspeccionar
import pandas as pd

df = pd.read_csv('data/metadata/kinase_labels.csv')

# Ver ejemplos
print(df.head())

# Estadísticas
print(df.describe())

# Buscar valores nulos
print(df.isnull().sum())

# Valores únicos
print(df['conformational_state'].unique())
```

## Performance

### Monitoreo

```bash
# Monitor durante descarga
watch -n 1 'ls data/raw/pdbs/ | wc -l'

# Monitor de recursos
# Windows: Task Manager
# Linux: top, htop
# Mac: Activity Monitor

# Monitor con Python
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"Disk: {psutil.disk_usage('/').percent}%")
```

### Benchmarking

```python
import time

# Medir tiempo de descarga
start = time.time()
downloader.download_all()
elapsed = time.time() - start
print(f"Descarga completada en {elapsed:.1f}s")

# Medir tiempo de preprocessing
start = time.time()
preprocessor.run()
elapsed = time.time() - start
print(f"Preprocessing completado en {elapsed:.1f}s")
```

## Validación de Datos

### Verificar Integridad

```python
# Script para validar
from pathlib import Path
import pandas as pd

def validate_dataset():
    # Verificar CSV
    df = pd.read_csv('data/metadata/kinase_labels.csv')
    assert len(df) > 0, "CSV vacío"
    assert not df.isnull().any().any(), "CSV con valores NULL"
    
    # Verificar archivos PDB
    for idx, row in df.iterrows():
        path = Path(row['filepath'])
        assert path.exists(), f"PDB no encontrado: {path}"
    
    # Verificar splits
    for split in ['train', 'val', 'test']:
        path = Path(f'data/splits/{split}.csv')
        assert path.exists(), f"Split no encontrado: {path}"
        split_df = pd.read_csv(path)
        assert len(split_df) > 0, f"Split vacío: {split}"
    
    print("✅ Dataset validado exitosamente")

validate_dataset()
```

## Recuperación de Errores

### Si se Interrumpe la Descarga

```python
# 1. Verificar qué se descargó
ls -la data/raw/pdbs/ | tail

# 2. Crear lista de PDBs faltantes
import pandas as pd

all_df = pd.read_csv('data/metadata/kinase_labels.csv')
downloaded = set([f.replace('.pdb', '').upper() for f in os.listdir('data/raw/pdbs')])

missing = all_df[~all_df['pdb_id'].isin(downloaded)]
print(f"Faltantes: {len(missing)} PDBs")

# 3. Reintentar solo los faltantes
# O simplemente correr downloader.download_all() nuevamente
# - ignorará archivos ya descargados
```

### Si se Interrumpe el Preprocessing

```python
# 1. Ver qué se procesó
ls data/processed/

# 2. Continuar desde donde paró
# Script modificado:

processed = set(os.listdir('data/processed'))
for idx, row in df.iterrows():
    if row['pdb_id'] not in processed:
        # Procesar este
        ...
```

## Almacenamiento

### Gestión de Espacio

```bash
# Ver tamaño de directorios
du -sh data/raw/pdbs/
du -sh data/processed/
du -sh figures/

# Si necesitas liberar espacio
rm -rf figures/
rm -rf data/processed/  # Puede regenerarse
```

### Backup

```bash
# Backup solo de metadata (lo importante)
cp data/metadata/kinase_labels.csv kinase_labels_backup.csv
cp data/splits/*.csv splits_backup/

# Si necesitas recuperar PDBs descargados
# Usa KLIFS API nuevamente
```

## Tips Generales

### ✅ Lo que Funciona Bien

- Procesar EGFR primero (muchas estructuras, descarga rápida)
- Usar batch_size=32 (buen balance memoria/velocidad)
- Validar con pequeño subset antes de correr completo

### ❌ Lo que NO hacer

- No eliminar `data/raw/pdbs/` - toma tiempo descargar
- No cambiar estructura de `data/splits/` sin entender consecuencias
- No usar valores muy altos de batch_size sin GPU potente

### 🔍 Debugging Tips

```bash
# Ver primeras líneas de logs
head -20 pipeline.log

# Ver últimas líneas
tail -50 pipeline.log

# Buscar errores
grep -i "error" pipeline.log

# Contar eventos
grep -c "Downloaded" pipeline.log
```

## Contacto/Soporte

Si encuentras un error:

1. **Verifica KLIFS está online**: https://klifs.net
2. **Valida tu setup**: `make validate`
3. **Lee los logs**: revisa salida de los scripts
4. **Prueba subset pequeño**: filtra solo una kinasa para testear
5. **Reporta issue** con logs relevantes

---

**Última actualización:** 2025-05-25
