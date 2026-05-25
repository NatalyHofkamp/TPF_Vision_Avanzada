# Pipeline de Dataset de Kinasas

## Resumen

Este pipeline automatiza la descarga, procesamiento y preparación del dataset KLIFS (Kinase-Ligand Interaction Fingerprints and Structures) para fine-tuning de modelos de generación condicionada de proteínas.

## Flujo de trabajo

```
1. download_klifs_dataset.py
   ↓
   Descarga estructuras de kinasas humanas de KLIFS
   ↓
2. explore_dataset.py
   ↓
   Analiza estadísticas del dataset
   ↓
3. preprocess_dataset.py
   ↓
   Convierte PDB a tensores para entrenamiento
```

## Scripts

### 1. `download_klifs_dataset.py`

Descarga estructuras proteicas desde la API KLIFS.

**Lo que hace:**

- Obtiene lista de kinasas disponibles en KLIFS
- Filtra solo kinasas humanas relacionadas con cáncer (EGFR, BRAF, ABL1, etc.)
- Obtiene estructuras disponibles para esas kinasas
- Descarga automáticamente archivos PDB
- Clasifica estructuras como ACTIVAS o INACTIVAS basado en:
  - **DFG-state**: "DFG-in" (activa) vs "DFG-out" (inactiva)
  - **alphaC-state**: posición de la hélice alfa-C
- Genera archivo CSV con metadata
- Divide dataset en train/val/test (70/15/15)

**Salidas:**

```
data/metadata/kinase_labels.csv
data/raw/pdbs/*.pdb
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

**Uso:**

```bash
python scripts/download_klifs_dataset.py
# o
make download-dataset
```

**Formato del CSV:**

```csv
pdb_id,kinase_name,kinase_family,species,conformational_state,dfg_state,alphac_state,ligand_present,resolution,filepath
1ABB,EGFR,Tyrosine kinase,Homo sapiens,active,DFG-in,alphaC-in,1,2.15,data/raw/pdbs/1abb.pdb
```

### 2. `explore_dataset.py`

Analiza el dataset y genera estadísticas.

**Lo que calcula:**

- Total de estructuras y kinasas
- Distribución de estados conformacionales (activa/inactiva)
- Top kinasas por cantidad de estructuras
- Análisis de DFG y alphaC states
- Estadísticas de resolución cristalográfica
- Distribución de ligandos
- Balance de clases en splits

**Salidas:**

- Tabla de estadísticas en consola
- Gráfico: `figures/dataset_analysis.png`

**Uso:**

```bash
python scripts/explore_dataset.py
# o
make explore-dataset
```

### 3. `preprocess_dataset.py`

Convierte estructuras PDB en tensores para entrenamiento.

**Lo que hace:**

- Extrae coordenadas de átomos Cα (carbono alfa)
- Calcula matrices de distancia entre residuos
- Convierte a tensores PyTorch
- Guarda datos procesados para entrenamiento rápido

**Lo que es Cα:**

- Carbono alfa = átomo central de cada aminoácido
- Las coordenadas Cα definen la estructura 3D de la proteína
- La matriz de distancias captura relaciones espaciales entre residuos

**Salidas:**

```
data/processed/
├── 1ABB/
│   ├── ca_coords.pt          # Tensor con coordenadas Cα
│   ├── distance_matrix.pt    # Tensor con distancias
│   └── metadata.pt           # Metadata (label, resolución, etc.)
├── 2BCD/
│   └── ...
```

**Uso:**

```bash
python scripts/preprocess_dataset.py
# o
make preprocess
```

## Concepto: Estados Conformacionales

Las proteínas kinasas tienen dos conformaciones principales:

### ACTIVA (conformación catáliticamente competente)

- **DFG-in**: la región DFG está insertada en el sitio activo
- **alphaC-in**: la hélice alfa-C está bien posicionada
- **Características**: sitio activo preparado para catálisis
- **Implica**: la kinasa está lista para fosforilar sustratos

### INACTIVA (conformación no catalítica)

- **DFG-out**: la región DFG está expulsada del sitio activo
- **alphaC-out**: la hélice alfa-C está replegada
- **Características**: sitio activo bloqueado
- **Implica**: la kinasa está inhibida o en reposo

## Uso Completo

```bash
# 1. Instalar dependencias
make install

# 2. Descargar dataset KLIFS
make download-dataset

# 3. Explorar estadísticas
make explore-dataset

# 4. Procesar a tensores
make preprocess

# 5. Ver visualizaciones (notebook)
jupyter notebook notebooks/dataset_visualization.ipynb
```

## Estructura de Directorios

```
data/
├── raw/
│   └── pdbs/                    # Archivos PDB descargados
├── metadata/
│   └── kinase_labels.csv        # Metadata completa
├── splits/
│   ├── train.csv               # 70% del dataset
│   ├── val.csv                 # 15% del dataset
│   └── test.csv                # 15% del dataset
└── processed/
    └── [pdb_id]/               # Tensores procesados

scripts/
├── download_klifs_dataset.py   # Pipeline de descarga
├── explore_dataset.py          # Análisis del dataset
└── preprocess_dataset.py       # Conversión a tensores

notebooks/
└── dataset_visualization.ipynb # Análisis interactivo
```

## Características Principales

### ✅ Manejo de Errores

- Reintentos automáticos con backoff exponencial
- Rate limiting para respetar la API
- Logging detallado de errores

### ✅ Progreso Visible

- Barra de progreso (tqdm) para todas las operaciones
- Logs informativos en cada paso

### ✅ Clasificación Automática

- Usa reglas de bioinformática para clasificar conformaciones
- Basado en DFG y alphaC states

### ✅ Balance de Datos

- Split estratificado mantiene proporción de clases
- Garantiza distribución equilibrada en train/val/test

### ✅ Documentación

- Comentarios explicativos sin asumir experiencia en bioinformática
- Docstrings en cada función
- Este README completo

## Ejemplos de Datos

### Kinasas Relacionadas con Cáncer Incluidas

- **EGFR**: receptor de factor de crecimiento epidérmico
- **BRAF**: proteína mutada en melanoma
- **ABL1**: fusión BCR-ABL en leucemia
- **KIT**: receptor de factor de células madre
- **PDGFRA**: receptor de factor de crecimiento derivado de plaquetas
- ... y más

### Ejemplo de Registro

```python
{
    'pdb_id': '1ABB',
    'kinase_name': 'EGFR',
    'kinase_family': 'Tyrosine kinase',
    'species': 'Homo sapiens',
    'conformational_state': 'active',  # ← Lo que queremos predecir
    'dfg_state': 'DFG-in',
    'alphac_state': 'alphaC-in',
    'ligand_present': 1,
    'resolution': 2.15,  # Angstroms
    'filepath': 'data/raw/pdbs/1abb.pdb'
}
```

## Troubleshooting

### Error: "No kinases found"

- Revisa conexión a internet
- La API KLIFS puede estar down (comprueba https://klifs.net)

### Error: "File not found"

- Asegúrate de haber corrido `download-dataset` primero

### Memoria insuficiente

- Reduce `batch_size` en `preprocess_dataset.py`
- Procesa solo un subset del dataset

## Próximos Pasos

Una vez con el dataset preparado, puedes:

1. **Fine-tuning de FoldFlow** con las estructuras descargadas
2. **Conditioning activo/inactivo** usando los labels del dataset
3. **Generación condicionada** de nuevas conformaciones
4. **Entrenamiento generativo** de proteínas con control fino

## Referencias

- KLIFS: https://klifs.net
- API Documentation: https://klifs.net/swagger/swagger.json
- BioPython: https://biopython.org
- PyTorch: https://pytorch.org
