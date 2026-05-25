# Quick Start Guide

## Pipeline de Dataset KLIFS - Inicio Rápido

### 1️⃣ Instalación (5 min)

```bash
# Instalar todas las dependencias
make install
```

### 2️⃣ Descargar Dataset (30-60 min)

```bash
# Descarga estructuras de kinasas humanas del cáncer desde KLIFS
make download-dataset
```

✅ **Qué obtuviste:**

- `data/metadata/kinase_labels.csv` - Información de cada estructura
- `data/raw/pdbs/*.pdb` - Archivos de estructura 3D
- `data/splits/train.csv`, `val.csv`, `test.csv` - Dataset dividido

### 3️⃣ Explorar Dataset (2 min)

```bash
# Ver estadísticas del dataset
make explore-dataset
```

📊 **Salida:**

```
===================
ESTADÍSTICAS BÁSICAS
===================
Total: 500+ estructuras
Activas: 60%
Inactivas: 40%
Top kinasas: EGFR, BRAF, ABL1...
Resolución media: 2.15 Å
```

### 4️⃣ Procesar a Tensores (15-30 min)

```bash
# Convierte PDB a tensores PyTorch para entrenamiento
make preprocess
```

✅ **Qué obtuviste:**

- `data/processed/` - Tensores listos para ML
- Cada estructura = coordenadas Cα + matriz de distancias

### 5️⃣ Visualizar (Interactivo)

```bash
jupyter notebook notebooks/dataset_visualization.ipynb
```

📊 Incluye:

- Distribución de estados conformacionales
- Análisis de kinasas
- Estadísticas de resolución
- Balance de clases en splits

---

## Estructura de Datos

### Metadata CSV

```csv
pdb_id,kinase_name,conformational_state,dfg_state,alphac_state,resolution,filepath
1ABB,EGFR,active,DFG-in,alphaC-in,2.15,data/raw/pdbs/1abb.pdb
1ABC,EGFR,inactive,DFG-out,alphaC-out,2.30,data/raw/pdbs/1abc.pdb
```

### Labels

- **active** = conformación catalíticamente competente
- **inactive** = conformación inactiva

---

## Conceptos Clave

### DFG State

- **DFG-in**: región D-F-G dentro del sitio activo → proteína ACTIVA
- **DFG-out**: región D-F-G fuera del sitio activo → proteína INACTIVA

### alphaC State

- **alphaC-in**: hélice alfa-C bien posicionada → proteína ACTIVA
- **alphaC-out**: hélice alfa-C replegada → proteína INACTIVA

### Cα Coordinates

- Carbono alfa de cada aminoácido
- Define la estructura 3D
- Usado para machine learning

---

## Comandos Principales

```bash
# Instalar dependencias
make install

# Descargar dataset KLIFS (target principal)
make download-dataset

# Explorar estadísticas
make explore-dataset

# Procesar a tensores
make preprocess

# Ver visualizaciones
jupyter notebook notebooks/dataset_visualization.ipynb
```

---

## ¿Qué Hacer Después?

Ahora tienes un dataset limpio y estructurado para:

1. **Fine-tuning FoldFlow** con conditioning activo/inactivo
2. **Generación condicionada** de proteínas kinasas
3. **Entrenamiento** de modelos de aprendizaje profundo
4. **Análisis estructural** de dinámicas conformacionales

---

## Troubleshooting

| Problema | Solución |
|----------|----------|
| API error | Verifica: https://klifs.net |
| Out of memory | Reduce batch_size en scripts |
| Missing files | Asegúrate de haber corrido download primero |
| Import errors | `pip install -r requirements.txt` |

---

## Archivos Generados

```
✅ data/metadata/kinase_labels.csv        (500+ estructuras)
✅ data/raw/pdbs/*.pdb                    (Archivos 3D)
✅ data/splits/{train,val,test}.csv       (Split 70/15/15)
✅ data/processed/*/                      (Tensores PyTorch)
✅ figures/dataset_analysis.png           (Visualizaciones)
```

**Tamaño total:** ~2-5GB (varía según kinasas descargadas)

---

## Documentación Completa

Ver `scripts/README.md` para detalles técnicos de cada script.
