# 🔒 Guía: Data Leakage y Estrategias de Split para Biología Estructural

## 📌 Resumen Ejecutivo

El refactoring de splits ahora usa **grouped splits por kinase** en lugar de splits estratificados simples. Esto previene **data leakage estructural** y da una medida más realista de la capacidad del modelo para generalizar a proteínas completamente nuevas.

---

## 1️⃣ ¿Qué es Data Leakage?

### Definición Simple

**Data leakage** ocurre cuando información del conjunto de test "filtra" hacia el conjunto de entrenamiento, permitiendo que el modelo vea información que no debería ver durante el entrenamiento.

### Analogía: Examen en la Escuela

Imagina que:
- **SIN leakage**: El profesor te enseña temas generales en clase (Train)
- **CON leakage**: El profesor te da el examen exacto para practicar en casa

En el segundo caso, tus notas son muy altas, pero no significa que hayas aprendido a resolver problemas nuevos. Simplemente memorizaste la solución.

---

## 2️⃣ ¿Por Qué es Peligroso?

### Problema 1: Metrics Overoptimistic (Métricas Infladas)

```
❌ CON LEAKAGE:
  Train Accuracy: 95%
  Test Accuracy:  94%  ← Parece excelente
  Conclusión: "¡Modelo perfecto!"
  
✅ SIN LEAKAGE:
  Train Accuracy: 95%
  Test Accuracy:  62%  ← Baja mucho
  Conclusión: "El modelo memoriza, no generaliza"
```

### Problema 2: False Confidence

El modelo parece funcionar bien en validación, así que lo deployas en producción. Allí falla miserablemente con datos reales no vistos.

### Problema 3: Decisiones de Investigación Equivocadas

Comparas dos arquitecturas:
- Modelo A: Accuracy 88% (SIN leakage)
- Modelo B: Accuracy 92% (CON leakage)

Eliges Modelo B, pero en realidad Modelo A es mejor. Solo que tiene menos leakage.

---

## 3️⃣ Leakage Estructural en Tu Pipeline Original

### El Problema

Tu pipeline original hacía:

```python
# ❌ PROBLEMA: Split estratificado simple
train, test = train_test_split(
    df,
    test_size=0.3,
    stratify=df['conformational_state']  # Solo estratifica por estado
)
```

**¿Qué pasaba?**

```
Dataset con estructuras:
├── EGFR estructura A → TRAIN (por casualidad)
├── EGFR estructura B → TEST  (por casualidad)  ← LEAKAGE!
├── BRAF estructura X → TRAIN
├── BRAF estructura Y → TEST  ← LEAKAGE!
└── ...
```

### Por Qué es Problemático

El modelo durante entrenamiento ve:
- Estructura A de EGFR (TRAIN)
- Estructura B de EGFR (TEST) - **¡Es la misma proteína!**

La estructura 3D de una proteína es muy similar entre estructuras de la misma proteína con diferentes conformaciones. El modelo puede:

1. **Memorizar patrones específicos de EGFR** en lugar de aprender principios generales
2. **Overfitting a patrones de proteínas específicas** en lugar de patrones de activación/inactivación universales
3. **Fallar completamente con proteínas nuevas** no vistas en entrenamiento

### Impacto Real

```
❌ CON LEAKAGE (Estratificado):
  - Test Accuracy: 87% 
  - Pero todas las kinasas de test estaban en train
  - Modelo dice: "EGFR es así" memorizado
  
✅ SIN LEAKAGE (Grouped):
  - Test Accuracy: 65%
  - Test tiene kinasas completamente nuevas
  - Modelo dice: "Estos patrones funcionan en nuevas proteínas"
```

---

## 4️⃣ Grouped Splits: La Solución

### Cómo Funciona

```python
# ✅ SOLUCIÓN: Agrupamientos por grupos biológicos
# Todos los EGFR → Una split
# Todos los BRAF → Una split
# Todos los ABL1 → Una split
# etc.

Distribución:
├── TRAIN (70%)
│   ├── Todas las estructuras de EGFR     } Completo
│   ├── Todas las estructuras de BRAF     } por grupo
│   ├── Todas las estructuras de AKT1     }
│   └── ...
├── VAL (15%)
│   ├── Todas las estructuras de CDK4     } Completo
│   ├── Todas las estructuras de KIT      } por grupo
│   └── ...
└── TEST (15%)
    ├── Todas las estructuras de PDGFRA   } Completo
    ├── Todas las estructuras de FGFR1    } por grupo
    └── ...
```

### Garantías

✅ **Sin overlap de kinases entre splits**
- Train ve EGFR, BRAF, ABL1, ...
- Test ve PDGFRA, FGFR1, ALK, ... (completamente diferentes)

✅ **Simula el caso de uso real**
- En producción, verás kinasas completamente nuevas
- Este setup simula eso fielmente

✅ **Mejor generalización**
- Modelo aprende: "Estos patrones de activación funcionan en cualquier proteína"
- No memoriza: "EGFR tiene estos patrones"

---

## 5️⃣ Por Qué Grouped es Mejor para Biología Estructural

### Especificidad de Proteínas

Cada proteína kinasa tiene:
- Estructura 3D única
- Secuencia de aminoácidos única
- Patrones de plegamiento únicos
- Composición química única

**Problema**: El modelo fácilmente confunde "estructura de EGFR" con "característica de activación".

### Generalización Importante

En aplicaciones reales:
- Estudias EGFR, BRAF, ABL1 (training)
- Necesitas predecir conformación en NOVO_KINASE no estudiada
- Si tu modelo solo memorizó patrones de EGFR, fracasará

### Biological Validity

La pregunta real es:
- ❌ "¿Puedo predecir si ESTA estructura de una kinasa conocida es activa?" (Fácil)
- ✅ "¿Puedo predecir si CUALQUIER kinasa nueva es activa/inactiva?" (Difícil, realista)

Los grouped splits responden la pregunta ✅.

---

## 6️⃣ Limitaciones: Trabajar con ~13 Kinasas

### El Desafío

Tienes ~13 kinasas diferentes. Para grouped split:
- 70% train → ~9 kinasas
- 15% val → ~2 kinasas
- 15% test → ~2 kinasas

### Problemática

**Balance de clases vs. Grupo separation**

```
Dilema:
├─ OPTION A: Perfect stratification
│  └─ Cada split tiene 50% activas, 50% inactivas
│     Pero asignas grupos arbitrariamente
│
└─ OPTION B: Group separation
   └─ Separas grupos completamente
      Pero proporciones pueden ser 70% activas, 30% inactivas
```

### La Solución del Proyecto: Priorizar Leakage

```yaml
# Decisión en config.yaml:
prioritize_group_separation: true
```

**Significa**: "Primero evita leakage, balance perfecto es secundario"

### Impacto

✅ **Lo Bueno**:
- Cero leakage estructural
- Validación realista
- Generalización comprobada

⚠️ **Lo Difícil**:
- Balance de clases imperfecto
- Tienes que entrenar con datasets más desbalanceados
- Necesitas técnicas de balanceamiento: weighted loss, resampling, etc.

---

## 7️⃣ Impacto en Accuracy y Generalización

### Predicción: Caída de Accuracy Esperada

```
Con Leakage (estratificado):      Sin Leakage (grouped):
├─ Train Acc: 92-95%             ├─ Train Acc: 85-90%
├─ Val Acc:   88-92%             ├─ Val Acc:   68-75%
├─ Test Acc:  85-90%  ← FALSO!   └─ Test Acc:  65-75%  ← REAL
│                                    (con kinasas nuevas)
└─ Generalización: MALA
   (Memorización de proteínas)
```

### Análisis Detallado

**Accuracy caerá ~15-25%**: Normal y esperado
- Es el "costo real" de honestidad experimental
- Tu modelo es más conservador pero más creíble

**Generalización mejorará**: Kinasas nuevas funcionarán mejor
- Si tu modelo aprende: "Estos patrones = activo en CUALQUIER kinasa"
- Generalizará mejor a datos reales

### Interpretación Correcta

```
INCORRECTO:
"Antes tenía 90% accuracy, ahora 70%"
"¡Mi refactoring rompió el modelo!"

CORRECTO:
"Antes medía mal, tenía 90% accuracy FALSO"
"Ahora mido bien, tengo 70% accuracy REAL"
"Mi modelo sigue siendo igual de bueno, solo mido correctamente"
```

---

## 8️⃣ Cambios en Tu Pipeline

### Workflow Antiguo (❌ CON LEAKAGE)

```
1. download_klifs_dataset.py
   └─ Genera: train.csv, val.csv, test.csv (estratificado)
   
2. preprocess_dataset.py
   └─ Procesa todas las estructuras
   
3. training/finetune.py
   └─ Usa train.csv para entrenamiento
   └─ Valida con val.csv (PROBLEMA: kinasas ya vistas)
```

### Workflow Nuevo (✅ SIN LEAKAGE)

```
1. download_klifs_dataset.py
   └─ Genera: train.csv, val.csv, test.csv (GROUPED por kinase_name)
      └─ Validación explícita: cero overlap de kinases
      └─ Logging detallado: qué kinase en cada split
   
2. preprocess_dataset.py
   └─ Procesa todas las estructuras
   └─ Valida que splits CSVs existen
   └─ Verifica integridad: PDB IDs en splits tienen tensores
   
3. training/finetune.py
   └─ Usa train.csv para entrenamiento (kinases conocidas)
   └─ Valida con val.csv (kinases nuevas, 15%)
   └─ Evalúa con test.csv (kinases completamente nuevas, 15%)
```

### Archivos Modificados

| Archivo | Cambio | Propósito |
|---------|--------|----------|
| `config.yaml` | Nuevo `strategy: "grouped"` | Configurar tipo de split |
| `download_klifs_dataset.py` | Reescrito `create_train_val_test_splits()` | Implementar grouped splits |
| `preprocess_dataset.py` | Agregado `_validate_preprocessing()` | Validar integridad datos |

### Archivos NO Modificados

- `training/finetune.py` → Ya funciona con los splits
- `evaluation/*.py` → Ya funciona con los splits
- Dataloader code → Compatible automáticamente

### Si Necesitas Regenerar

**Pregunta: ¿Borro data/processed?**

```
Respuesta: No necesariamente.

Opción A: Regenerar todo (RECOMENDADO primera vez)
  1. Elimina: data/processed/
  2. Corre: make download-dataset
  3. Corre: make preprocess
  
Opción B: Reutilizar tensores
  1. Corre: make download-dataset  (genera nuevos splits)
  2. Reutiliza: data/processed/    (tensores viejos, splits nuevos)
  
Razón: Los tensores NO cambian, solo qué subset usa cada split.
Los tensores son del PDB, los splits son tu decisión experimental.
```

---

## 9️⃣ Validación Visual en Logs

### Qué Buscar en los Logs

**Ejecución exitosa:**

```
================================================================================
CREANDO SPLITS CON CONTROL DE LEAKAGE
================================================================================
Estrategia: GROUPED
Agrupar por: kinase_name
Proporciones: Train=70.0%, Val=15.0%, Test=15.0%
Total estructuras: 523
Total kinases únicos: 13

🔬 ESTRATEGIA GROUPED SPLIT
   Dividiendo 13 grupos (kinases) entre splits
   Distribución de grupos:
     - Train: 9 grupos
     - Val:   2 grupos
     - Test:  2 grupos

📊 ESTADÍSTICAS DE SPLITS

TRAIN:
  Estructuras:   366 ( 70.0%)
  Activas:       243 ( 66.4%)
  Inactivas:     123 ( 33.6%)
  kinase_names:    9

VAL:
  Estructuras:    80 ( 15.3%)
  Activas:        46 ( 57.5%)
  Inactivas:      34 ( 42.5%)
  kinase_names:    2

TEST:
  Estructuras:    77 ( 14.7%)
  Activas:        52 ( 67.5%)
  Inactivas:      25 ( 32.5%)
  kinase_names:    2

🧬 KINASE_NAMES POR SPLIT
  TRAIN (9): ABL1, AKT1, BRAF, CDK4, CDK6, EGFR, ERBB2, FGFR1, KIT
  VAL   (2): MET, PDGFRA
  TEST  (2): ALK, PIK3CA

✅ VALIDACIÓN DE LEAKAGE
   ✓ Sin overlap entre TRAIN, VAL, TEST
   ✓ Separación de kinase_names garantizada
   ✓ Cobertura: 523 estructuras distribuidas
   ✓ Proporciones finales: Train 70.0%, Val 15.3%, Test 14.7%

✅ Splits guardados en data/splits/
================================================================================
```

### Qué Significa

| Indicador | Bueno | Malo |
|-----------|-------|------|
| `✓ Sin overlap` | ✅ Excelente | ❌ LEAKAGE |
| `kinase_names por split` | Diferentes | Overlap → PROBLEMA |
| `Estructuras distribuidas` | ~100% cobertura | Menos = datos perdidos |
| `Proporciones` | Cercanas a 70/15/15 | Muy desbalanceadas = warning |

---

## 🔟 Troubleshooting

### Problema: "Proporciones no son exactas 70/15/15"

**Causa**: Con ~13 kinases, imposible distribución perfecta

**Solución**: Normal, esperable
```
13 kinases
├─ 9 grupos en train (69.2%)
├─ 2 grupos en val   (15.3%)
└─ 2 grupos en test  (15.4%)
```

### Problema: "Balance ACTIVE/INACTIVE desigual entre splits"

**Causa**: Prioridades: "Evita leakage > Balance perfecto"

**Solución**: Usar `weighted_loss` en training
```python
# En training, calcula peso de clases
pos_weight = n_inactive / n_active  # Pesa las minoritarias
loss = BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Problema: "Test accuracy bajó 20%"

**¿Malo?** Solo si era leakage antes

**Verificar**:
```
Si ANTES (con leakage):
  Train: 95%, Val: 92%, Test: 90%
  
Y AHORA (sin leakage):
  Train: 92%, Val: 65%, Test: 63%
  
→ MEJOR porque es honesto
  La verdadera capacidad era 63%, no 90%
```

### Problema: "Caen demasiado los números"

**Esto significa**: Tu modelo memorizaba mucho
**Oportunidad**: Mejorar arquitectura, agregar regularización, data augmentation

---

## 1️⃣1️⃣ Conclusiones

### ✅ Beneficios del Nuevo Sistema

| Aspecto | Antes | Ahora |
|--------|-------|-------|
| **Leakage** | Presente (kinases overlap) | Ausente (kinases separadas) |
| **Validación** | Poco realista | Realista (kinases nuevas) |
| **Métricas** | Infladas | Honestas |
| **Generalización** | Desconocida | Medida realmente |

### 🎯 Objetivos del Proyecto

Tu proyecto ahora responde correctamente:

✅ "¿Puedo predecir conformación de kinasas completamente nuevas?"

En lugar de:

❌ "¿Puedo memorizar estructuras de kinasas conocidas?"

### 📊 Próximos Pasos

1. **Regenerar splits**: `make download-dataset`
2. **Preprocesar**: `make preprocess` (valida integridad)
3. **Entrenar**: `python training/finetune.py` (con splits nuevos)
4. **Evaluar**: `python evaluation/evaluate.py` (en test real)
5. **Analizar**: Compare métricas con baseline, ajuste según necesidad

### 🔍 Referencia: Conceptos de ML

**Términos relacionados**:
- **Data Leakage**: Información de test filtra a train
- **Overfitting**: Memorización vs. generalización
- **Distribution Shift**: Train y test de distribuciones diferentes
- **Stratification**: Balance de clases en splits
- **Grouped Stratification**: Balance + groups no overlappean

---

## Apéndice: Config YAML Antigua vs Nueva

### ❌ Antigua (Con Leakage)

```yaml
splits:
  train: 0.70
  validation: 0.15
  test: 0.15
  random_state: 42
  stratified: true  # Mezcla kinases entre splits
```

### ✅ Nueva (Sin Leakage)

```yaml
splits:
  train: 0.70
  validation: 0.15
  test: 0.15
  random_state: 42
  
  strategy: "grouped"            # Agrupa por kinase
  group_by: "kinase_name"        # Cada kinase completa en 1 split
  prioritize_group_separation: true  # Evita leakage > balance perfecto
  prevent_leakage: true          # Validar cero overlap
  verbose_reporting: true        # Logs detallados
```

---

**Última actualización**: 2026-05-25
**Versión**: 2.0 (Grouped Splits)
