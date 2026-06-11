# Resultados de Flow Matching + CFG

Este directorio contiene una **prueba técnica preliminar** del modelo de
Flow Matching + Classifier-Free Guidance para transiciones de kinasas.

## Conclusión principal

El código funciona de punta a punta y genera estructuras para:

- `t = 0.00`
- `t = 0.25`
- `t = 0.50`
- `t = 0.75`
- `t = 1.00`

Sin embargo, estos resultados todavía **no demuestran que el modelo haya
aprendido la transición inactiva -> activa**.

Los estados intermedios son continuos y diferentes entre sí, pero permanecen
muy cerca de la estructura inactiva. En esta corrida:

| Métrica | Resultado |
|---|---:|
| Pares evaluados | 8 |
| Kinasa evaluada | PDGFRA |
| RMSD inicial inactiva -> activa | 12.15 A |
| RMSD generada -> activa | 12.19 A |
| RMSD generada -> inactiva | 1.35 A |
| Mejora relativa media | -0.45% |
| Generaciones más cercanas a activa que a inactiva | 0% |
| Longitud media de la trayectoria | 1.35 A |

El RMSD generado-activo es ligeramente peor que el RMSD inicial-activo.
Además, la estructura final queda a sólo 1.35 A de la inactiva. Por lo tanto,
el campo aprendido produce cambios pequeños, pero no dirige correctamente la
estructura hacia la conformación activa.

## ¿Se ven pasos intermedios?

Sí, se ven cinco frames continuos en:

`plots/checkpoint_structural_trajectory.png`

El gráfico muestra cambios graduales entre `t=0` y `t=1`. No obstante, que una
trayectoria sea suave no significa que sea biológicamente plausible.

En la proyección PCA:

- Distancia media inactiva-activa: 11.62 unidades PCA.
- Desplazamiento total medio generado: 0.27 unidades PCA.
- Progreso medio proyectado en dirección a la activa: -0.5%.

Los puntos generados se agrupan sobre las estructuras inactivas y no alcanzan
la nube activa. La respuesta correcta para esta corrida es:

> Hay pasos intermedios numéricos, pero todavía no hay evidencia de pasos
> intermedios conformacionales activos ni biológicamente plausibles.

## Alcance de esta corrida

Estos archivos provienen de una validación reducida:

- 2 trials de Optuna.
- 2 épocas por trial.
- 64 puntos C-alpha remuestreados.
- Máximo de 8 pares por kinasa.
- Evaluación guardada sobre 8 pares de PDGFRA.

Los conteos `67/16/16` de `pair_counts.json` pertenecen a esta ejecución
reducida. No representan todos los pares disponibles. Una auditoría sin límite
reducido había encontrado 454 pares train, 120 validation y 106 test.

No debe usarse esta prueba para extraer conclusiones biológicas definitivas ni
para comparar arquitecturas.

## Guía de archivos

### Auditoría de datos

`data_structure_report.md`

Resume las rutas detectadas, cantidad de tensores, longitudes de proteínas,
duplicados y chequeos de leakage. Los 863 conjuntos de tensores son válidos.
No hay PDBs ni kinasas compartidas entre train, validation y test.

`data_structure_report.json`

La misma información en formato procesable.

`pairing_report.csv`

Indica cuántas estructuras activas e inactivas existen por kinasa y cuántos
pares fueron seleccionados en esta corrida.

`pair_counts.json`

Cantidad total de pares seleccionados por split para la última ejecución.

`vocab.json`

Índices usados por los embeddings de estado, kinasa, DFG y alphaC.

### Optuna

`best_hyperparameters.json`

Mejor configuración entre los trials ejecutados. El trial preliminar eligió:

- arquitectura convolucional;
- learning rate `8.47e-5`;
- batch size `4`;
- hidden dimension `192`;
- embedding dimension `64`;
- 6 capas;
- guidance scale `1.32`;
- 8 pasos de integración.

Con sólo dos trials no puede afirmarse que estos parámetros sean óptimos.

`optuna_trials.csv`

Una fila por trial con parámetros, validation loss y validation RMSD.

`optuna_study.db`

Base SQLite que permite continuar el estudio sin perder los trials anteriores.

`optuna_plots/*.html`

Gráficos interactivos:

- `optimization_history.html`: evolución del objetivo.
- `parameter_importance.html`: importancia estimada.
- `parallel_coordinates.html`: relación conjunta entre parámetros y objetivo.
- `slice.html`: efecto individual de cada parámetro.

La importancia de parámetros no es confiable con sólo dos trials.

`best_optuna_checkpoint.pt`

Checkpoint del mejor trial preliminar.

### Métricas

`metrics/checkpoint_per_pair.csv`

Una fila por par inactiva-activa. Las columnas principales son:

| Columna | Interpretación |
|---|---|
| `rmsd_generated_to_target` | Distancia final al destino; menor es mejor |
| `rmsd_initial_to_target` | Distancia inicial al destino |
| `rmsd_generated_to_source` | Distancia final al estado de origen |
| `rmsd_generated_to_active` | Distancia final a la activa real |
| `rmsd_generated_to_inactive` | Distancia final a la inactiva real |
| `distance_matrix_mae` | Error geométrico interno; menor es mejor |
| `conformational_diversity` | Magnitud de variación durante la trayectoria |
| `path_length` | Distancia total recorrida |
| `relative_improvement` | Positivo indica acercamiento al destino |
| `closer_to_target_than_initial` | Indica mejora respecto del punto inicial |
| `closer_to_target_than_source` | Indica cruce hacia el estado destino |

Los CSV generados antes de incorporar el experimento bidireccional pueden usar
los nombres anteriores `rmsd_initial_to_active` y `closer_to_active`.

`metrics/checkpoint_per_kinase.csv`

Promedios por kinasa. La evaluación actual sólo contiene PDGFRA, por lo que no
permite identificar mejores y peores kinasas.

`metrics/checkpoint_manifold_projection.csv`

Coordenadas PCA y t-SNE de estructuras reales y generadas. Permite reconstruir
y analizar las trayectorias en el espacio conformacional.

### Gráficos

`plots/checkpoint_metrics.png`

- Izquierda: distribución de RMSD generado-activo.
- Centro: RMSD por kinasa.
- Derecha: comparación inicial-activo contra generado-activo.

En el panel derecho, un punto debajo de la diagonal indica mejora. Los puntos
actuales están prácticamente sobre la diagonal, con mejoras pequeñas en sólo
dos de ocho pares.

`plots/checkpoint_conformational_field_pca.png`

Los círculos azules son inactivas, las cruces naranjas son activas y las
trayectorias violetas son generaciones. Las trayectorias permanecen alrededor
de las inactivas.

`plots/checkpoint_conformational_field_tsne.png`

Proyección no lineal complementaria. También muestra que los puntos generados
se mantienen asociados a los estados inactivos.

`plots/checkpoint_structural_trajectory.png`

Representación 3D de los cinco tiempos para el par PDGFRA `8pqk -> 8pqh`.
Las diferencias son pequeñas y no se muestra la activa real superpuesta, por
lo que este gráfico sirve para comprobar continuidad, no similitud biológica.

No se generó UMAP por una incompatibilidad local entre `numba` y `coverage`.
Esto no afecta PCA, t-SNE, entrenamiento ni las métricas estructurales.

### Archivos `smoke_flow_cfg`

Son resultados de una prueba de una época destinada únicamente a verificar que
el programa corre. No deben incluirse en el análisis científico.

## Qué falta para el experimento completo

Todavía no existen en este directorio:

- `comparison_table.csv`;
- resultados completos de los cinco baselines;
- entrenamiento largo del mejor modelo;
- evaluación sobre todos los pares test;
- comparación real entre PDGFRA y PIK3CA.

Se generan ejecutando:

```bash
cd /Users/josefinadehan/tp_vision_avanzada/TPF_Vision_Avanzada

python training/flow_matching_cfg_transition_optuna.py all \
  --trials 50 \
  --epochs 100 \
  --patience 15
```

Esta ejecución puede tardar varias horas en CPU.

## Cómo reconocer un resultado exitoso

Después del entrenamiento completo deberían observarse simultáneamente:

1. `rmsd_generated_to_active < rmsd_initial_to_active`.
2. Mejora relativa claramente positiva.
3. Una proporción alta de `closer_to_active=True`.
4. Trayectorias PCA que salgan de la nube inactiva y avancen hacia la activa.
5. Distancia a la inactiva creciente con `t`.
6. Distancia a la activa decreciente con `t`.
7. Errores de matrices de distancia menores que los modelos sin conditioning.
8. Resultados consistentes en PDGFRA y PIK3CA, no sólo en pares aislados.
9. Ventaja de Flow Matching + CFG frente a los otros baselines.

Aunque estas condiciones se cumplan, la plausibilidad biológica requiere
validaciones adicionales: correspondencia correcta de residuos, geometría
all-atom, choques estéricos, motivos DFG/HRD/alphaC, dinámica molecular,
estabilidad y perfiles de energía libre.

## Experimento bidireccional

El script también permite entrenar y comparar:

- `Inactiva -> Intermedia -> Activa`
- `Activa -> Intermedia -> Inactiva`

La prueba rápida se ejecuta con:

```bash
python training/flow_matching_cfg_transition_optuna.py smoke-bidirectional
```

La corrida smoke ya validada está en `bidirectional_smoke/`. Sus métricas sólo
comprueban funcionamiento y no deben interpretarse como resultado científico.

El experimento real se ejecuta con:

```bash
python training/flow_matching_cfg_transition_optuna.py bidirectional \
  --epochs 100 \
  --patience 15
```

Si existe `best_hyperparameters.json`, el comando `bidirectional` reutiliza esa
configuración de Optuna y entrena un modelo independiente para cada dirección.
No se reutiliza automáticamente el checkpoint de activación para desactivación,
porque ese checkpoint nunca fue entrenado con destino `inactive`.

También pueden proporcionarse checkpoints específicos:

```bash
python training/flow_matching_cfg_transition_optuna.py bidirectional \
  --activation-checkpoint ruta/modelo_activacion.pt \
  --deactivation-checkpoint ruta/modelo_desactivacion.pt
```

### Salidas bidireccionales

`bidirectional_comparison_table.csv`

Tabla principal solicitada:

```text
Dirección | RMSD final | Mejora relativa |
% más cerca del target | % cruza al estado destino
```

`bidirectional_detailed_comparison.csv`

Incluye además distancia final a activa, distancia final a inactiva, desviación
del RMSD, longitud de trayectoria y error de matrices de distancia.

`metrics/bidirectional_activation_per_pair.csv`

Resultados de `inactiva -> activa`.

`metrics/bidirectional_deactivation_per_pair.csv`

Resultados de `activa -> inactiva`.

`metrics/bidirectional_per_kinase.csv`

Comparación de ambos sentidos separada por kinasa.

`plots/bidirectional_comparison.png`

Compara RMSD, mejora relativa y porcentajes de éxito de ambos sentidos.

`plots/bidirectional_per_kinase.png`

Muestra qué dirección funciona mejor para cada kinasa.

`plots/bidirectional_deactivation_structural_trajectory.png`

Muestra los cinco estados del camino activa -> inactiva.

`bidirectional_biological_interpretation.md`

Responde automáticamente si el modelo aprende mejor activación o desactivación,
si el camino inverso es más estable, si alcanza el estado inactivo y en qué
kinasas funciona mejor.

En estas tablas:

- `% más cerca del target` significa que el RMSD final al destino es menor que
  la distancia que existía entre la estructura inicial y el destino.
- `% cruza al estado destino` exige que la estructura final quede más cerca del
  destino que del estado de origen. Es un criterio más fuerte.
