# Diagnóstico de transiciones latentes con Flow Matching

Implementación:

`training/flow_matching_transition_diagnostics.py`

El experimento utiliza exclusivamente endpoints conocidos:

- `x0`: estructura inactiva real.
- `x1`: estructura activa real.

Los estados en `t=0.25`, `0.50` y `0.75` no existen como ground truth en el
dataset. Son hipótesis latentes generadas por el modelo.

Las combinaciones DFG/alphaC se conservan como descriptores funcionales para
analizar estructuras, pero nunca se utilizan como labels de intermedios.

## Alineamiento de endpoints

El script recupera de los PDB:

- cadena;
- residue IDs e insertion codes;
- secuencia;
- coordenadas C-alpha;
- gaps internos;
- motivos DFG, HRD, VAIK y alphaC.

El matching usa residue IDs compartidos y bloques homólogos de secuencia. Se
filtran pares por cobertura, identidad, longitud, resolución y ligando.

## Pérdidas

- `flow_loss`: campo derivado únicamente de `x0` y `x1`.
- `endpoint_loss`: obliga a terminar cerca de la activa real.
- `distance_loss`: restringe matrices de distancia durante todo el camino.
- `motif_loss`: controla DFG, alphaC, HRD y activation loop.
- `smoothness_loss`: penaliza aceleraciones o saltos entre estados latentes.
- `directionality_loss`: exige alejarse progresivamente del origen y acercarse
  al destino.
- `self_consistency_loss`: genera ida y vuelta y recupera el endpoint inicial.
- `classifier_loss`: reconoce los endpoints generados como activos/inactivos.

La referencia de `distance_loss` en tiempos intermedios se deriva de las
matrices de los dos endpoints. Es un regularizador geométrico, no una medición
experimental ni un label del camino real.

## Variantes

- `current_index_matching`: referencia con remuestreo por índice.
- `aligned_endpoint_flow`: endpoints alineados y Flow Matching básico.
- `aligned_endpoint_constraints`: restricciones geométricas y funcionales.
- `aligned_full_conditioning`: agrega kinasa, ligando y resolución.
- `bidirectional_self_consistent`: entrenamiento en ambos sentidos y ciclo.
- `inverse_deactivation`: activa real -> inactiva real.
- `linear_interpolation_oracle`: baseline geométrico que conoce el target real.

La interpolación lineal no es un modelo generativo independiente: usa
directamente `x1`, por lo que su endpoint es perfecto por construcción.

## Evaluación de hipótesis intermedias

No se calcula RMSD contra un intermedio real inexistente. Se evalúan:

- suavidad de la trayectoria;
- progreso monotónico hacia la activa;
- consistencia de matrices de distancia;
- estabilidad de DFG, alphaC, HRD y activation loop;
- evolución de la distancia Lys-Glu;
- RMSD del endpoint generado;
- clasificación del endpoint;
- consistencia de ciclo inactiva-activa-inactiva.

Cada archivo `*_latent_hypotheses.csv` marca:

- `t=0`: estructura fuente observada;
- `t=0.25`, `0.50`, `0.75`: hipótesis latentes.
- `t=1`: endpoint generado y supervisado contra la estructura target observada.

En el baseline oracle, `t=1` sí contiene directamente el target observado,
porque la interpolación lineal usa ambos endpoints por construcción.

## Ejecución

```bash
python training/flow_matching_transition_diagnostics.py audit
python training/flow_matching_transition_diagnostics.py smoke
python training/flow_matching_transition_diagnostics.py run --epochs 50
```

Resultados:

`results/flow_matching_latent_transition_diagnostics/`

## Interpretación

Una trayectoria exitosa debe ser suave, geométricamente estable, avanzar hacia
la activa, preservar estructura local razonable y recuperar la inactiva al
invertir el camino.

Esto no demuestra que la trayectoria sea el mecanismo biológico real. Su
validez requiere dinámica molecular, energía libre, reconstrucción all-atom y
validación experimental.
