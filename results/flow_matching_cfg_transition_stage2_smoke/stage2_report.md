# Stage-2 transition learning report

## Main result

- Best learned model: **no_conditioning**.
- Best guidance scale: **0.50**.
- Best global RMSD: **6.478 A**.
- Relative improvement: **-789.25%**.
- The model still cannot be claimed as a biological mechanism without external validation.

## Comparison summary

| Model | Guidance | RMSD final | Improvement | Directionality | Smoothness | Cycle RMSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear_interpolation_oracle | 1.00 | 0.000 | 100.00% | 1.000 | 0.000 | 0.000 |
| no_conditioning | 0.50 | 6.478 | -789.25% | 0.250 | 0.046 | 9.821 |
| inverse_deactivation | 0.50 | 7.388 | -1010.85% | 0.125 | 0.058 | 9.751 |
| aligned_full_conditioning | 1.50 | 7.966 | -1133.84% | 0.188 | 0.063 | 18.661 |
| bidirectional_self_consistent | 1.00 | 8.918 | -1299.84% | 0.000 | 0.068 | 12.170 |

## Camino conformacional generado por el modelo

- Modelo de referencia para el camino exportado: `aligned_full_conditioning`.
- Puntos exportados: `t=0.00`, `0.25`, `0.50`, `0.75`, `1.00`.
- Los puntos intermedios son hipótesis latentes generadas por el modelo.
- Los archivos PDB se guardan bajo `outputs/transition_paths/`.

### Archivos exportados

- [pair_000_6joi_to_8pqi](/Users/josefinadehan/tp_vision_avanzada/TPF_Vision_Avanzada/results/flow_matching_cfg_transition_stage2_smoke/outputs/transition_paths/PDGFRA/pair_000_6joi_to_8pqi)
- [pair_001_6joj_to_8pqh](/Users/josefinadehan/tp_vision_avanzada/TPF_Vision_Avanzada/results/flow_matching_cfg_transition_stage2_smoke/outputs/transition_paths/PDGFRA/pair_001_6joj_to_8pqh)
- [pair_002_5sxe_to_5sx9](/Users/josefinadehan/tp_vision_avanzada/TPF_Vision_Avanzada/results/flow_matching_cfg_transition_stage2_smoke/outputs/transition_paths/PIK3CA/pair_002_5sxe_to_5sx9)

### Interpretación

La etapa 2 mejora el experimento si baja RMSD, aumenta la direccionalidad,
reduce el error geométrico interno y mantiene suavidad en la trayectoria. La
validez biológica sigue sin poder afirmarse solo con estas métricas; hace falta
validación experimental o simulación molecular adicional.
