# Conclusión automática

## active_to_inactive
- Ganancia de probabilidad del estado objetivo vs copiar source: -0.044.
- Mejora de distancia a la distribución objetivo: -0.032.
- Validez geométrica: 0.146; geometría rota: 0.819.
- MAE por residuo vs referencia débil: 4.148; mejor que ruido: no.
- Cambio local funcional suficiente: sí.
- Diagnóstico: evidencia insuficiente.

## inactive_to_active
- Ganancia de probabilidad del estado objetivo vs copiar source: -0.133.
- Mejora de distancia a la distribución objetivo: -0.094.
- Validez geométrica: 0.386; geometría rota: 0.538.
- MAE por residuo vs referencia débil: 3.550; mejor que ruido: no.
- Cambio local funcional suficiente: sí.
- Diagnóstico: evidencia insuficiente.

## Veredicto
El modelo no cumple todos los criterios de plausibilidad y superioridad frente a baselines. No debe declararse exitoso solo por la predicción del clasificador congelado.
