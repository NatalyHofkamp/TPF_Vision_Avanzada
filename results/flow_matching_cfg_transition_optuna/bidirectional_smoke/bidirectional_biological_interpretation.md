# Comparación automática de direcciones

## Respuesta breve

- El modelo aprende mejor: **ninguna; ambas fallan el criterio de mejora y cruce al destino**, usando mejora relativa como criterio.
- La dirección con resultado numérico menos desfavorable es: **desactivación**.
- La transición activa -> inactiva es más estable: **no**.
- La generación inversa termina más cerca de la inactiva que de la activa:
  **no**.
- Kinasas donde el camino inverso mejora más que el directo:
  **PDGFRA**.

## Activación: inactiva -> activa

- RMSD final al target: 22.751 A.
- Mejora relativa: -118.84%.
- Pares que mejoran respecto del inicio: 0.0%.
- Pares que terminan más cerca del target que del source:
  0.0%.

## Desactivación: activa -> inactiva

- RMSD final al target: 17.348 A.
- Mejora relativa: -65.20%.
- Pares que mejoran respecto del inicio: 0.0%.
- Pares que terminan más cerca del target que del source:
  0.0%.
- Distancia final a la activa: 12.209 A.
- Distancia final a la inactiva: 17.348 A.

## Interpretación biológica

Una asimetría entre direcciones puede reflejar diferencias del dataset, la
heterogeneidad de las conformaciones activas e inactivas o un sesgo del modelo;
no demuestra por sí sola que la activación o desactivación biológica sea más
fácil. Si la desactivación resulta más estable y cruza al destino con mayor
frecuencia, puede sugerir que el conjunto de estados inactivos es un atractor
estructural más compacto bajo esta representación. Esta hipótesis requiere
validación por motivos funcionales, reconstrucción all-atom, dinámica molecular
y perfiles de energía libre.
