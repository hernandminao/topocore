# Casos de Fallo Documentados -- TopoCore

Tres casos de fallo reales encontrados durante esta sesión de
validación, cada uno con su umbral cuantificado y verificado con
datos/casos reales -- no hipotéticos.

---

## Caso 1: `GridGroundClassifier` degenera a "todo es terreno" con densidad baja

**Síntoma real observado**: Type I = 0.00%, Type II = 100.00% (todos
los puntos clasificados como terreno, sin importar si lo eran).

**Umbral**: ocurre cuando el promedio de puntos por celda es
**menor a 1** (`densidad_real × cell_size² < 1`).

**Por qué**: el algoritmo toma el punto más bajo de cada celda como
terreno. Con menos de 1 punto por celda en promedio, la mayoría de
celdas tienen 0 o 1 punto -- cada punto aislado se vuelve trivialmente
"el mínimo de su propia celda", sin ninguna comparación real posible
contra vecinos.

**Caso real que lo confirma**:
- Forest Site 7 (ISPRS), densidad real 0.153 pts/m², `cell_size=0.5`
  -> 0.038 puntos/celda promedio -> Kappa=0.00 (degenerado)
- Mismo sitio, `cell_size=2.5` -> 0.96 puntos/celda promedio (cerca
  del umbral, pero ya por debajo de 1) -> Kappa=29.83 (funcional,
  aunque bajo)

**Mitigación**: elegir `cell_size` tal que `densidad_real × cell_size²
>= 1`, idealmente varias veces mayor para robustez estadística real
(no solo 1 punto de margen).

---

## Caso 2: Duplicación de coordenadas XY tras adelgazado por voxel 3D

**Síntoma real observado**: `TriangulationError: Duplicated XY
coordinates were found in the input point set.` al construir un TIN
después de `VoxelSampler`.

**Umbral**: ocurre cuando, dentro de una misma columna horizontal
(mismo par X,Y aproximado), existe variación vertical real que cruza
más de 1 celda de `voxel_size` en Z -- es decir, cuando
`(z_max_local - z_min_local) > voxel_size` para un área horizontal
menor a `voxel_size`.

**Por qué**: `VoxelSampler` agrupa en 3D completo (X, Y, Z), no en una
rejilla 2D. Para terreno con variación vertical real (pendiente,
micro-relieve, o clasificación de terreno imperfecta que deja puntos
casi-terreno a distinta altura), 2 voxels de distinta altura dentro de
la misma columna horizontal pueden promediar (centroide) a coordenadas
XY idénticas.

**Caso real que lo confirma**: `537200_3666200.laz`, `voxel_size=0.5`
-> 1 punto con XY duplicada de 131,393 totales. Con `PointCloud_
autzen_classified.laz`, `voxel_size=2.0` -> 6 puntos duplicados de
2,539,207 totales. Proporcionalmente raro, pero determinista y
garantizado con suficiente volumen de datos reales.

**Mitigación aplicada**: deduplicar por XY antes de construir el TIN,
conservando el Z más bajo de cada par (convención de superficie de
terreno) -- implementado en `_extraer_puntos3d()`
(`flujo_nube_completo_dxf_rico.py`), con aviso explícito de cuántos
puntos se dedujeron.

---

## Caso 3: El costo de CSF/PMF escala con el área geográfica, no con la cantidad de puntos

**Síntoma real observado**: un archivo con MÁS puntos pero MENOR área
(`537200_3666200.laz`, 18M puntos, ~110m×99m) clasifica más rápido que
uno con MENOS puntos pero MAYOR área (`autzen_classified.laz`, 10.6M
puntos, ~1044m×1419m reales).

**Umbral**: el número de celdas de la malla de simulación (CSF) o
raster compacto (PMF) es `(ancho_real / resolución) × (alto_real /
resolución)` -- independiente del número de puntos. Para PMF
específicamente, existe un límite de seguridad concreto:
`max_grid_cells=8,000,000` (por defecto) -- excederlo lanza
`GroundError` de inmediato, antes de cualquier cómputo pesado.

**Por qué**: ambos algoritmos construyen una rejilla/malla sobre el
área geográfica completa cubierta por los datos, con celdas de tamaño
`resolución`/`cell_size` fijo -- el número de puntos reales dentro de
cada celda no afecta cuántas celdas existen.

**Caso real que lo confirma**:
- CSF: `autzen_classified.laz` con `cloth_resolution=0.5` -> malla de
  6855×9315 = 63.8 millones de celdas (el área real es ~1044m×1419m)
- PMF: mismo archivo, `cell_size=1.0` -> `GroundError` real,
  reproducido: "PMF compact grid would contain 15951456 cells,
  exceeding max_grid_cells=8000000. Increase cell_size to
  approximately 1.41207 or raise max_grid_cells."

**Mitigación**: elegir `cell_size`/`cloth_resolution` en función del
área real cubierta, no de la densidad de puntos ni del tamaño del
archivo en disco -- fórmula real: `resolución_mínima ≈ sqrt(ancho_real
× alto_real / max_grid_cells)`.

---

## Nota metodológica

Los 3 casos fueron encontrados de forma reactiva, ejecutando el
pipeline sobre datos reales (no diseñados deliberadamente para
provocar el fallo) -- lo que da más confianza en que son
representativos de fallos que un usuario real encontraría, no casos
de laboratorio artificiales.
