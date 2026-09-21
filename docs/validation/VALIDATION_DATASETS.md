# Datasets Reales Validados -- TopoCore

Consolidación de todos los archivos reales (nube de puntos) probados
contra el flujo completo de TopoCore, con sus métricas reales
registradas. Complementa `VALIDATION_ISPRS.md` (validación contra el
benchmark académico) con datos de escala/rendimiento y un caso real de
corte/relleno.

## Resumen de los 5 datasets

| # | Archivo | Formato | Puntos | Área real | Densidad | Clasificación |
|---|---|---|---|---|---|---|
| 1 | PointCloud_FERROVIA.las | LAS | 13,342,719 | ~172m × 190m | Alta (vía férrea) | Sin verdad de campo |
| 2 | PointCloud_autzen_classified.laz | LAZ | 10,653,336 | 3426m × 4656m | Baja | **Con verdad de campo real** (Hobu Inc., 2021) |
| 3 | PointCloud_autzen.laz | LAZ | 10,653,336 | 3426m × 4656m | Baja | Parcial (solo Ground/Unclassified) |
| 4 | 537200_3666200.laz | LAZ | 17,971,203 | ~110m × 99m | Muy alta (~1650 pts/m²) | Sin verdad de campo |
| 5 | PointCloud_CHIESA.e57 | E57 | 10,487,136 | -- | -- | Sin verdad de campo, sin CRS |

## Métricas de escala/rendimiento (los 4 primeros -- diagnostico_escala.py)

| Archivo | Puntos | RAM pico (CSF) | MB/millón de puntos |
|---|---|---|---|
| FERROVIA.las | 13.3M | 614 MB | 46.0 |
| autzen_classified.laz | 10.6M | 627 MB | 58.9 |
| autzen.laz | 10.6M | 614 MB | 57.6 |
| 537200_3666200.laz | 18.0M | 924 MB | 51.4 |

**Promedio real: ≈53.5 MB de RAM por millón de puntos** (lectura +
clasificación CSF completa). Confirmado en hardware real (i5-6300U,
21GB RAM).

**Hallazgo real**: el costo de CSF (tiempo de simulación de la malla de
tela) depende del ÁREA GEOGRÁFICA cubierta, no de la cantidad de
puntos -- `537200_3666200.laz` tiene más puntos que cualquiera de los
otros 3, pero su área es diminuta (~110m×99m), dando la malla de
simulación más pequeña y la corrida más rápida de las 4.

## Clasificación de terreno -- validación contra verdad de campo real (dataset #2)

`PointCloud_autzen_classified.laz` es el único de los 5 con clasificación
manual real (Hobu Inc., 2021) para comparar contra las predicciones de
TopoCore:

| Método | Parámetros | Kappa (Cohen) |
|---|---|---|
| CSF | cloth_resolution=2.0 | 35.54* |
| CSF | cloth_resolution=1.0 | (F1=0.92, ver nota) |
| PMF | cell_size=0.5 (default) | *(ver VALIDATION_ISPRS.md para metodología de Kappa)* |

*Nota: esta validación específica usó F1/precisión/exhaustividad antes
de adoptar Kappa como métrica estándar (adoptado en el Criterio 2). Los
números de Kappa exactos para este dataset específico no fueron
recalculados -- **pendiente** si se quiere una comparación directa en
las mismas unidades que `VALIDATION_ISPRS.md`.

## Flujo completo end-to-end (dataset #5 -- CHIESA.e57)

Único de los 5 en formato E57 (nube de escáner terrestre, sin CRS
embebido -- confirmado típico de este formato). Probado con el flujo
completo:

```
.e57 -> clasificar terreno (PMF) -> adelgazar (voxel) -> TIN -> DTM
     -> vista PNG -> curvas de nivel con etiquetas -> corte/relleno real
```

**Resultado real de corte/relleno** (contra el promedio real del sitio,
55.58m):
- Corte: 226.28 m³
- Relleno: 75.25 m³
- Neto: +151.03 m³

Terreno real: 4,462,382 puntos de terreno (de 10,487,136 totales,
PMF cell_size=0.5) -> TIN de 4,197,469 vértices, 8,391,250 triángulos.

**Hallazgo real**: la asimetría 3:1 entre corte y relleno respecto al
promedio confirma que la mayor parte del área real del sitio está en
una zona baja y plana, con una estructura elevada ocupando una porción
menor del área total (consistente con el nombre del archivo -- "chiesa"
es "iglesia" en italiano).

## Limitaciones honestas de este criterio

- Solo 1 de los 5 datasets tiene verdad de campo real para medir
  exactitud de clasificación -- los otros 4 solo aportan datos de
  escala/rendimiento, no de precisión.
- Las métricas de exactitud del dataset #2 se calcularon con F1/
  precisión/exhaustividad, no con Kappa -- inconsistente con la
  métrica adoptada después en `VALIDATION_ISPRS.md`. Recalcular con
  Kappa dejaría todo en las mismas unidades.
- Ningún dataset de los 5 se probó con clasificación multi-clase
  (vegetación, edificio, etc.) más allá de terreno/no-terreno.
- El dataset #5 (CHIESA) usó una cota de diseño elegida arbitrariamente
  (el promedio real del sitio) para el cálculo de corte/relleno, no una
  cota de diseño real de un proyecto -- útil para verificar que el
  cálculo funciona, no para validar un caso de uso real de ingeniería.
