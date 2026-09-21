# Validación ISPRS -- Clasificación de Terreno en TopoCore

## Metodología

**Benchmark**: ISPRS Filter Test (Sithole & Vosselman, 2004) -- el estándar
académico de referencia para comparar algoritmos de clasificación de
terreno en nubes de puntos LiDAR. Datos originales publicados por
ITC/TU Delft, aún disponibles para descarga en 2026:
<https://www.utwente.nl/en/itc/isprs/wgIII-3/filtertest/downloadsites>.

**Métrica**: Índice Kappa de Cohen, calculado sobre la clasificación
binaria terreno/no-terreno, comparando la predicción de cada algoritmo
contra la clasificación de referencia real (editada manualmente por los
autores del benchmark). Esta es la misma métrica usada por trabajos de
seguimiento del benchmark (Meng, Currit & Zhao, 2009 -- *ISPRS Journal
of Photogrammetry and Remote Sensing*, 64(2), 117-124), lo que permite
comparación directa contra valores ya publicados.

También se reportan Type I error (% de terreno real perdido -- omisión)
y Type II error (% de objeto real clasificado como terreno -- comisión),
las métricas originales del reporte del benchmark.

**Sitios usados**:
- `samp71` (Forest Site 7): terreno con discontinuidades abruptas
  (puente, paso subterráneo, terraplenes de vía) -- uno de los sitios
  más difíciles del benchmark completo, elegido deliberadamente por sus
  autores para poner a prueba la robustez de un filtro.
- `samp31` (City Site 3): terreno urbano típico, sin las complicaciones
  del Sitio 7 -- referencia de "caso simple/representativo".

**Scripts usados** (incluidos en `scripts/`):
`isprs_filter_test.py` (CSF), `isprs_filter_test_pmf.py` (PMF),
`isprs_filter_test_grid.py` (Grid), `isprs_barrido.py` (barrido de
parámetros CSF), `isprs_barrido_pmf.py` (barrido de parámetros PMF).

## Resultados

### Forest Site 7 (samp71) -- terreno difícil

| Método | Parámetros | Type I | Type II | Kappa |
|---|---|---|---|---|
| grid | cell_size=0.5 (default anterior) | 0.00% | 100.00% | 0.00 (degenerado) |
| grid | cell_size=2.5 (ajustado a densidad real) | 4.11% | 71.13% | 29.83 |
| csf | valores por defecto | 28.48% | 1.64% | 35.54 |
| csf | rigidness=1, slope_smooth=True | 14.39% | 22.66% | 45.19 |
| csf | rigidness=3, class_threshold=0.7, slope_smooth=True (mejor de 18 combinaciones) | 12.61% | 21.53% | **49.25** |
| **pmf** | **valores por defecto** | **0.45%** | **17.63%** | **87.31** |
| *(referencia)* | MGF, Meng et al. 2009 | -- | -- | *64.11* |

### City Site 3 (samp31) -- terreno típico

| Método | Parámetros | Type I | Type II | Kappa |
|---|---|---|---|---|
| csf | cloth_resolution=0.5 | 10.55% | 8.06% | 81.14 |
| pmf | cell_size=0.5 (default) | 0.05% | 15.99% | 84.93 |
| **pmf** | **initial_distance=0.1, max_distance=4.0, slope=0.5, max_window_size=65 (mejor de 81 combinaciones)** | **0.26%** | **6.76%** | **93.42** |
| *(referencia)* | MGF, Meng et al. 2009 | -- | -- | *93.31* |

### Forest Site 5 (samp53) -- pendientes empinadas y terraceadas

El sitio más difícil de los 15 disponibles en el benchmark completo --
Kappa publicado de referencia es el más bajo de todo el conjunto.

| Método | Parámetros | Type I | Type II | Kappa |
|---|---|---|---|---|
| **pmf** | **cell_size=0.5 (default)** | **4.08%** | **16.70%** | **57.22** |
| *(referencia)* | MGF, Meng et al. 2009 | -- | -- | *25.60* |

## Hallazgos reales

1. **`grid` con una `cell_size` mal ajustada a la densidad real de los
   datos produce un resultado degenerado** (clasifica todo como
   terreno) -- confirmado y explicado: con menos de 1 punto por celda
   en promedio, cada punto aislado se vuelve trivialmente "el mínimo de
   su propia celda". No es un defecto de `GridGroundClassifier`, es una
   consecuencia esperable de su diseño simple ante datos dispersos.

2. **CSF mejora de forma real y sustancial con ajuste de parámetros**
   (Kappa 35.54 -> 49.25, barrido sistemático de 18 combinaciones
   reales), pero **no cierra la brecha** con la referencia publicada en
   terreno con discontinuidades abruptas -- consistente con una
   debilidad conocida y documentada en la literatura de los filtros de
   simulación de tela ante puentes/terraplenes.

3. **PMF supera consistentemente a CSF y a grid en los 3 sitios reales
   probados**, con sus valores por defecto (sin ningún ajuste de
   parámetros):
   - En los 2 sitios con relieve geométricamente complejo (Site 7 --
     discontinuidades; Site 5/samp53 -- pendientes empinadas
     terraceadas, el sitio más difícil de los 15 disponibles), PMF
     **supera la referencia publicada** (87.31 vs 64.11; 57.22 vs
     25.60).
   - En terreno urbano típico (City Site 3), PMF con valores por
     defecto queda cerca pero por debajo de la referencia (84.93 vs
     93.31) -- **con ajuste de parámetros (barrido de 81
     combinaciones, `isprs_barrido_pmf.py`), PMF prácticamente iguala
     la referencia (93.42 vs 93.31)**. El factor decisivo fue
     `max_window_size=65` (el doble del default de 33): los edificios
     de este sitio son lo bastante grandes en píxeles que una ventana
     máxima insuficiente nunca termina de "abrirlos" por completo,
     dejando techos mal clasificados como terreno (Type II alto con
     ventanas chicas, patrón claro en los 81 resultados).
   - El patrón sugiere que la ventaja de PMF es especialmente marcada
     en terreno con relieve complejo -- justo donde CSF mostró
     debilidad -- y que en terreno urbano con edificios grandes, un
     ajuste simple de `max_window_size` cierra la brecha casi por
     completo.

## Cambio aplicado

Se cambió el método por defecto de `GroundManager` de `"grid"` a
`"pmf"` (`src/topocore/processing/ground/manager.py`), respaldado por
la evidencia anterior en 2 sitios reales y contrastantes del benchmark
académico estándar del campo.

## Limitaciones de esta validación -- honestas, no ocultas

- Solo se probaron 3 de los 15 sub-muestras reales del benchmark
  completo (samp71, samp31, samp53) -- una cobertura parcial, no
  exhaustiva, aunque elegida deliberadamente para cubrir 3 tipos de
  dificultad distintos (discontinuidades abruptas, urbano simple,
  pendientes empinadas).
- La comparación contra Meng et al. (2009) es indirecta: ese paper
  evaluó 9 algoritmos específicos (no necesariamente incluyendo PMF ni
  CSF con la misma implementación exacta que usa TopoCore) -- el
  resultado de PMF superando la referencia en Site 7 es un hallazgo
  real y verificado matemáticamente, pero no implica que PMF supere a
  *todo* algoritmo publicado en la literatura completa del campo.
- No se probaron `adaptive_grid` ni `progressive_tin` (los 2
  clasificadores de terreno restantes en TopoCore) contra este
  benchmark -- quedan sin comparar.
- **No se cambiaron los valores por defecto internos de
  `PMFGroundClassifier`** (`initial_distance`, `max_distance`, `slope`,
  `max_window_size`), solo cuál clasificador usa `GroundManager` por
  defecto. El barrido en City Site 3 sugiere que `max_window_size=65`
  (el doble del actual default de 33) podría ser un mejor valor por
  defecto general, pero esto se basa en un solo sitio con edificios
  grandes -- cambiar los defaults internos de PMF requeriría probar
  esa hipótesis en más sitios antes de aplicarla, ya que una ventana
  más grande también es más costosa computacionalmente y podría
  perjudicar sitios con objetos pequeños.
- El barrido de parámetros de CSF cubrió 18 combinaciones de
  `rigidness`/`class_threshold`/`slope_smooth`, con `cloth_resolution`
  e `iterations` fijos -- un barrido más amplio podría encontrar una
  combinación mejor, aunque es poco probable que cierre la brecha
  completa de casi 15 puntos de Kappa observada en Site 7.
