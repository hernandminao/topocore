# TopoCore

Librería Python de procesamiento geoespacial para topografía y LiDAR:
levantamientos con código de campo, nubes de puntos (LAS/LAZ/E57),
clasificación de terreno, TIN/DTM, curvas de nivel, volúmenes de
corte/relleno, y exportación a DXF/GPKG/GeoTIFF.

## Validación

La clasificación de terreno (PMF -- Zhang et al. 2003) fue evaluada
contra el **ISPRS Filter Test** (Sithole & Vosselman, 2004), el
benchmark académico de referencia del campo, en 3 sitios reales de
dificultad contrastante. Resultados (índice Kappa de Cohen, comparado
contra Meng, Currit & Zhao, 2009):

| Sitio | Dificultad real | TopoCore (PMF) | Referencia publicada |
|---|---|---|---|
| Forest Site 5 | Pendientes empinadas terraceadas (el más difícil del benchmark) | 57.22 | 25.60 |
| Forest Site 7 | Puentes, pasos subterráneos, terraplenes | 87.31 | 64.11 |
| City Site 3 | Urbano, edificios complejos (parámetros ajustados) | 93.42 | 93.31 |

Ver `docs/validation/VALIDATION_ISPRS.md` para la metodología
completa, los 12 sitios restantes sin probar, y las limitaciones de
esta comparación.

Documentación adicional de validación:
- `docs/validation/VALIDATION_DATASETS.md` -- 5 datasets LiDAR reales
  probados (hasta 18M puntos), con métricas de escala y rendimiento.
- `docs/validation/VALIDATION_FAILURE_CASES.md` -- 3 casos de fallo
  reales, documentados con sus umbrales exactos.

## Instalación

```bash
pip install topocore
```

Dependencias opcionales:

```bash
pip install topocore[terrain]   # CSF (clasificación de terreno alternativa)
pip install topocore[geoid]     # GDAL (exportación GeoTIFF)
pip install topocore[ml]        # scikit-learn (clasificación multi-clase)
pip install topocore[dxf]       # ezdxf (exportación DXF)
```

## Inicio rápido

### Levantamiento topográfico (survey)

```python
from topocore.pipeline import SurveyPipeline, SurveyPipelineConfig, SurveyType
from topocore.survey.formats import SurveyFormat

config = SurveyPipelineConfig(
    survey_type=SurveyType.ROAD,
    survey_format=SurveyFormat.PNEZD,
    reference_code="EJE",
    output_dir="outputs/mi_via",
)
resultado = SurveyPipeline(config).run("levantamiento.csv")
```

### Nube de puntos (LiDAR)

```python
from topocore.io.las.reader import LASReader
from topocore.processing.ground.manager import GroundManager

nube = LASReader("terreno.las").read()
terreno = GroundManager(method="pmf", cell_size=0.5).extract(nube)
```

## Limitaciones conocidas

- El método de clasificación de terreno por defecto es PMF, no CSF
  (cambiado tras validación empírica -- ver `VALIDATION_ISPRS.md`).
- El costo de CSF/PMF escala con el **área geográfica** cubierta, no
  con la cantidad de puntos -- ver `VALIDATION_FAILURE_CASES.md` para
  el umbral exacto y cómo elegir `cell_size`/`cloth_resolution`.
- No existe teselado (tiling) espacial automático -- archivos de
  varios GB con área geográfica muy grande pueden necesitar
  preprocesamiento externo.
- Los clasificadores multi-clase (Random Forest, XGBoost, LightGBM)
  requieren entrenamiento con datos propios -- no se incluye un
  modelo pre-entrenado.

## Licencia

MIT
