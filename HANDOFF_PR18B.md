# PR18B — LandXML IO — Handoff de esta sesión

**Estado:** implementado y probado en sandbox. Pendiente: que Hernán aplique
estos archivos a su repo real y corra la suite completa (415 tests previos +
50 nuevos) junto con `ruff`/`mypy` sobre el repo real, ya que este sandbox
solo tenía `src/`, sin `tests/` ni `pyproject.toml`.

## Qué se entrega

```
src/topocore/terrain/tin.py         MODIFICADO: se agregó TIN.from_mesh()
src/topocore/io/landxml/            NUEVO paquete completo (PR18B)
tests/terrain/test_tin_from_mesh.py NUEVO: 15 tests
tests/io/landxml/                   NUEVO: 35 tests
```

**50/50 tests pasando (dos corridas consecutivas), `ruff check`, `ruff format
--check` y `mypy --disallow-untyped-defs` limpios** en todos los archivos
tocados/creados. No se corrió sobre el resto del repo (regla del proyecto).

## Decisión que abrió esta sesión: `TIN.from_mesh()`

Congelada y aprobada explícitamente por Hernán. `TIN.from_points()` (el único
constructor previo) siempre recalcula una triangulación Delaunay nueva —
inservible para representar un `<Surface>` de LandXML, cuya `<Faces>` casi
nunca es Delaunay puro (breaklines, bordes recortados, ediciones manuales).

`TIN.from_mesh(vertices, simplices)`:
- Nunca llama `scipy.spatial.Delaunay`.
- Construye `neighbors` reutilizando la misma idea de adyacencia por arista
  que ya usa `constrained_delaunay._build_mesh()` (no es un algoritmo nuevo).
- Sigue la convención de `scipy.spatial.Delaunay.neighbors`:
  `neighbors[i, j]` es el vecino opuesto al vértice `j`.
- Valida: vértices/simplices no vacíos, shape `(n, 3)`, índices en rango,
  sin vértices duplicados por triángulo, sin triángulos degenerados
  (área XY ≈ 0 con tolerancia `terrain.constants.EPSILON`).
- `from_points()` queda intacto — cero cambios a la API existente.

## Paquete `topocore.io.landxml`

Contrato congelado con Hernán (ver transcript de la sesión):

- Ubicación: `topocore.io.landxml` (decisión final, revirtiendo la ubicación
  `topocore.landxml` que constaba en el documento de contexto anterior — el
  cambio fue explícitamente reconsiderado y confirmado por Hernán en esta
  sesión, no un descuido).
- Alcance: solo `<Surfaces>`/TIN y `<CgPoints>`. `<Alignments>`, `<Profile>`,
  `<Feature>` embebido en `<CgPoint>` quedan fuera — su presencia en un
  archivo no rompe la lectura, simplemente no se representan.
- Validación: semántica propia (mismo patrón que `dxf.validation`/
  `gpkg.validation`), sin XSD externo, sin `lxml`. `LandXMLValidator` es
  usable de forma independiente (`validate_xml` sobre el árbol crudo,
  `validate_document` sobre el `LandXMLDocument` ya construido). El Writer
  llama `validate_document` antes de tocar disco.
- Modelos nuevos (mínimos, sin duplicar dominio): `NamedSurface`,
  `NamedPointGroup`, `LandXMLDocument`, `LinearUnit`.
- Convención de coordenadas: LandXML es siempre `"north east elev"` →
  `Point3D(x=east, y=north, z=elev)`. Aislada en `coordinates.py`, nunca
  repetida inline en reader/writer.
- Dependencia: `xml.etree.ElementTree` de la librería estándar. Ninguna
  dependencia nueva.
- Excepciones: `LandXMLError(TopoCoreError)` → `LandXMLParseError`,
  `LandXMLValidationError`, `LandXMLWriteError`.

## Roadmap — candidato futuro registrado, sin numerar

Por decisión explícita, **no se asignó número de PR** a `<Alignments>`/
`<Profile>` (a diferencia de la propuesta inicial de Hernán de `PR18C`/
`PR18D`), porque implican un dominio vial nuevo por completo (geometría
horizontal, estacionamiento, curvas verticales) que aún no se ha auditado.
Queda anotado como intención de largo plazo, no como forma congelada:

```
CANDIDATO FUTURO (sin auditar, sin número de PR asignado)
Dominio de Alineamiento Vial (topocore.alignment o similar)
  - Motivación: soporte LandXML <Alignments>/<Profile>, competir con
    Civil 3D / OpenRoads / Trimble Business Center en diseño vial.
  - Requiere auditoría de dominio dedicada antes de asignar número de PR.
  - No bloquea PR18B, PR19, PR20.
```

`<Feature>` embebido en `<CgPoint>` queda con el mismo problema de fondo que
resolvimos con `NamedSurface`/`NamedPointGroup`: `SurveyPoint` está congelado
y no tiene dónde meter propiedades arbitrarias — cuando se aborde, necesitará
un wrapper similar, no una extensión de `SurveyPoint`.

## Siguiente paso sugerido para el próximo chat

1. Aplicar estos archivos al repo real, correr la suite completa (415 + 50)
   y `ruff`/`mypy` sobre todo lo tocado en PR18B únicamente.
2. Si algo del código real diverge de este sandbox (por cambios que Hernán
   haya hecho fuera de sesión), auditar antes de continuar — no asumir.
3. Congelar PR18B en el roadmap y decidir si el siguiente paso es PR19
   (QA/Validation) o PR20 (Optimization), según lo que ya estaba planeado
   antes de esta sesión.

## Actualización — PR18C: dominio Alignment/Profile + integración LandXML

Auditado, diseñado y cerrado en esta misma sesión (después de PR18B):

### `topocore.alignment` (dominio nuevo, primer nivel, sin número de PR asignado
formalmente hasta que Hernán lo confirme)

```
src/topocore/alignment/
├── __init__.py
├── elements.py            LineElement, ArcElement, SpiralElement (tipos bajos, sin
│                           dependencia de algorithms -- evita ciclo con dispatch)
├── vertical_elements.py    GradeSegment, VerticalCurve (Hickerson 1964)
├── exceptions.py
├── models.py               Alignment, DesignProfile (orquestación)
└── algorithms/
    ├── __init__.py
    ├── horizontal.py       station_to_point(), dispatch Line/Arc/Spiral
    └── spiral.py           evaluate_spiral(), curvature_at() (Fresnel)
```

**Decisiones matemáticas cerradas, verificadas contra fuentes externas independientes:**

- `TIN.from_mesh()` -- extensión aprobada a `terrain.tin` (constructor alternativo,
  no reemplaza `from_points()`), preserva conectividad `<Faces>` no-Delaunay.
- `SpiralElement` -- clotoide real vía `scipy.special.fresnel` (no aproximación en
  serie). Transformación rígida local→global derivada una vez (no ajustada punto a
  punto). Valida en construcción: consistencia de cuerda Y consistencia de `PI`
  (`PI = intersección de tangentes`, fórmula `PI_local = u(L) - v(L)*cot(θs)`,
  verificada por dos métodos independientes). **CERRADO.**
- `VerticalCurve` -- **modelo Hickerson (1964)**, no "parábola única continua"
  (esa fue una primera implementación incorrecta, corregida tras verificación
  contra dos ejemplos resueltos publicados de forma independiente, PDHonline
  Course L121). `PVI` es la intersección de tangentes, generalmente **fuera** de
  la curva real. Curvatura discontinua en `CVC` para el caso asimétrico
  (`rate_in ≠ rate_out`) -- propiedad documentada del método, no un defecto.
  **CERRADO.**

### `topocore.io.landxml` -- extendido con `<Alignments>`

```
src/topocore/io/landxml/
├── codecs.py    NUEVO: parse/format_radius (INF), parse/format_rotation (cw/ccw),
│                parse/format_station_elevation
├── models.py    + NamedAlignment
├── reader.py    + lectura de <CoordGeom>/<Profile>/<ProfAlign>
├── writer.py    + escritura correspondiente
├── validation.py + validación estructural de <Alignment>/<Line>/<Curve>/<Spiral>
└── constants.py  + CRV_TYPE_ARC, SPI_TYPE_CLOTHOID
```

Alcance: `crvType="arc"` y `spiType="clothoid"` únicamente -- cualquier otro valor,
`<Curve>` sin `<Center>`, o espiral compuesta curva-curva (ambos radios finitos) se
**salta con advertencia explícita** (identificando `Alignment`/elemento, y para el
caso de espiral compuesta, los valores `radiusStart`/`radiusEnd`), nunca en silencio
ni como error duro.

**Cambio a código de PR18B:** `DEFAULT_COORDINATE_PRECISION` subido de 8 a 10
decimales. Causa: la tolerancia absoluta de `SpiralElement` (`1e-9`) es más
estricta que el error de redondeo introducido por 8 decimales en texto (~3e-9).
Verificado que no rompió ningún test existente de `<Surfaces>`/`<CgPoints>`.

**Estado final de la sesión: 193/193 tests, dos corridas consecutivas, `ruff`/`mypy`
limpios en todo el árbol tocado.**

### Pregunta abierta para Fase 4 (real, no resuelta -- NO decidir sin Hernán)

Confirmado empíricamente que el error de redondeo de 8 decimales (~3e-9) es
**independiente de la magnitud de la coordenada** (formato `%.Nf` es por posición
decimal, no por cifras significativas) -- por lo tanto un archivo real de
Civil3D/TBC exportado con 8 decimales (la precisión típica auditada al inicio de
PR18B) tiene el mismo riesgo de ser rechazado falsamente por la validación de
consistencia de cuerda de `SpiralElement`, sin importar si usa coordenadas UTM,
Plano Estatal, o locales. Dos caminos posibles, ninguno decidido:

1. Tolerancia relajada solo en el Reader de LandXML al validar `SpiralElement`
   durante importación (sin tocar `SpiralElement` ni `DEFAULT_MATH_CONFIG`).
2. Revisar si `DEFAULT_MATH_CONFIG.absolute_tolerance=1e-9` es demasiado estricta
   para datos geométricos reales en general (alcance mucho mayor).

**No se ha tocado ningún código para resolver esto** -- queda pendiente de que
Hernán decida, ya que ambos caminos tocan código congelado o configuración
compartida por todo el proyecto.


Antes de aceptar el PR como definitivo se hizo una auditoría del propio
código (`reader.py`, `coordinates.py`) buscando errores de arquitectura y
casos LandXML no cubiertos por los 50 tests originales. Se encontraron y
corrigieron 3 problemas reales:

1. `<Surface>` sin `<Definition>` se descartaba en silencio (violaba la
   regla de "nunca fallar en silencio") -> ahora genera warning en el
   reporte.
2. Bug de precedencia en `_read_units_and_crs`: si coexistían `<Metric>` e
   `<Imperial>`, `Imperial` siempre ganaba de forma incondicional, no por
   una decisión explícita -- oculto porque no había ningún test de
   `linear_unit`/`crs` en round-trip.
3. `pntRef` (punto por referencia, no soportado) daba un mensaje de error
   genérico en vez de nombrar la causa real.

Se agregaron 8 tests nuevos (`tests/io/landxml/test_gaps_audit.py`)
cubriendo: múltiples superficies/grupos con ids de punto reutilizados
entre grupos, round-trip de `linear_unit` y `crs`, `Surface` sin
`Definition`, `Pnts` vacío, ids alfanuméricos no secuenciales, y el
mensaje explícito de `pntRef`.

**Estado final: 59/59 tests, dos corridas consecutivas, `ruff check`,
`ruff format --check` y `mypy --disallow-untyped-defs` limpios.**

Se verificó también, con test explícito, que el namespace LandXML con
prefijo (`<landxml:LandXML xmlns:landxml="...">`) se maneja igual que el
namespace por defecto -- `local_tag()` normaliza ambos vía la notación
Clark de ElementTree.

### Pendiente para la auditoría de integración en el repo real (los 4 pasos
que Hernán definió)

No verificable en este sandbox porque no incluía `tests/` del repo real:

- Suite completa del repo real (415 tests previos + 59 nuevos = 474) y
  `ruff`/`mypy` sobre el repo completo, no solo los archivos tocados en
  PR18B.
### Actualización — auditoría de precisión (misma sesión, después de integrar Alignments)

Hernán decidió: **no tocar `DEFAULT_MATH_CONFIG`** (tolerancia matemática interna
del proyecto, `abs_tol=1e-9`) y **no ajustar coordenadas en silencio** para que
pasen la validación. Planteó la pregunta arquitectónica de si la tolerancia de
importación de LandXML debería vivir separada de la tolerancia del dominio.

**Auditoría sintética ejecutada** (no solo con `SpiralElement` -- también con
`ArcElement`, que tiene el mismo tipo de chequeo `distance(center,start)==radius`):

```
precisión ≤ 8 decimales  →  RECHAZADO (Arc y Spiral, ambos)
precisión ≥ 9 decimales  →  ACEPTADO
```

Umbral limpio y determinista, **independiente de la magnitud de la coordenada**
(probado en pequeña, UTM~500,000, Plano Estatal~2,000,000). El primer intento de
prueba de `ArcElement` pasó por accidente (geometría alineada a ejes donde el
redondeo se cancelaba exactamente) -- con geometría genuinamente no alineada, el
mismo problema aparece igual que en `SpiralElement`.

Nuestro propio `LandXMLWriter` ya está a salvo (10 decimales, un decimal de
margen). El riesgo real es exclusivamente al **leer archivos externos** con 8
decimales (la precisión típica auditada al inicio de PR18B).

**Decisión de Hernán sobre el camino a seguir:** necesita ver un archivo real de
Civil3D/TBC antes de elegir entre (a) reconciliación transparente en el Reader
(recalcular desde radius/length cuando la discrepancia cae en una tolerancia IO
explícita, reportado siempre, nunca silencioso) o (b) rechazo estricto exigiendo
que el archivo se corrija en origen.

**Intento de conseguir un archivo real vía búsqueda: sin éxito.** La página
oficial de muestras de landxml.org (`landxml.org/webapps/landxmlsamples.aspx`)
bloquea acceso automatizado vía `robots.txt`; ninguna otra fuente pública
encontrada ofrecía un `.xml` real descargable con `<Alignments>`/`<Spiral>`
genuinos. **Esta es ahora una dependencia bloqueante real: se necesita que
Hernán aporte un archivo real** (proyecto propio o export de su software de
campo) antes de que se pueda resolver esta pregunta con evidencia, no
supuestos.

**No se implementó ninguna de las dos opciones.** `SpiralElement`/`ArcElement`
y `DEFAULT_MATH_CONFIG` quedan exactamente como estaban. La integración
`<Alignments>`/`<Profile>` de LandXML queda funcionalmente completa y probada
(193/193 tests) pero **no se marca cerrada** -- exactamente el estado que
Hernán definió: `🚧 revisión` pendiente de esta cuestión de tolerancia.

## Actualización final — bug real corregido con archivo PLATEIA, sesión cerrada

**Segundo archivo real subido: PLATEIA 2007** (software de diseño vial esloveno),
`Sample_Plateia2007LandXML11.XML`, con 6 `<Spiral>` genuinos (0 en el archivo de
Civil3D). Auditoría igual de rigurosa que con Civil3D:

### Tolerancia LandXML fijada en `2e-6` (no `1e-8`)

Causa raíz identificada con precisión: **no es una propiedad de `Spiral` vs.
`Arc`** -- es la cantidad de decimales que usa cada software al exportar
(PLATEIA: 6 decimales; Civil3D: 8-10). El error de redondeo esperado por
truncamiento decimal explica cuantitativamente ambas magnitudes observadas
(Civil3D `~8e-9` con 8-10 decimales; PLATEIA `~1e-6` con 6 decimales).
Hernán decidió: **un solo valor fijo `2e-6`, sin derivarlo automáticamente
de los decimales del archivo** (el conteo de decimales no garantiza la
incertidumbre geométrica real).

### Bug real encontrado y corregido en `SpiralElement` (Entrega 2, ya "cerrada")

Al validar la espiral real "PREHODNICA 2" (espiral de **salida**:
`radius_start` finito, `radius_end=INF`), el `PI` esperado difería del real
por **~5 metros** -- una discrepancia que no se explica por redondeo.
Investigación:

1. Se calibró la convención angular del archivo con una `<Line>` conocida
   (coincide con `atan2` estándar, CCW desde este).
2. Se verificó que el `PI` real coincide (0.8mm) con la intersección de
   tangentes usando `dirStart`/`dirEnd` propios de PLATEIA -- confirma que
   el dato del archivo es correcto, el bug está en nuestra fórmula.
3. Se aisló la causa: el marco local canónico define `+u` como la dirección
   de `l'` creciente. Para una espiral de **entrada**, `l'` crece en el
   mismo sentido que el recorrido físico (start→end). Para una espiral de
   **salida**, `l'` crece hacia `start` -- sentido **contrario** al
   recorrido físico -- por lo que el espejado de `clockwise` debía
   invertirse. El chequeo de cuerda nunca lo detectó porque `hypot(u,v)` es
   insensible al signo de `v`; solo el rumbo/`PI` lo revela.

**Corregido en `topocore/alignment/elements.py`
(`SpiralElement.__post_init__`) y `topocore/alignment/algorithms/spiral.py`**
(`_rigid_transform`, `evaluate_spiral`; `curvature_at` NO necesitó cambio,
su convención de signo ya era correcta independiente de entrada/salida).
Verificado hasta la millonésima de metro contra el `PI` real, y con
verificación cruzada independiente contra `dirStart` propio del archivo
(no derivado de nuestra fórmula). Los 110 tests previos de
`topocore.alignment` (Entrega 1+2) se re-verificaron intactos; un fixture
sintético de Entrega 2 (`_exit_ccw()`) tenía un `pi` que solo "pasaba" por
el bug viejo -- corregido con el valor recalculado correctamente.

### Hallazgo aislado, documentado, sin corregir (por decisión explícita)

De las 10 uniones entre elementos del único `<Alignment>` del archivo
PLATEIA, **9 coinciden exactamente** (texto idéntico byte a byte) y
**1 difiere en `~1e-6`** (`PREHODNICA 4`→`PREHODNICA 5`, probable cero
final recortado en la exportación). No es un patrón sistemático como el de
Civil3D (81% de los arcos afectados) -- es una anomalía aislada. Hernán
decidió **no** extender `Alignment.__post_init__` con tolerancia de LandXML
por un solo caso sin evidencia adicional de que sea sistémico. Queda
documentado como posible error del archivo, `Alignment` sin tocar.

**Estado final de la sesión: 218/218 tests, dos corridas consecutivas,
`ruff`/`mypy` limpios en todo el árbol tocado.**

### Fixtures reales incluidas en este paquete

- `tests/io/landxml/fixtures/GSG_features_alignments.xml` -- Civil3D 2007
  real (47 `<Curve>`, 0 `<Spiral>`, 5 `<CgPoints>` incl. 1 sin `name` y
  puntos `pntRef`).
- `tests/io/landxml/fixtures/Sample_Plateia2007LandXML11.XML` -- PLATEIA
  2007 real (2 `<Line>`, 3 `<Curve>`, 6 `<Spiral>`, `<Profile>`/`<ProfAlign>`
  con `<ParaCurve>` simétricas y asimétricas reales).

## Criterio de cierre -- reafirmado, no relajado

Ninguna de las dos entregas (PR18B, PR18C) se marca `CONGELADO` solo porque
el sandbox tenga 218/218. El criterio sigue siendo exactamente el que
Hernán definió desde el principio: aplicar al repo real → suite completa →
`ruff` → `ruff format --check` → `mypy` → segunda corrida de `pytest` →
auditoría funcional con los dos archivos LandXML reales (Civil3D y
PLATEIA) → **solo entonces** `CONGELADO`.




Estado correcto mientras no se completen las 4 fases de integración que
definió Hernán:

> PR18B implementado, auditado y validado en sandbox; pendiente de
> integración y validación global en el repositorio real.

```
Fase 1 — aplicar terrain/tin.py, io/landxml/, tests/terrain/,
         tests/io/landxml/, HANDOFF_PR18B.md al repo real
Fase 2 — pytest sobre el repo completo (no asumir 474+59=533 sin correrlo:
         puede haber tests ya modificados/integrados fuera de esta sesión)
Fase 3 — ruff check . / ruff format --check . / mypy . SOBRE TODO EL REPO,
         no acotado a los archivos de PR18B
Fase 4 — validación funcional con archivos LandXML reales (ver abajo)
```

Solo al pasar las 4 fases: `PR18B = CONGELADO`.

### Fase 4 — asimetría importante entre los 8 casos de validación

De los 8 casos que definió Hernán, **7 ya están cubiertos por tests
sintéticos** de esta sesión (breaklines/Faces no-Delaunay, unidades feet,
CRS, múltiples superficies, múltiples grupos de puntos, IDs alfanuméricos)
-- sandbox y repo real deberían comportarse igual ahí, porque no dependen
de nada externo al código.

Los otros 2 -- **`LandXML Civil 3D → TopoCore → TIN`** y
**`TopoCore → Civil 3D`** -- son cualitativamente distintos: son la única
validación de la lista que requiere un archivo LandXML real exportado por
software de terceros, no uno escrito a mano basándose en documentación
auditada (MicroSurvey/Bentley/Autodesk). Es el punto ciego real que queda
sin cerrar: un `.xml` real de Civil3D/TBC podría traer variaciones de
formato (atributos extra no anticipados, orden de elementos, particularidades
de exportación de esa versión específica del software) que ningún test
sintético puede cubrir por no haberse auditado contra un archivo real.

**Actualización: este punto ciego SÍ se cerró en esta misma sesión.** Se
consiguieron y auditaron dos archivos reales (Civil3D 2007 con 47 `<Curve>`
reales, PLATEIA 2007 con 6 `<Spiral>` reales) -- ver la sección "Actualización
final" más arriba para el detalle completo, incluyendo un bug real
encontrado y corregido en `SpiralElement` que ningún test sintético había
detectado. Lo que queda pendiente ya no es "conseguir un archivo real" sino
exactamente las 4 fases de integración en el repo real de Hernán (aplicar
código, suite completa, `ruff`/`mypy`, segunda corrida) -- el sandbox no
puede sustituir eso.

## Corrección post-entrega: `_clothoid_local_uv` y stubs de tipo de scipy

Hernán reportó un error de `mypy` en su repo real que **no apareció en el
sandbox**: `_clothoid_local_uv()` (en `topocore/alignment/elements.py`)
devolvía `scale * c`/`scale * s` sin forzar `float()`, y en un entorno con
stubs de tipo más completos para `scipy.special` (a diferencia del sandbox,
donde `fresnel()` caía como `Any` por el comentario
`# mypy: disable-error-code=import-untyped`), mypy detecta correctamente
que el retorno es `np.floating`, no `float` nativo -- incompatible con la
firma declarada `-> tuple[float, float]`.

**Corregido**: `u = float(scale * c)`, `v = float(scale * s)`. Verificado
que `fresnel()` solo se llama en ese único lugar del proyecto (no hay otro
sitio con el mismo problema). 218/218 tests siguen pasando en el sandbox
tras el cambio.

**Nota para la integración en el repo real**: esto sugiere que el entorno
de Hernán tiene stubs de `scipy` más completos que el sandbox -- vale la
pena, durante la Fase 3 (`mypy .` sobre el repo completo), estar atento a
si aparecen más discrepancias de este tipo (valores de `numpy`/`scipy` sin
forzar a `float`/`int` nativos) en otros módulos que el sandbox no pudo
detectar por tener stubs más laxos.

## Segunda corrección post-entrega: `GradeSegment.grade_at`

Se encontraron dos problemas relacionados con `GradeSegment.grade_at` al
integrar en el repo real de Hernán, ninguno presente en el sandbox tal como
se entregó originalmente:

1. **Firma incompleta**: en la copia de Hernán, `grade_at(self) -> float`
   había perdido el parámetro `station` (probablemente al limpiar el
   comentario `# noqa: ARG002` durante la integración) -- rompía la unión de
   tipos `VerticalElement = GradeSegment | VerticalCurve` en mypy
   ("Too many arguments for grade_at of GradeSegment" al llamar
   `current.grade_at(current.end_station)` en `DesignProfile.__post_init__`).

2. **Advertencia de parámetro no usado** (Pylance/SonarQube S1172): el
   parámetro `station` es requerido por la interfaz compartida con
   `VerticalCurve.grade_at` (que sí lo usa), pero `GradeSegment.grade_at`
   no lo necesita (la pendiente es constante). El comentario `# noqa:
   ARG002` solo silenciaba `ruff`, no Pylance ni SonarQube.

**Corregido con la convención de prefijo `_`** (`_station: float`),
reconocida como "no usado intencionalmente" por ruff, Pylance y SonarQube
a la vez, sin necesitar un comentario de supresión distinto por
herramienta. Todas las llamadas a `grade_at()` en el proyecto son
posicionales, así que el cambio de nombre no rompe nada.

218/218 tests siguen pasando en el sandbox tras el cambio.

## Tercera corrección post-entrega: cobertura de test para el archivo PLATEIA completo

Hernán preguntó si quedaban pruebas reales de interoperabilidad para
Civil3D, PLATEIA y campo. Auditoría honesta reveló un hueco: Civil3D tenía
un test de lectura end-to-end del archivo completo
(`test_real_civil3d_file.py`), pero **PLATEIA no** -- la fixture
`Sample_Plateia2007LandXML11.XML` estaba copiada en
`tests/io/landxml/fixtures/` sin que ningún test la leyera vía
`LandXMLReader`. Si se hubiera intentado, habría fallado por la anomalía de
continuidad ya documentada (`PREHODNICA 4 → PREHODNICA 5`, `~1e-6`,
decisión explícita de Hernán de no tocar `Alignment` por un caso aislado).

**Agregados 2 tests nuevos** en `test_real_civil3d_file.py`:
- `test_real_plateia_file_currently_fails_on_known_discontinuity` -- documenta
  el comportamiento ACTUAL (falla) como intencional, no como hueco silencioso.
  Si algún día se resuelve la anomalía o se extiende `Alignment` con
  tolerancia, este test empieza a fallar -- señal inequívoca de que la
  decisión cambió, en vez de quedar como una omisión sin detectar.
- `test_real_plateia_file_error_is_cleanly_wrapped_not_a_traceback` --
  confirma que aun fallando, el error sigue siendo `LandXMLParseError`
  limpio con la causa encadenada, no un traceback crudo.

No hay ningún otro dato "de campo" (GNSS RTK, software propio de Hernán,
etc.) usado en esta sesión -- solo los dos archivos reales subidos
(Civil3D, PLATEIA).

**Estado final: 220/220 tests, dos corridas consecutivas, `ruff`/`mypy`
limpios.**

## PR19 — alcance redefinido (decisión de Hernán, no ejecutado en esta sesión)

Corrección importante de Hernán: **PR19 no debe limitarse a LandXML.** El
roadmap original ya lo definía como QA/Validation general; esta sesión, al
ir encontrando hallazgos reales de LandXML, corría el riesgo de reducirlo
implícitamente a "QA de LandXML". Estructura definitiva que Hernán fijó:

```
PR19 — QA / Validation (transversal a todo TopoCore)
│
├── 19.1  Test Architecture
├── 19.2  Regression Suite       (todo bug real PR1-PR18 -> test permanente)
├── 19.3  Numeric Validation     (tolerancias, precisión, unidades, CRS,
│                                  transformaciones, geometría -- el tipo
│                                  de bug que SpiralElement mostró: tests
│                                  sintéticos pasaban, archivo real reveló
│                                  un problema de orientación)
├── 19.4  IO Validation
│   ├── LAS/LAZ
│   ├── PLY
│   ├── E57
│   ├── ASCII
│   └── LandXML  <- esta sesión (PR18B/PR18C) alimenta esta sub-fase,
│                    no es el objetivo completo de PR19
├── 19.5  Geodesy Validation
├── 19.6  Terrain Validation
├── 19.7  Processing Validation
├── 19.8  Analysis Validation
├── 19.9  Export Validation
├── 19.10 Workflow Validation
├── 19.11 Real-world Interoperability (Civil3D, TBC, PLATEIA, GNSS/campo)
└── 19.12 Final Quality Gate
```

**Regla de clasificación de hallazgos, para que PR19 no se convierta en
"arreglar todo lo que se encuentre" (otro PR18):**

```
AUDITAR -> CLASIFICAR
  CRÍTICO              -> corregir en PR19
  IMPORTANTE           -> corregir si pertenece genuinamente a QA
  FUNCIONALIDAD NUEVA  -> siguiente PR, no PR19
  CAMBIO ARQUITECTÓNICO -> registrar, no tocar en PR19
```

No se ha empezado ningún trabajo de PR19 en esta sesión -- queda como
decisión de alcance registrada para cuando se inicie.

## PR19 arranca — hallazgo crítico: el repo real tiene 0 tests

Hernán confirmó que el volcado de `src/` que sí sobrevivió a las 379
archivos `.py` reales (todo el trabajo de PR14-PR18T), pero **la carpeta
`tests/` no existe o está vacía en su repositorio real** -- los "415/415
tests pasando" que describía el documento de contexto original vivieron
únicamente en sandboxes efímeros de sesiones anteriores de Claude, que
nunca se guardaron de vuelta al repositorio real.

Esto redefine PR19: no es refinar cobertura existente, es **construir la
primera red de regresión real** que este código ha tenido. Hernán definió
la estructura completa de PR19 (12 sub-fases, transversal a los 23 módulos,
con regla de clasificación de hallazgos CRÍTICO/IMPORTANTE/FUNCIONALIDAD
NUEVA/CAMBIO ARQUITECTÓNICO para que no se vuelva otro PR18) -- ver sección
"PR19 -- alcance redefinido" más arriba.

### Primer módulo: `geodesy` (transformaciones de coordenadas)

Prioridad de Hernán: el riesgo más crítico posible (una transformación
incorrecta pone objetos en el lugar equivocado del mundo real, en
silencio). Auditado `transformer.py` -- su propio docstring documenta
verificación previa contra la fórmula EPSG 9606 y geometría derivable a
mano, que se rehizo aquí desde cero.

**Hallazgo durante la verificación (no un bug del código real):** un
primer intento de fórmula EPSG 9606 independiente usó el signo de rotación
equivocado (la clásica confusión Position Vector vs. Coordinate Frame
Rotation) y discrepaba del pipeline real hasta en ~40m/~0.3m. Aislando el
paso `+proj=helmert` puro de PROJ (sin conversión geográfica de por medio)
y probando cada uno de los 7 parámetros por separado, se confirmó: tx/ty/tz/
scale coincidían exactamente; solo las 3 rotaciones diferían, las tres con
el mismo signo invertido. Esto localizó el error en la fórmula de
verificación, no en `topocore.geodesy.transformer` (que nunca reimplementa
la matriz de rotación -- solo arma un pipeline de PROJ y deja que PROJ lo
ejecute). Con el signo corregido, la fórmula independiente coincide con el
código real a precisión doble exacta (`diff=0.0`), confirmado con 3 puntos
no degenerados y los 7 parámetros de Helmert combinados a la vez.

**`tests/geodesy/test_transformer_from_operation.py` (12 tests, todos
pasando):**
- `IDENTITY` delega al constructor normal.
- Traslación pura y rotación `rz` pura verificadas con geometría derivable
  a mano en `(lon=0, lat=0)`.
- 3 casos no degenerados con los 7 parámetros de Helmert combinados,
  verificados contra la fórmula EPSG 9606 corregida.
- Los 3 casos explícitamente no soportados (documentados en el propio
  docstring del código): `GRID_SHIFT`, Helmert de 14 parámetros
  (dependiente del tiempo), Helmert entre CRS proyectados -- cada uno con
  su test de regresión, tal como pedía el propio docstring
  ("`test_transformer_from_operation.py` for one regression test per row
  of this table").
- Validación de construcción de `CoordinateOperation` (Helmert sin
  parámetros, Identity con parámetros -- ambos deben rechazarse).

**Estado: 232/232 tests en todo el sandbox (dos corridas), `ruff`/`mypy`
limpios** (acotado con `--follow-imports=silent` para no arrastrar
advertencias preexistentes de otros módulos del repo real que esta sesión
no tocó -- `scipy` sin stubs en `processing/ground/pmf.py`,
`features/_shared.py`, `features/buildings/roofs.py`, ya presentes en el
código fuente original, no introducidas aquí).

Se instaló `pyproj` en el sandbox (no estaba presente) para poder verificar
contra objetos reales en vez de mocks -- coherente con la metodología de
toda la sesión.

**Pendiente:** el resto de `geodesy` (20 archivos: `crs.py`, `validation.py`,
`geodesic.py`, `utm.py`, `accuracy.py`, `_cache.py`, etc.) no auditado
todavía. `transformer.py`/`operation.py`/`helmert.py`/`operation_type.py`
son los únicos con cobertura nueva hasta ahora.

## `geodesy` -- continuación: dos bugs reales encontrados

### Bug 1: UTM zone 61 en `longitude=180.0` (código real)

`UTMZone.from_latlon()` daba `zone_number=61`, `epsg=32661` en
`longitude=180.0` exacto -- valor que `validate_lat_lon` permite
explícitamente (rango `[-180,180]` inclusivo) pero que no corresponde a
ninguna zona UTM real (válidas: 1-60) ni a ningún código EPSG real.
Aritmética propia, no de `pyproj`. **Corregido**: `zone_number = min(...,
60)`, mismo comportamiento que ya tenía el borde occidental (zona 1 en
`longitude=-180.0`). 46 tests nuevos en `test_utm.py`, incluyendo Noruega,
Svalbard, bandas de letra MGRS y el caso de regresión exacto.

### Bug 2: arquitectura de tests -- colisión de nombres (hallazgo de
arquitectura, no de dominio)

Al correr la suite completa por primera vez con los nuevos tests de
`geodesy`, aparecieron dos problemas reales de estructura:
1. `tests/geodesy/test_validation.py` chocaba con
   `tests/io/landxml/test_validation.py` (mismo basename, ninguna carpeta
   tenía `__init__.py` -- pytest no podía distinguirlos).
2. Al agregar `__init__.py` a `tests/io/`, el nombre `io` **choca con el
   módulo estándar de Python** (`io` es parte de la librería estándar) --
   convertir esa carpeta en paquete rompía la recolección de tests.

**Corregido**: `__init__.py` en `tests/alignment/`, `tests/geodesy/`,
`tests/terrain/`, `tests/io/landxml/` -- pero **no** en `tests/io/` (para
no chocar con el `io` de la librería estándar; el nombre que necesita ser
único es `landxml`, no `io`). Esto es en sí mismo la primera pieza real de
PR19 19.1 (Test Architecture): **nunca nombrar una carpeta de test igual
a un módulo de la librería estándar de Python** (`io`, `math`, `types`,
etc.), y cada carpeta de test con archivos propios necesita `__init__.py`
para namespacing correcto entre módulos.

### Estado del módulo `geodesy`

```
transformer.py (from_operation)   ✅ 12 tests, EPSG 9606 verificado
operation.py                      ✅ validación cubierta
helmert.py / operation_type.py    ✅ cubiertos indirectamente
utm.py                            ✅ 46 tests, 1 bug real corregido (zona 61)
geodesic.py                       ✅ 15 tests, verificado contra pyproj.Geod
                                      aislado + geometría analítica
validation.py                     ✅ 30 tests
crs.py                            ⏳ sin test dedicado todavía (cubierto
                                      indirectamente por los demás)
_cache.py, accuracy.py,
local_crs.py, vertical_datum.py,
datum.py, projection.py,
transform.py (integración con
survey/point_cloud/feature_
collection)                       ⏳ sin auditar todavía
```

**Estado final: 323/323 tests en todo el sandbox (dos corridas
consecutivas), `ruff`/`mypy` limpios** (acotado con `--follow-imports=silent`
para no arrastrar advertencias preexistentes de módulos no tocados en esta
sesión).

## `geodesy` -- módulo completo (20/20 archivos con cobertura)

```
transformer.py     ✅ 12 tests  (EPSG 9606 verificado, bug de firma en verificación propia corregido)
operation.py        ✅ cubierto indirectamente
helmert.py / operation_type.py  ✅ cubiertos indirectamente
utm.py               ✅ 46 tests  (bug real corregido: zona 61 en longitude=180.0)
geodesic.py           ✅ 15 tests  (pyproj.Geod aislado + geometría analítica)
validation.py          ✅ 30 tests
transform.py            ✅ 18 tests  (transform_survey/transform_feature_collection/
                                       transform_point_cloud -- verificados cruzados
                                       entre sí para el mismo punto físico)
crs.py                   ✅ 18 tests
accuracy.py/local_crs.py/
vertical_datum.py/datum.py/
projection.py/ellipsoid.py/
_cache.py                 ✅ 18 tests (test_value_objects.py)
```

**Total geodesy: 175 tests. Dos bugs reales de código encontrados y
corregidos** (UTM zona 61; mi propia fórmula de verificación EPSG 9606 con
signo de rotación invertido, detectada antes de convertirse en un falso
positivo permanente). **Un bug de arquitectura de tests** (colisión con el
módulo estándar `io`), documentado como regla general para el resto de
PR19.

**Estado final: 377/377 tests en todo el sandbox (dos corridas
consecutivas), `ruff`/`mypy` limpios.**

### Siguiente módulo pendiente de PR19

Con `geodesy` cerrado, el resto de la estructura 19.4-19.10 sigue sin
auditar: `io` (LAS/LAZ/PLY/E57/ASCII -- 56 archivos), `processing` (73
archivos, el módulo más grande), `terrain` (resto más allá de
`TIN.from_mesh()` -- DTM, contornos, interpoladores), `analysis`,
`features`, `dxf`/`gpkg` (export), `workflow`.

## HALLAZGO CRÍTICO: TD-001 nunca se persistió al repo real -- re-implementado

Al auditar `terrain` para PR19, se encontró que **ninguna de las 4 clases
de interpolación** (`NearestInterpolator`, `IDWInterpolator`,
`BarycentricInterpolator`, `LinearInterpolator`) heredaba de
`BaseInterpolator` ni implementaba `interpolate_many()` -- exactamente el
estado que el documento de contexto original describía como **ya resuelto**
por TD-001, antes de esta sesión.

**Verificado con un volcado fresco del repositorio real de Hernán**
(`repo-to-text_2026-08-14-05-18-58-UTC.txt`, subido específicamente para
esta verificación) que **TD-001 efectivamente falta en el repo real** --
no era un volcado desactualizado. Se auditaron también TD-002/003/004 en
ese mismo volcado fresco: **las tres SÍ están correctamente presentes**
(excepción única `TopoCoreError`, `Chunk.clone()`/`PointCloud.clone()`
reales, validación de atributos requeridos en `Chunk.__init__`). El
problema quedó aislado específicamente a TD-001 -- no es una pérdida
generalizada de todo el trabajo previo, sino de esta pieza puntual (muy
probablemente: el diff de esa pieza específica nunca se aplicó al repo
real desde el sandbox efímero de esa sesión anterior).

### TD-001 re-implementado en esta sesión

```
nearest.py       ✅ hereda BaseInterpolator + interpolate_many() vectorizado
                     vía TIN.vertex_array() y broadcasting numpy
idw.py            ✅ hereda BaseInterpolator + interpolate_many() vectorizado,
                     preserva el caso especial de coincidencia exacta con
                     vértice (antes: retorno directo; ahora: por fila)
barycentric.py     ✅ hereda BaseInterpolator + interpolate_many() como
                      bucle documentado (decisión consciente: find_triangle()
                      es O(triangle_count) sin índice espacial, diferido a
                      PR20 -- no hay atajo de vectorización posible aquí)
linear.py           ✅ hereda BaseInterpolator + interpolate_many() delega
                       directamente en BarycentricInterpolator
```

**`tests/terrain/test_interpolate_many.py` (20 tests)** -- verificación
principal: `interpolate_many()` contra `interpolate()` llamado punto por
punto (mismo criterio que el documento de contexto original describía para
TD-001, no un valor calculado a mano), más casos de empate en Nearest
(debe coincidir con el "primer ocurrente" de `min()`), coincidencia exacta
de vértice en IDW sobreviviendo la vectorización, y verificación de que
`LinearInterpolator` genuinamente delega (no reimplementa por separado).

### `terrain` -- progreso hasta ahora

```
tin.py (from_mesh, PR18B)         ✅
slope.py / aspect.py / _geometry.py  ✅ 24 tests, verificado con geometría
                                        analítica (4 direcciones cardinales,
                                        plano de 45°, invariancia de orden
                                        de vértices)
hillshade.py                       ✅ 15 tests, fórmula Lambertiana
                                        verificada (sol cenital=255 exacto,
                                        incidencia perpendicular=255 exacto,
                                        autosombra=0 exacto)
nearest.py/idw.py/barycentric.py/
linear.py (interpolate_many)        ✅ 20 tests -- TD-001 re-implementado
```

**Estado final: 436/436 tests en todo el sandbox (dos corridas
consecutivas), `ruff`/`mypy` limpios.**

### Implicación para el resto de PR19

Este hallazgo cambia el nivel de confianza en la lista "completado" del
documento de contexto original: **no asumir que ningún ítem de PR14-PR18T
está realmente en el repo real sin verificarlo primero**, incluso si el
documento lo describe como terminado. El patrón (sandbox efímero de una
sesión → nunca persistido al repo real) ya se confirmó dos veces en esta
sesión (tests completos, y ahora TD-001 específicamente).

## Bug real #2 en Terrain descubierto durante PR19: `Grid`/`Raster` bounds inconsistentes

**Bug real de Terrain descubierto en PR19, no una simple ampliación de
cobertura** (a diferencia de TD-001, este NO estaba documentado en ningún
lugar como conocido -- es un hallazgo genuinamente nuevo).

`Grid.columns`/`Grid.rows` usan `math.ceil(width/resolution) + 1` --
correcto y deliberado, garantiza cobertura completa del terreno cuando la
resolución no divide exactamente el ancho/alto. Pero el resto de la API
(`bounds`, `contains()`, `Raster.transform`) seguía describiendo la
extensión **nominal** (`max_x`/`max_y` pedidos por el usuario), no la
extensión **real** que la grilla efectivamente genera y llena con datos.

**Impacto confirmado**: con `resolution=3.0` sobre un ancho de `10.0`
(no múltiplo exacto), la última columna real de la grilla cae en
`x=12.0` -- 2 unidades más allá de `max_x=10.0` -- y `Grid.contains()`
rechazaba esa misma celda que `DTM.from_tin()` genuinamente había llenado
con una elevación interpolada real. `Raster.transform` (usado para
georreferenciar exportaciones GDAL/GeoTIFF) usaba el `max_y` nominal, lo
que habría producido rásteres exportados mal alineados en cualquier
software GIS cada vez que la resolución no dividiera exactamente el alto
solicitado -- un escenario común en topografía real (predios con
dimensiones arbitrarias, resoluciones redondas).

**Decisión de Hernán, congelada**: no reducir la grilla para forzarla
dentro de `max_x`/`max_y` (perdería cobertura real del terreno).
`columns`/`rows`/`coordinate()` quedan intactos. Se agregó
`Grid.actual_max_x`/`actual_max_y` (extensión real generada), y
`bounds`/`contains()`/`Raster.transform` ahora son coherentes con esa
extensión real en vez de la nominal.

**Regla que Hernán dejó para PR19**: *"La extensión espacial de un
Raster debe describir exactamente el dominio espacial de las celdas que
realmente contiene."*

**`tests/terrain/test_grid_bounds.py` (9 tests)** -- ambos casos que pidió
Hernán explícitamente: ancho divisible (`10/2`, extensión nominal =
extensión real, sin cambios) y ancho no divisible (`10/3`, extensión real
> nominal, `bounds`/`contains()`/`transform` deben coincidir), más el
caso específico de que la última celda generada por una grilla satisfaga
el `contains()` de esa misma grilla.

### `terrain` -- progreso actualizado

```
tin.py (from_mesh)                  ✅ PR18B
slope.py/aspect.py/_geometry.py       ✅ 24 tests
hillshade.py                           ✅ 15 tests
nearest/idw/barycentric/linear
  (interpolate_many)                    ✅ 20 tests -- TD-001 recuperado
cell.py                                  ✅ auditado, sin hallazgos
grid.py/raster.py                         ✅ 9 tests -- bug real #2 corregido
                                             (bounds/contains/transform)
dtm.py                                     ⏳ auditado parcialmente (usa
                                               Grid, ya cubierto indirectamente)
resto (contours, filters, sampling,
  breaklines, nodata, conversion,
  interpolation, models, validation,
  algorithms/delaunay,
  algorithms/constrained_delaunay)          ⏳ pendiente
```

**Estado final: 445/445 tests en todo el sandbox (dos corridas
consecutivas), `ruff`/`mypy` limpios.**

## Bug real #3 (CRÍTICO) en Terrain descubierto durante PR19: `ContourGenerator` pierde curvas en elevaciones exactas de vértices

**Clasificado CRÍTICO por Hernán** tras verificarse con un caso realista (no
solo sintético): `_triangle_segment()` descartaba una arista cuando
`(z0-level)*(z1-level) >= 0` -- correcto para el caso normal, pero también
descartaba el caso donde un extremo cae **exactamente** sobre el nivel
(`producto == 0`), tratándolo como "no cruza" en vez de un punto de paso
legítimo.

**Impacto confirmado con dos casos**:
- Sintético (pirámide de 5 vértices): nivel exactamente en la base (`z=0`,
  compartido por 4 vértices) → 0 contornos generados, aunque el perímetro
  completo de la base es una curva de nivel real.
- **Realista** (grilla 5×5, 25 vértices, Delaunay real vía
  `TIN.from_points()`): plataforma nivelada de 3×3 a `z=100.0` exacto
  rodeada de talud hacia `z=90.0` -- exactamente el escenario de una placa
  de cimentación o andén nivelado con gradería perimetral. Tanto la
  elevación exacta de la plataforma (`100.0`) como la del anillo
  perimetral (`90.0`) desaparecían por completo -- las dos líneas que un
  topógrafo más necesitaría ver delineadas en un plano de curvas de nivel.

**Corregido**: `_triangle_segment()` reescrito para clasificar cada vértice
como ARRIBA/ABAJO/SOBRE el nivel (`EPSILON` de tolerancia). Con exactamente
2 vértices SOBRE el nivel, esa arista **es** el segmento de contorno (sin
interpolación). Con exactamente 1 vértice SOBRE el nivel y los otros dos en
lados opuestos, el segmento es (vértice SOBRE, punto interpolado en la
arista opuesta). El caso genuinamente degenerado (1 vértice SOBRE, los
otros dos del MISMO lado -- la curva solo toca un punto, sin línea real) se
preserva sin cambios, verificado explícitamente con test.

**`tests/terrain/test_contours.py` (9 tests)** -- ambos casos de
reproducción (pirámide + plataforma realista), casos de control sin
regresión, el caso degenerado preservado, y `generate()` incluyendo niveles
que caen exactamente en elevaciones de vértice.

### `terrain` -- progreso actualizado

```
tin.py (from_mesh)                  ✅ PR18B
slope.py/aspect.py/_geometry.py       ✅ 24 tests
hillshade.py                           ✅ 15 tests
nearest/idw/barycentric/linear
  (interpolate_many)                    ✅ 20 tests -- TD-001 recuperado
cell.py                                  ✅ auditado, sin hallazgos
grid.py/raster.py                         ✅ 9 tests -- bug real #2 corregido
contours.py                                ✅ 9 tests -- bug real #3 CRÍTICO corregido
dtm.py                                      ⏳ cubierto indirectamente
resto (filters, sampling, breaklines,
  nodata, conversion, interpolation,
  models, validation,
  algorithms/delaunay,
  algorithms/constrained_delaunay)            ⏳ pendiente
```

**Estado final: 454/454 tests en todo el sandbox (dos corridas
consecutivas), `ruff`/`mypy` limpios.**

**Tres bugs reales encontrados y corregidos en `terrain` durante esta
sesión de PR19** (TD-001 recuperado, `Grid`/`Raster` bounds, y ahora
`ContourGenerator` con elevaciones exactas -- este último clasificado
CRÍTICO).

## `terrain/algorithms` -- Delaunay y Delaunay restringido, sin bugs reales

Auditados `delaunay.py` (wrapper sobre `scipy.spatial.Delaunay`) y
`constrained_delaunay.py` (647 líneas, algoritmo de Sloan 1993 con
inserción de restricciones vía *edge flipping*).

**`delaunay.py` (16 tests)**: validación de duplicados/colinealidad,
`compute_bbox`, `validate_result`, delegación a `triangle_vertices`/
`neighbor_indices`. Un ajuste de test propio (no del código): `scipy`
no garantiza el orden exacto de vértices dentro de un símplex, solo el
conjunto -- corregido el test para comparar como conjunto, no tupla.

**`constrained_delaunay.py` (21 tests)**: predicados geométricos
(`_orient`, `_in_circle`, `_segments_cross`) verificados contra
geometría analítica conocida (círculo unitario para `_in_circle`,
incluyendo robustez ante orden de bobinado CW/CCW). Test de integración
forzando una diagonal que la triangulación Delaunay simple NO elegía,
confirmando que la maquinaria de recuperación por *flips* funciona de
extremo a extremo. Los 3 casos de error documentados (auto-referencia,
índice fuera de rango, restricciones que se cruzan) verificados.

**Hallazgo investigado y descartado como no alcanzable**: `_segments_cross`
da `True` para dos segmentos que solo se tocan en un extremo compartido
(no un cruce propiamente dicho) -- causado por que el chequeo
`(d>0)!=(d2>0)` no distingue "negativo" de "exactamente cero". Se
rastrearon los 3 únicos puntos de llamada del módulo: los 4 índices que
reciben siempre vienen de una arista compartida `(a,b)` más los ápices
`(c,d)` de sus dos triángulos adyacentes vía `_apex()`, que por
construcción nunca repite vértices -- los 4 puntos son siempre
geométricamente distintos en la práctica. **No se modificó el
algoritmo** (el caso ambiguo no es alcanzable desde ningún punto de
llamada real); el test documenta el comportamiento real en vez de
asumir la semántica ideal del docstring.

### `terrain` -- progreso actualizado

```
tin.py                              ✅ PR18B
slope/aspect/_geometry                ✅ 24 tests
hillshade                              ✅ 15 tests
interpolate_many (TD-001)                ✅ 20 tests -- recuperado
cell.py                                   ✅ sin hallazgos
grid.py/raster.py                          ✅ 9 tests -- bug real #2
contours.py                                 ✅ 9 tests -- bug real #3 CRÍTICO
algorithms/delaunay.py                       ✅ 16 tests
algorithms/constrained_delaunay.py            ✅ 21 tests
dtm.py                                         ⏳ cubierto indirectamente
resto (filters, sampling, breaklines,
  nodata, conversion, interpolation,
  models, validation, enums, types)              ⏳ pendiente
```

**Estado final: 491/491 tests en todo el sandbox (dos corridas
consecutivas), `ruff`/`mypy` limpios.**

## `terrain` -- MÓDULO COMPLETO (29/29 archivos con cobertura real)

Cierre del módulo `terrain` para PR19. Cobertura final: 30 archivos fuente
(29 con lógica propia, más `types.py` de solo alias sin comportamiento en
tiempo de ejecución, sin test dedicado por no tener nada que verificar) y
18 archivos de test.

### Cuarto bug real encontrado y corregido: `DTM.from_tin()` truena fuera del casco convexo

El array de valores se pre-llena con `NaN` (`np.full(grid.shape, np.nan,
...)`) -- la intención de diseño es clara: celdas que no se puedan calcular
quedan como NoData. Pero el bucle nunca capturaba `InterpolationError`, así
que **una sola celda de la grilla fuera del casco convexo del TIN hacía
que todo `DTM.from_tin()` truene**, en vez de dejar esa celda como `NaN`.
Escenario común en la práctica: una grilla rectangular de DTM sobre un
levantamiento real casi nunca coincide exactamente con el casco convexo
irregular de los puntos.

**Corregido**: `try/except InterpolationError: continue` alrededor de la
llamada a `interpolator.interpolate()`, coherente con el pre-llenado del
array. Clasificado como bug real por Hernán.

### Quinto bug real: `InterpolationMethod` duplicado con comparación por identidad

`topocore.terrain.enums` y `topocore.terrain.interpolation` declaraban
**cada uno su propia clase `InterpolationMethod`** -- mismo nombre,
miembros parcialmente distintos (`NATURAL_NEIGHBOR` en `enums.py`, nunca
implementado en ningún lugar del repo, vs. `NEAREST` en `interpolation.py`,
sí implementado). Como `TerrainInterpolator.interpolate()` comparaba con
`is` (identidad de clase) en vez de `==` (valor), pasar
`enums.InterpolationMethod.LINEAR` -- el camino de importación "obvio",
ya que ahí vive todo lo demás del módulo como `BreaklineType` -- hacía que
**todas las comparaciones fallaran en silencio**, cayendo al último
`else` (`NEAREST`) sin ningún error. Confirmado con un TIN real:
`7.5` (linear correcto) vs. `10.0` (nearest silencioso) para el mismo
punto de consulta.

**Verificado que `enums.InterpolationMethod` no se usaba en ningún otro
lugar del repo** antes de decidir el enfoque de consolidación.
**Corregido**: `enums.py` es ahora la única fuente de verdad (con
`NEAREST`, sin `NATURAL_NEIGHBOR` muerto); `interpolation.py` reimporta
desde ahí. Las dos clases son ahora literalmente la misma
(`EnumsMethod is InterpMethod == True`), haciendo el bug estructuralmente
imposible de repetir.

### Resumen de los 5 bugs reales de `terrain` encontrados en esta sesión de PR19

1. **TD-001** -- `interpolate_many()` nunca se persistió al repo real (recuperado).
2. **`Grid`/`Raster` bounds** -- inconsistentes con la extensión real cuando la resolución no divide exactamente el ancho/alto (afecta export GDAL).
3. **`ContourGenerator`** (CRÍTICO) -- curvas de nivel desaparecían en elevaciones exactas de vértices (plataformas/andenes nivelados).
4. **`DTM.from_tin()`** -- truena en vez de dejar `NaN` fuera del casco convexo.
5. **`InterpolationMethod` duplicado** -- comparación por identidad entre dos enums del mismo nombre, silenciosamente usa el método equivocado.

Ninguno de estos fue encontrado por "que no truene" -- todos requirieron
verificación matemática/geométrica activa contra geometría analítica
conocida o casos reales, exactamente la disciplina que motivó la
estructura de PR19 desde el principio.

### Cobertura final del módulo

```
tin.py                              ✅ PR18B
slope/aspect/_geometry                ✅ 24 tests
hillshade                              ✅ 15 tests
interpolate_many (TD-001)                ✅ 20 tests
cell.py                                   ✅
grid.py/raster.py                          ✅ 9 tests -- bug #2
contours.py                                 ✅ 9 tests -- bug #3 CRÍTICO
algorithms/delaunay.py                       ✅ 16 tests
algorithms/constrained_delaunay.py            ✅ 21 tests
filters.py                                     ✅ 21 tests
sampling.py                                     ✅ 11 tests
breaklines.py                                    ✅ 10 tests
interpolation.py + enums.py                       ✅ 11 tests -- bug #5
dtm.py                                             ✅ 4 tests -- bug #4
weights.py                                          ✅ 16 tests
models.py                                            ✅ 14 tests
validation.py                                         ✅ 19 tests
nodata.py                                              ✅ 9 tests
conversion.py                                           ✅ 3 tests
types.py                                                 -- solo alias, sin test
```

**Estado final: 605/605 tests en todo el sandbox (dos corridas
consecutivas), `ruff`/`mypy` limpios.**

**MÓDULO `terrain` CERRADO para PR19** -- pendiente únicamente de la
integración en el repo real de Hernán (aplicar código, suite completa,
`ruff`/`mypy`, segunda corrida) antes de poder declararlo `CONGELADO`.

## PR19 -- `processing` arranca: bug GRAVE en `FeatureManager` corregido primero (paso independiente)

Hernán definió el orden para `processing` (el módulo más grande y de mayor
riesgo): `normals` → `features` → `sampling` → `filters` → `segmentation`
→ `classification` → `ground` → `registration` (esta última al final por
su complejidad). Auditando `processing.normals` se encontró que su caché
(`NormalManager._cache`) nunca se lee ni se escribe -- solo se limpia --
haciéndolo completamente inerte (desperdicia cómputo, pero nunca da
resultados incorrectos). Antes de decidir cómo arreglarlo, se revisaron
los demás `manager.py` de `processing` para encontrar el patrón correcto
a replicar, y **eso reveló un bug mucho más grave, de rebote**:

### Bug GRAVE encontrado: `FeatureManager` devolvía resultados de una nube anterior, en silencio

`features/manager.py` congelaba `id(cloud)` **una sola vez, en el
constructor** (o `0` si no se pasaba nube), y reutilizaba ese valor fijo
como parte de la clave de caché en **todas** las llamadas posteriores a
`compute()`/`compute_all()` -- sin importar qué nube se pasara realmente
en cada llamada. Confirmado con datos reales: una nube con
`z=[10,20,30]` y otra con `z=[100,200,300]`, ambas usando el mismo
`FeatureManager()` (sin nube en el constructor) -- la segunda llamada
devolvía `[10,20,30]` (el resultado de la *primera* nube) en vez de
`[100,200,300]`. **Sin ningún error, sin ninguna advertencia.**

Reusar un `FeatureManager` entre varias nubes (patrón natural: procesar
un lote de tiles con un manager compartido preconfigurado) producía
características completamente equivocadas para cada nube después de la
primera.

**Corregido**: `id(cloud)` se calcula ahora en cada llamada (nunca se
almacena), replicando el patrón que ya funcionaba correctamente en
`topocore.processing.filters.manager.FilterManager` (que calcula
`id(current)` dentro de su propio bucle, no en el constructor). Se quitó
el campo `_cloud_id` de `__slots__` por completo -- ya no hace falta
almacenar nada.

**Limitación conocida y heredada, documentada, no corregida aquí**: `id()`
es una dirección de memoria, reutilizable tras recolección de basura -- un
manager de vida muy larga podría teóricamente ver una colisión coincidente
entre una nube ya liberada y una nueva asignada en la misma dirección. No
se corrige en este paso (sería un rediseño de la estrategia de caché --
p.ej. hash de contenido -- no una corrección de bug); el tamaño acotado de
`LRUCache` limita cuánto puede persistir una entrada obsoleta. Mismo
riesgo, no introducido aquí, ya presente en `filters/manager.py`.

**`tests/processing/features/test_manager.py` (12 tests)**: el caso exacto
de reproducción (dos nubes, resultados distintos), alternancia repetida
entre nubes, 5+ nubes consecutivas, confirmación de que el caché sigue
funcionando genuinamente (mismo objeto en cache hit, `clear_cache()`
fuerza recómputo con contador de llamadas), y el camino de construcción
con nube inicial (`FeatureManager(cloud=...)`) también pasaba por el
mismo `_cloud_id` roto.

**Estado: 617/617 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Panorama de caché en los managers de `processing` (auditado, no todos corregidos todavía)

```
filters/manager.py       ✅ correcto -- id(cloud) calculado por llamada
features/manager.py      ✅ corregido en este paso (bug grave)
normals/manager.py       ⚠️ caché muerto -- pendiente (paso siguiente)
ground/manager.py        ⚠️ mismo patrón muerto que normals -- pendiente
neighbors/manager.py     ✅ correcto -- diseño distinto (una instancia por
                             nube), clave por índice de punto es segura
classification/registration/
  sampling/segmentation    -- sin intento de caché, nada que revisar
```

### Siguiente paso (independiente, según decisión de Hernán)

1. `normals/manager.py` -- decidir/implementar caché real siguiendo el
   patrón correcto de `filters/manager.py`.
2. `ground/manager.py` -- mismo análisis.
3. Tests específicos de cambio de nube para impedir regresiones (mismo
   patrón que `test_manager.py` de `features`).
4. Suite completa + `ruff` + `mypy`.

## PR19 -- caché real implementado en `normals`/`ground` (paso independiente, completado)

Siguiendo la decisión de Hernán de separar esto del arreglo de
`FeatureManager`: implementado caché real en `NormalManager` y
`GroundManager`, siguiendo el patrón correcto ya usado en
`filters/manager.py` (`id(cloud)` calculado en cada llamada, nunca
almacenado).

### `NormalManager`

`_cache_key()` ya existía en el código (declarado, nunca conectado) --
al diseñar la conexión se encontró que **su propio diseño tenía un fallo
latente todavía sin estrenar**: el `viewpoint` se guardaba como
`viewpoint is not None` (booleano), no como el valor real -- dos
viewpoints distintos habrían colisionado en la misma entrada de caché en
cuanto se conectara. Corregido antes de conectar: la clave ahora incluye
las coordenadas reales del viewpoint como tupla, y también `sigma`
(relevante solo para `weighted_pca`, tampoco estaba en la clave
original). `estimate()`/`estimate_at()`/`estimate_curvature()`/
`estimate_both()` comparten un único `_estimate_both_cached()` interno,
así que llamar más de uno con la misma nube/parámetros calcula el PCA
una sola vez.

### `GroundManager`

Mismo patrón muerto que `NormalManager` (`_cloud_id` fijo en `0`, nunca
`.get()`/`.set()`). Más complejo por tener 20+ parámetros específicos por
método y aceptar overrides por llamada vía `**kwargs` (que nunca
persisten a `self`). **Alcance acotado deliberadamente**: caché real
implementado solo para `classify()` (el único método para el que el tipo
de caché original -- `BoolArray1D` -- estaba diseñado). `extract()`/
`estimate_elevation()` con extractor/estimador dedicado quedan sin
cachear (documentado explícitamente en el código como decisión de
alcance, no como omisión) -- devuelven tipos distintos (`PointCloud`,
`FloatArray1D`) que el diseño original nunca contempló, y extenderlo
sería un rediseño mayor, no la corrección de este hallazgo puntual.

**Hallazgo adicional, no perseguido en este paso**: al verificar
`GroundManager.classify()` con un caso de bloque elevado (simulando un
edificio), el método `grid` clasificó el 100% de los puntos como suelo,
sin separar el bloque -- posible problema del algoritmo `classify()` en
sí, no del caché. Le toca su turno cuando `ground` reciba su auditoría
completa más adelante, según tu propio orden (`normals → features →
sampling → filters → segmentation → classification → ground →
registration`).

**Tests**: `tests/processing/normals/test_manager.py` (10 tests) y
`tests/processing/ground/test_manager.py` (8 tests) -- cambio de nube sin
colisión, alternancia repetida, viewpoints/sigma/parámetros distintos sin
colisión, caché genuino (mismo objeto en cache hit), invalidación correcta
vía setters y `clear_cache()`.

**Arquitectura de tests**: se agregaron `__init__.py` a
`tests/processing/`, `tests/processing/normals/`, `tests/processing/ground/`,
`tests/processing/features/` -- mismo patrón ya establecido para evitar
colisión de nombres de módulo entre carpetas (`test_manager.py` existe en
3 lugares distintos).

**Estado: 635/635 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Panorama final de caché en managers de `processing`

```
filters/manager.py       ✅ correcto (ya lo estaba)
features/manager.py      ✅ corregido (bug grave: resultados obsoletos entre nubes)
normals/manager.py       ✅ caché real implementado (estaba muerto)
ground/manager.py        ✅ caché real implementado para classify() (estaba muerto)
neighbors/manager.py     ✅ correcto (diseño distinto, no necesitaba cambios)
classification/registration/
  sampling/segmentation    -- sin caché, nada que revisar
```

## Corrección de un hallazgo anterior: `GroundManager.classify()` SÍ funciona correctamente

El hallazgo anotado en el paso de caché de `normals`/`ground`
("`GridGroundClassifier` no separó un bloque elevado del suelo") era un
**falso positivo del test, no un bug del código**. Investigado y
descartado con evidencia:

`GridGroundClassifier` clasifica **por celda**: compara cada punto contra
el mínimo de *su propia celda de la grilla*. La prueba original usaba
exactamente un punto por celda (grilla 1×1 con puntos espaciados 1.0) --
en ese caso degenerado, cada punto (incluidos los del "edificio") es
trivialmente el mínimo de su propia celda vacía, sin nada real con qué
compararse, así que **todo** se clasifica como suelo por construcción --
comportamiento matemáticamente correcto para ese input, no un bug.

**Reproducción correcta** (densidad realista, varios puntos por celda,
terreno y techo compartiendo las mismas celdas -- el escenario real de
LiDAR): 500 puntos de terreno (z~0) y 80 puntos de techo elevado (z~3.0,
bloque de 4×4 celdas), `cell_size=1.0`, `height_threshold=0.2`.
Resultado: **500/500 puntos de terreno correctamente clasificados como
suelo, 0/80 puntos de techo correctamente excluidos**. El algoritmo
funciona exactamente como documenta su docstring.

**No se requiere ninguna corrección.** Este hallazgo queda cerrado.

## `processing.features` -- MÓDULO COMPLETO. Segundo bug grave encontrado

### Bug GRAVE: `RelativeHeightFeatureComputer` nunca leía X/Y, devolvía la altura absoluta sin cambios

Confirmado con reproducción directa: para una nube con 3 puntos de suelo
(`z=0,1,2`) y 2 puntos no-suelo (`z=5,10`), el resultado era
`[0, 1, 2, 5, 10]` -- **exactamente los valores de Z sin modificar**, no
"altura relativa al suelo".

Causa: la función nunca extraía X/Y de los puntos -- solo Z y
clasificación. Construía un array `ground_points` metiendo los valores de
Z de los puntos de suelo en la posición X (con Y/Z en cero), y hacía **una
sola consulta global** para toda la nube usando `(0, 0, z[0])` en vez de
una consulta por punto usando sus propias coordenadas reales. El resultado
terminaba siendo, por construcción accidental de la búsqueda, el mismo
`ground_z` restado de todos los puntos (coincidiendo con 0 en el caso de
reproducción, dejando la altura absoluta intacta).

**Corregido** replicando el patrón ya correcto y probado de
`GroundManager._nearest_ground_elevation()`: coordenadas `(x,y,z)` reales
de los puntos de suelo, una consulta por punto usando sus propias
coordenadas. Verificado con un caso de dos clusters de suelo en ubicaciones
y elevaciones distintas -- cada punto no-suelo ahora usa correctamente la
elevación del cluster geométricamente más cercano (`3.0 - 0.0 = 3.0` para
uno, `3.0 - 5.0 = -2.0` para el otro), no un valor global equivocado.

### Resto del módulo: `PCAFeatures` verificado sin bugs

`planarity`/`linearity`/`sphericity`/`verticality`/`anisotropy` verificados
contra tres configuraciones geométricas puras y deterministas (plano
horizontal, pared vertical, línea recta) -- **coincidencia exacta** en
todos los casos (`1.0000`/`0.0000`, no solo "aproximado"). `base.py` es
solo interfaces abstractas sin lógica propia.

**Tests**: `tests/processing/features/test_geometric.py` (11 tests) y
`test_pca.py` (9 tests).

**Estado: 655/655 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Resumen de bugs graves de caché/manager encontrados en `processing` hasta ahora

1. `FeatureManager` -- resultados obsoletos entre nubes (corregido)
2. `RelativeHeightFeatureComputer` -- nunca leía X/Y, altura relativa rota (corregido)

**Módulo `processing.features` cerrado.** Siguiente en el orden de
Hernán: `processing.sampling`.

## `processing.sampling` -- MÓDULO COMPLETO. Dos bugs reales corregidos

Auditados los 8 archivos del módulo con geometría analítica conocida
(voxel: indexación exacta en bordes negativos, centroide exacto,
`closest` verificado; density: clusters denso/disperso con densidad real
medida antes de calibrar el `target_density` de la prueba). `voxel.py`,
`random.py`, `uniform.py`, `density.py`, `base.py` -- **sin bugs
encontrados**.

### Bug real: `StratifiedSampler` con semilla fija embebida en el código

El método `"random"` usaba `np.random.default_rng(42)` -- **sin ningún
parámetro `seed` en el constructor**, a diferencia de `RandomSampler`/
`VoxelSampler`/`DensitySampler`, que sí lo exponen. Confirmado: dos
llamadas (misma instancia, o instancias distintas) daban **siempre
exactamente el mismo resultado**, sin ninguna forma de cambiarlo.
Verificado que ningún otro lugar del repo dependía del valor `42` antes
de tocarlo. **Corregido**: parámetro `seed: int | None = None` agregado,
consistente con el resto del módulo.

### Bug real (preexistente, no introducido en esta sesión): `SamplingManager` no pasaba `seed`

Al agregar `seed` a `StratifiedSampler`, se encontró que
`SamplingManager._create_sampler()` **nunca pasaba `seed`** para los
métodos `"voxel"` ni `"stratified"` -- aunque ambos samplers subyacentes
sí lo aceptan. Confirmado con evidencia real: `manager.sample(cloud,
method="random", seed=7)` para `voxel` daba resultados distintos en
llamadas repetidas (el seed se ignoraba en silencio). El caso de `voxel`
es preexistente (no relacionado a mi cambio de esta sesión); Hernán
confirmó corregirlo también. **Ambos corregidos**: `seed=params.get("seed")`
agregado a las ramas `"voxel"` y `"stratified"` de `_create_sampler()`.

**Tests**: 45 tests nuevos en `tests/processing/sampling/` (`test_voxel.py`,
`test_random_uniform.py`, `test_stratified.py`, `test_density.py`,
`test_manager.py`).

**Estado: 700/700 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Panorama de `processing` hasta ahora

```
normals/     ✅ completo
features/    ✅ completo -- 2 bugs graves corregidos
ground/      ✅ caché corregido, classify() confirmado correcto
sampling/    ✅ completo -- 2 bugs de seed corregidos
filters, segmentation, classification  ⏳ pendiente
registration  ⏳ al final (criterio de Hernán)
```

**Módulo `processing.sampling` cerrado.** Siguiente en el orden de
Hernán: `processing.filters`.

## `processing.filters` -- MÓDULO COMPLETO. Sin bugs encontrados

Auditados los 8 archivos del módulo con casos analíticos claros:

- `statistical.py` (SOR) y `radius.py` (ROR): verificados con un cluster
  denso de 100 puntos + 2 outliers aislados lejanos -- **100% del cluster
  conservado, 100% de los outliers removidos**, por ambos filtros de
  forma independiente.
- `clip_polygon.py`: algoritmo de ray-casting (PNPOLY) verificado con
  cuadrado simple y **polígono cóncavo en forma de L** -- el "muesco"
  se detecta correctamente como exterior, confirmando que funciona para
  formas no convexas, no solo el caso trivial.
- `crop_box.py`, `pass_through.py`: verificados con límites inclusivos y
  casos de rechazo (NaN, rangos inválidos).
- `manager.py`: ya confirma el patrón correcto de caché (`id(current)`
  calculado dentro del bucle, nunca almacenado) -- el mismo patrón usado
  como referencia para corregir `features`/`normals` antes en esta
  sesión.

**Ningún bug encontrado en todo el módulo.**

**Tests**: 32 tests nuevos en `tests/processing/filters/`
(`test_statistical_radius.py`, `test_geometric_filters.py`,
`test_manager.py`).

**Estado: 732/732 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Panorama de `processing` hasta ahora

```
normals/     ✅ completo
features/    ✅ completo -- 2 bugs graves corregidos
ground/      ✅ caché corregido, classify() confirmado correcto
sampling/    ✅ completo -- 2 bugs de seed corregidos
filters/     ✅ completo -- sin bugs
segmentation, classification  ⏳ pendiente
registration  ⏳ al final (criterio de Hernán)
```

**Módulo `processing.filters` cerrado.** Siguiente en el orden de
Hernán: `processing.segmentation`.

## `processing.segmentation` -- MÓDULO COMPLETO. Tres bugs reales corregidos

Auditados los 7 archivos del módulo. `dbscan.py` y `connected_components.py`
verificados sin bugs (dos clusters separados con conteos exactos, punto
borde clásico de DBSCAN correctamente unido al cluster, descarte de
componentes pequeños sin dejar huecos en los IDs).

### Bug real #1 (CRÍTICO): `RegionGrowingSegmenter` con radio de crecimiento derivado de un conteo, no de una distancia

`radius=self._k * 0.1` -- multiplicaba `k` (un **conteo** de vecinos,
pensado para estimación de normales) por una constante arbitraria, tratado
como si fuera una **distancia** espacial. Confirmado: un plano
perfectamente plano con espaciado real de `5.0` unidades (que debería
formar trivialmente 1 región, curvatura=0 en todas partes) daba
`num_segments=0`, **todos los puntos marcados como ruido** -- el radio
resultante (`10*0.1=1.0`) nunca encontraba ningún vecino a `5.0` de
distancia. **Corregido** derivando el radio de la densidad real medida de
la nube (distancia media a los k vecinos más cercanos), replicando
exactamente la técnica ya usada en `DBSCANSegmenter._compute_eps_values()`
y `ConnectedComponentsSegmenter._compute_threshold_values()`.

### Bug real #2: inconsistencia de signo de normales en superficies verticales

Encontrado al verificar el fix anterior con una esquina piso+pared: la
pared sola se dividía en 2 regiones. Causa: `orient_upward=True` voltea la
normal si `z<0`, pero en una superficie perfectamente vertical `z=0`
exactamente -- el volteo nunca se activa, y el signo de cada PCA local
queda arbitrario e independiente entre puntos (confirmado: 30/26 puntos
con normales `(0,-1,0)` vs `(0,1,0)`). **Corregido en
`RegionGrowingSegmenter._normals_consistent()`** (no en el módulo
compartido `normals`, que otros consumidores podrían necesitar con
semántica de dirección absoluta) usando `abs(dot(...))` en vez de
`dot(...)` directo -- normales antiparalelas cuentan como consistentes,
mientras que superficies genuinamente perpendiculares (esquina piso/pared
real) se siguen separando correctamente.

### Bug real #3 (CRÍTICO): `TreeSegmenter`/`BuildingSegmenter` filtran por Z absoluto, no por altura sobre el suelo

Ambos docstrings dicen *"height above ground"*, y `TreeSegmenter` incluso
documenta explícitamente **"1. Ground classification to separate trees
from ground"** como paso 1 de su algoritmo -- paso que nunca existió en el
código real. Confirmado con datos realistas (elevación absoluta ~1500m,
un árbol genuino de 1-10m de altura): `SegmentationError: No points found
above minimum height` con los parámetros por defecto documentados de la
propia clase -- **inutilizable con cualquier dato topográfico real no
normalizado a cero**. **Corregido** clasificando el suelo primero
(`GroundManager`, ya auditado y con caché corregido en esta sesión) y
filtrando por la altura de cada punto sobre su vecino de suelo
geométricamente más cercano (mismo patrón ya corregido para
`RelativeHeightFeatureComputer`). Se agregó el parámetro `ground_method`
a ambas clases, y se corrigió también `SegmentationManager` (mismo patrón
de parámetro no propagado que ya encontramos en `sampling`).

**Tests**: 32 tests nuevos en `tests/processing/segmentation/`
(`test_dbscan_connected.py`, `test_region_growing.py`, `test_specific.py`,
`test_manager.py`).

**Estado: 764/764 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Panorama de `processing` hasta ahora

```
normals/       ✅ completo
features/      ✅ completo -- 2 bugs graves
ground/        ✅ caché corregido, classify() confirmado correcto
sampling/      ✅ completo -- 2 bugs de seed
filters/       ✅ completo -- sin bugs
segmentation/  ✅ completo -- 3 bugs reales (2 CRÍTICOS)
classification  ⏳ pendiente
registration     ⏳ al final (criterio de Hernán)
```

**Módulo `processing.segmentation` cerrado.** Siguiente en el orden de
Hernán: `processing.classification`.

## Corrección post-entrega: `mypy` en el repo real detecta un error que el sandbox no atrapaba

Hernán corrió `uv run mypy src` en su repo real (376 archivos) y obtuvo:

```
src\topocore\processing\normals\manager.py:371: error: Returning Any from
function declared to return "tuple[ndarray[...], ndarray[...]]" [no-any-return]
```

**Causa**: `self._cache: LRUCache[CacheKey, Any]` -- el valor del caché
estaba tipado como `Any` en vez de `tuple[FloatArray2D, FloatArray1D]`. Al
recuperar (`self._cache.get(cache_key)`), mypy no podía confirmar que el
resultado coincidiera con el tipo de retorno declarado de
`_estimate_both_cached()`.

**Por qué el sandbox no lo atrapó**: la verificación de esta sesión usa
`mypy --follow-imports=silent` acotado al archivo tocado, para no
arrastrar advertencias preexistentes de módulos no relacionados en un
repo de 376 archivos. Este tipo de error de inferencia de retorno
(`no-any-return`) puede depender del contexto completo de tipos del
repo -- confirmado que con `MYPYPATH` apuntando al sandbox completo,
tanto con como sin `--follow-imports=silent`, el error no se reproducía
ahí, pero sí en el repo real de Hernán con los 376 archivos.

**Corregido**: `self._cache: LRUCache[CacheKey, tuple[FloatArray2D, FloatArray1D]]`
-- tipo de valor preciso en vez de `Any`. Verificado que ambas ramas de
`_estimate_both_cached()` ya devolvían exactamente ese tipo; el cambio es
puramente de anotación, sin impacto en tiempo de ejecución. 764/764 tests
del sandbox intactos tras el cambio.

**Nota para Hernán**: esto confirma que vale la pena, durante tu Fase 3
(`mypy .` sobre el repo completo), estar atento a hallazgos de inferencia
de tipo que el sandbox -- por diseño, acotado por archivo -- no puede
reproducir. Ya van dos casos así en esta sesión (este, y el de
`np.floating` vs `float` en `_clothoid_local_uv` de PR18C).

## `processing.classification` -- bug CRÍTICO y sistémico corregido (subsistema ML completo)

### Todo el subsistema de clasificadores ML estaba completamente roto

`MachineLearningClassifier.__init__()` creaba un `FeatureManager()` **sin
registrar ningún computador de features**. Confirmado con código real:
`RandomForestClassifier().fit(cloud, labels)` -- y por extensión las 4
subclases concretas (`GradientBoostClassifier`, `XGBoostClassifier`,
`LightGBMClassifier`, que comparten la misma base) -- truena de inmediato
con `ProcessingError: Feature 'height_above_ground' was not computed` en
la primera llamada. **Ningún clasificador ML de todo el proyecto era
utilizable.**

**Corregido en dos capas:**

1. Se agregó el registro real de computadores en
   `_register_feature_computers()`: `HeightFeatureComputer`,
   `PCAFeatureComputer` (ya existente, reutilizado para curvature →
   `surface_variation`, planarity, linearity, sphericity, verticality --
   mismo mapeo que ya usa `RuleBasedClassifier`), `DensityFeatureComputer`.

2. Al verificar el fix, se encontró **una segunda capa del mismo problema
   de fondo ya resuelto antes en esta sesión** para
   `TreeSegmenter`/`BuildingSegmenter`: `RelativeHeightFeatureComputer`
   exige que la nube **ya tenga clasificación previa** -- pero un
   clasificador ML se entrena sobre nubes **sin clasificar** (predecirla
   es el objetivo). Se creó `_GroundRelativeHeightFeatureComputer`
   (hereda de `ScalarFeatureComputer`), que clasifica el suelo
   geométricamente vía `GroundManager` sin depender de clasificación
   previa -- replicando el patrón ya probado en `segmentation/specific.py`.

Esto dejó el parámetro `ground_class` huérfano (ninguna función real) en
las 5 clases (`ml.py` + 4 subclases concretas). **Reemplazado por
`ground_method: str = "grid"`** en las 5, consistente con el mismo
parámetro ya usado en `TreeSegmenter`/`BuildingSegmenter`.

**Verificado con las 4 implementaciones concretas** (no solo
RandomForest): `fit()` + `classify()` funcionan correctamente en
`RandomForestClassifier`, `GradientBoostClassifier`, `XGBoostClassifier`,
`LightGBMClassifier`. Se instalaron `xgboost` y `lightgbm` en el sandbox
(no estaban presentes) para poder verificar con evidencia real, no solo
lectura de código.

**`rules.py`** (`RuleBasedClassifier`) verificado end-to-end con escena
sintética realista (suelo denso + techo elevado planar + cluster
irregular tipo árbol, elevación no basada en cero): suelo 100% correcto,
techo mayoritariamente `BUILDING` (~79%), vegetación mayoritariamente
correcta (~53%, limitado por la forma cruda del cluster sintético, no por
el código). Sin bug encontrado.

**Tests**: `tests/processing/classification/test_ml.py` (15 tests) --
las 4 implementaciones concretas parametrizadas contra el caso exacto de
reproducción, features personalizadas, todas las features derivadas de
PCA juntas, `ground_method` funcionando, y confirmación de que
`ground_class` fue genuinamente eliminado (no solo ignorado).

**Estado: 779/779 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios** (un `BLE001` y un `import-untyped` de `sklearn`
preexistentes, no introducidos en esta sesión, quedan intactos).

### Pendiente dentro de `processing.classification`

- `manager.py` (`ClassificationManager`) -- todavía no auditado a fondo.
- Cobertura de test formal para `rules.py` (verificado funcionalmente,
  falta escribir el test suite permanente).

### Panorama de `processing`

```
normals/, features/, ground/, sampling/, filters/, segmentation/  ✅ completos
classification/  ⚠️ bug crítico corregido (ml.py + 4 subclases);
                    rules.py verificado sin bugs; manager.py pendiente
registration/     ⏳ al final (criterio de Hernán)
```

## BUG CRÍTICO, NO DETERMINISTA: `compute_pca()` devolvía resultados de la nube equivocada por reutilización de `id()`

**El hallazgo más severo de toda la sesión de PR19**, encontrado por insistencia
explícita de Hernán de no dejarlo como "limitación documentada" cuando el
patrón inicial (test intermitente en `normals/manager.py`) apuntaba a algo
más profundo que una simple degeneración numérica.

### Investigación (siguiendo el árbol de diagnóstico exacto que pidió Hernán)

Se descartaron, en orden, con evidencia directa cada vez:
1. Caché de `NormalManager` -- el bug se reproducía con estimadores
   completamente frescos, sin manager, sin caché de por medio.
2. Colisión de `id(cloud)` -- confirmado que los `id()` de las nubes eran
   genuinamente distintos.
3. Desempate no determinista de vecinos (k-NN) -- confirmado que el
   conjunto exacto de 9 vecinos era idéntico entre la llamada mala y la
   buena.
4. Degeneración de autovalores -- verificado a mano que los autovalores
   (`0, 1.06, 1.33`) estaban bien separados, sin ambigüedad matemática.

### La causa real

`topocore.processing._shared.compute_pca()` tenía **su propio caché interno**
(`_PCA_CACHE`, a nivel de módulo) con `cache_key = (id(manager), k)`, donde
`manager` es un `NeighborhoodManager` **efímero**, construido nuevo en cada
llamada a `PCANormalEstimator.estimate()` y recolectado casi de inmediato.
Como Python reutiliza direcciones de memoria liberadas, el `id()` del
manager de una nube se reutilizaba con frecuencia para el manager de la
**siguiente nube completamente distinta**, causando que `compute_pca()`
devolviera, en silencio, el objeto `PCAComputation` **completo y obsoleto**
de la nube anterior (vecinos, covarianza, autovalores, autovectores -- todo,
no solo una aproximación numérica).

**Confirmado con traza definitiva**:
```
compute_pca llamado: manager_id=139903281751936  cache_hit=False   ← flat_cloud
compute_pca llamado: manager_id=139903281751936  cache_hit=True    ← MISMO id, tilted_cloud
compute_pca llamado: manager_id=139903281752000  cache_hit=False   ← tilted_cloud, ahora sí fresco
```

Reproducible en un bucle ajustado con ~1 de cada 10-30 intentos.

### Corrección

**Se eliminó `_PCA_CACHE` por completo.** Este caché nunca podía acertar
legítimamente en el uso normal (cada llamada crea un `NeighborhoodManager`
nuevo), así que eliminarlo no cuesta ningún beneficio real de rendimiento,
mientras elimina un bug de corrección grave y silencioso. El caché correcto
a este nivel ya existe una capa arriba, en `NormalManager` (clave por
`id(cloud)`, el objeto estable que sí controla quien llama -- ya auditado y
corregido antes en esta sesión).

**Verificado con 2000 intentos consecutivos del caso exacto de
reproducción: 0 fallas** (antes: reproducible en menos de 10 intentos).

**`tests/processing/test_shared.py` (6 tests)**: incluye el contrato
determinista mínimo que propuso Hernán (`compute_pca()` debe ser función
pura de sus datos reales, no de qué objeto `manager` se le pase), verificado
tanto para el mismo manager llamado dos veces como para dos managers
distintos envolviendo la misma nube, y confirmación explícita de que
`compute_pca()` no muta sus propias entradas.

**Estado: 810/810 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Por qué esto importa para todo lo demás de esta sesión

`PCAFeatures`, `RuleBasedClassifier`, y todo el subsistema de clasificación
ML dependen de `compute_pca()` a través de `PCANormalEstimator`. Antes de
este fix, **cualquier pipeline que procesara más de una nube de puntos en
la misma ejecución corría el riesgo de recibir normales/curvatura de la
nube equivocada, en silencio, de forma intermitente** -- exactamente el
escenario real de procesar un lote de tiles o ejecutar tests repetidos en
el mismo proceso. Este hallazgo confirma que valió la pena no declarar
`processing.classification` (ni ningún módulo que dependa de PCA) como
"cerrado" mientras existiera esta no-determinismo reproducible.

## CHECKPOINT DE CIERRE -- `processing.classification` + revisión cross-module de PR19

Antes de pasar a `processing.registration`, checkpoint formal (no una
auditoría nueva desde cero) verificando cada pieza con evidencia directa:

### 1. Classification -- todo reconfirmado

- Los 7 archivos tocados (`rules.py`, `ml.py`, `manager.py`, y las 4
  subclases concretas) compilan sin error.
- **Persistencia `fit → save → load → classify` reconfirmada con las 4
  implementaciones** (no solo la probada originalmente): RandomForest,
  GradientBoost, XGBoost, LightGBM -- las 4 dan resultados idénticos antes
  y después de guardar/cargar.
- `LRUCache` picklable confirmado con `pickle` directo (no solo a través
  de `joblib`).
- `_PCA_CACHE`: **0 referencias en todo el repo** -- confirmado eliminado
  por completo, no solo desconectado.
- Los 3 archivos de test de regresión (`test_manager.py` 14 tests,
  `test_rules.py` 7 tests, `test_ml.py` 15 tests) y el de `LRUCache`
  (`test_lru_cache.py` 4 tests) confirmados presentes y pasando.

### 2. Cross-module -- verificación específica del alcance del fix de `compute_pca()`

- `ruff` limpio en `_shared.py` tras el fix -- sin imports ni referencias
  muertas.
- **Verificado explícitamente que los DOS consumidores distintos de
  `compute_pca()` (`PCAFeatures` de `features` y `PCANormalEstimator` de
  `normals`) no se contaminan entre sí** -- 300 iteraciones alternando
  ambos consumidores sobre nubes distintas en el mismo proceso, sin ningún
  caso de contaminación. Esto era importante confirmar porque el bug
  original se descubrió a través de `normals`, pero `compute_pca()` es
  compartida por ambos módulos.
- **Búsqueda exhaustiva confirmando que no queda ningún otro caché por
  `id()` sobre objetos efímeros** en todo el repo: solo existen los 4
  `LRUCache` ya auditados (`normals`, `ground`, `features`, `filters`
  managers), todos usando `id(cloud)` -- el objeto estable que controla
  quien llama, no un objeto interno efímero.

### 3. Calidad

- `ruff`/`mypy` sin acotar sobre **todo** `processing` (no solo lo tocado)
  para verificación honesta: confirmado que los ~50 hallazgos de `ruff` y
  los 4 de `mypy` caen casi todos en archivos **nunca tocados** en esta
  sesión (deuda de estilo preexistente, ya documentada desde el inicio
  como fuera de alcance). El único archivo tocado con un hallazgo
  restante (`classification/ml.py`) tiene exactamente el mismo `BLE001`
  preexistente ya identificado antes, sin cambios.
- **Hallazgo anticipado para la próxima auditoría**: `mypy` sin acotar
  reveló 2 errores reales de tipo en `registration/point_to_plane.py`
  (líneas 170-171, `ndarray[floating[Any]]` vs `ndarray[float64]`
  esperado) -- no corregidos aquí (`registration` todavía no tiene su
  turno de auditoría), pero quedan anotados para cuando le toque.
- **Re-escaneo activo de tests intermitentes**: suite completa corrida 5
  veces adicionales (7 en total contando las 2 habituales) -- **810/810
  en las 7 corridas, sin ningún otro test intermitente detectado** más
  allá del ya encontrado y corregido (`compute_pca()`).

### 4. Estado formal de PR19 actualizado

```
PR19 -- QA / Regression / Deep Audit

geodesy                 ✅  (2 bugs reales: UTM zona 61, verificación EPSG propia)
terrain                 ✅  (5 bugs reales: TD-001, Grid bounds, Contours CRÍTICO,
                              DTM excepción no capturada, InterpolationMethod duplicado)
processing
├── normals             ✅  (caché real implementado, fallo latente de diseño evitado)
├── features            ✅  (2 bugs graves: FeatureManager id(cloud) congelado,
│                             RelativeHeightFeatureComputer nunca leía X/Y)
├── ground               ✅  (caché real implementado para classify())
├── sampling              ✅  (2 bugs de seed: StratifiedSampler fijo en 42,
│                              SamplingManager no propagaba seed)
├── filters                ✅  (sin bugs encontrados)
├── segmentation             ✅  (3 bugs reales, 2 CRÍTICOS: radio de crecimiento
│                                 conteo-no-distancia, normales antiparalelas,
│                                 TreeSegmenter/BuildingSegmenter filtraban por Z
│                                 absoluto)
├── classification           ✅  CERRADO (bug crítico sistémico: subsistema ML
│                                  completo inutilizable; + bug CRÍTICO transversal
│                                  no determinista en _shared.compute_pca())
└── registration              ⏳  SIGUIENTE -- 2 errores de tipo ya detectados
                                   en point_to_plane.py, pendientes de la
                                   auditoría propia del módulo

analysis                ⏳  sin empezar
io                      ⏳  sin empezar
export                  ⏳  sin empezar
workflow                ⏳  sin empezar
```

**Bugs reales totales encontrados y corregidos en PR19 hasta este punto: 17**
(2 geodesy + 5 terrain + 2 features + 1 sampling×2 + 3 segmentation +
2 classification/ml + 1 compute_pca transversal + ajustes de caché en
normals/ground). Ninguno se encontró por "que no truene" -- todos por
verificación matemática/geométrica activa o por seguir hasta el fondo un
comportamiento intermitente que habría sido fácil documentar como
"limitación conocida" y dejar pasar.

**810/810 tests en todo el sandbox, confirmado estable en 7 corridas
consecutivas, `ruff`/`mypy` limpios en todo lo tocado.**

**`processing.classification` queda formalmente CERRADO.** Siguiente:
`processing.registration`, con la misma regla de siempre -- no asumir que
está bien porque tenga tests; auditar contratos, matemática, casos
degenerados y comportamiento con datos reales antes de tocar nada.

## `processing.registration` -- arranca. `base.py` sin bugs, bug real de convergencia corregido en `icp.py`

### `base.py` (`Transformation`/`RegistrationResult`) -- verificado sin bugs

`apply_points`/`inverse`/`compose`/`__matmul__` verificados con rotación de
90° + traslación conocida: `apply` da el punto exacto esperado, `inverse`
recupera el original, `compose` coincide exactamente con aplicación
secuencial. Fórmulas de Kabsch en `point_to_point.py` verificadas por
lectura como textualmente correctas (SVD estándar, corrección de reflexión
correcta).

### Bug real: criterio de convergencia nunca dispara cerca del piso de precisión

`ICPBase.register()` usaba **solo** cambio relativo de RMSE para detectar
convergencia. Confirmado con caso sintético de transformación conocida
(rotación 30° + traslación, recuperable exactamente): el RMSE llegaba a
`~2e-14` (piso de precisión de punto flotante) desde la iteración 35 de 50,
pero el cambio relativo fluctuaba entre `1e-4` y `1e-3` -- **siempre por
encima de la tolerancia** (`1e-6`) -- porque a esa escala, ruido de punto
flotante en el numerador dividido por un denominador ya minúsculo produce
cambios relativos artificialmente grandes. Resultado: **cualquier registro
con ajuste casi perfecto agotaba siempre el máximo de iteraciones y
reportaba `converged=False`**, aunque la transformación estimada ya fuera
correcta a precisión de máquina -- una señal equivocada real para
cualquier código que confíe en `result.converged`.

**Corregido**: converge si el cambio relativo es bajo **O** si el RMSE
mismo ya está por debajo de la tolerancia. Verificado: el mismo caso ahora
converge en la iteración 11 (antes: nunca, agotaba las 50), con rotación y
traslación recuperadas exactamente.

**Confirmado que `PointToPlaneICP` no duplica esta lógica** -- delega por
completo en `ICPBase.register()`, así que el mismo fix cubre ambas
variantes sin cambios adicionales.

### Hallazgo nuevo, separado, NO corregido: `PointToPlaneICP` diverge con cualquier offset real

Al verificar el fix de convergencia con ambas variantes, `PointToPlaneICP`
falló con `"Not enough correspondences (0) found"` -- incluso con una
rotación inicial pequeña (5°), no solo la de 30° usada originalmente.
Rastreado: el número de correspondencias **cae progresivamente** iteración
a iteración (`200 → 193 → 187 → 48 → ...`) -- la transformación estimada
en cada paso empeora la alineación en vez de mejorarla. Con `source ==
target` exacto (offset cero) funciona perfectamente (`converged=True`,
`rmse=0.0`), confirmando que el problema es específico de la estimación de
transformación bajo un desplazamiento real, no un bug estructural general.

**No corregido todavía** -- pertenece a la propia auditoría matemática de
`point_to_plane.py` (`_estimate_transformation`/`_build_linear_system`,
la linealización de ángulo pequeño), el siguiente paso natural en el orden
que definió Hernán para este módulo. Queda documentado aquí para no
perderlo, sin bloquear el resto del checkpoint.

**Tests**: `tests/processing/registration/test_base.py` (9 tests),
`test_icp_convergence.py` (3 tests).

**Estado: 822/822 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Estado de `processing.registration`

```
base.py            ✅ verificado, sin bugs
icp.py (bucle)      ✅ bug real de convergencia corregido
point_to_point.py     ✅ verificado end-to-end con transformación conocida
point_to_plane.py      ⚠️ BUG REAL SIN CORREGIR -- diverge con cualquier
                           offset real, investigación pendiente
manager.py               ⏳ sin auditar
```

## Bug real (CRÍTICO) confirmado y corregido: `PointToPlaneICP` divergía por un signo invertido en el sistema lineal

Siguiendo el árbol de investigación exacto que definió Hernán: no se asumió
el signo como causa hasta demostrarlo numéricamente, inspeccionando `A`,
`b`, `x` directamente para casos puros (traslación sola, rotación sola)
antes de tocar el código.

### Investigación

**Caso 1 -- traslación Z pura (+0.5)**: con un plano plano (normal `(0,0,1)`
uniforme) y el source desplazado `+0.5` en Z, la traslación incremental
correcta es `-0.5` (mover el source de vuelta hacia el target). El sistema
resolvía `+0.5` -- signo invertido, magnitud correcta.

**Caso 2 -- rotación Y pura (+1°)**: con un plano extendido rotado
`+1°` alrededor de Y (una esfera resultó ser un caso degenerado sin señal
de rotación -- descartada como elección de prueba, no como hallazgo), el
`omega_y` correcto es `+0.01745` rad. El sistema resolvía `-0.01745` --
signo invertido, magnitud correcta.

**Prueba decisiva**: negar **únicamente** el lado derecho (`b`) del sistema
corrige **ambos** casos exactamente -- confirmando un único error de signo
uniforme en `rhs`, no un problema separado en la convención del producto
cruzado (`cross_terms`), que se dejó sin tocar.

### Causa raíz

```python
rhs = np.einsum("ij,ij->i", normals, source_points - target_points)  # incorrecto
```

debía ser:

```python
rhs = np.einsum("ij,ij->i", normals, target_points - source_points)  # correcto
```

Con el signo invertido, cada iteración de ICP aplicaba la transformación
incremental en la **dirección equivocada**, alejando el source del target
en vez de acercarlo -- explicando exactamente la divergencia observada
(conteo de correspondencias cayendo progresivamente hasta llegar a cero),
reproducible incluso con desplazamientos iniciales pequeños y realistas
(5°), no solo con el caso original de 30°.

**Corregido y verificado**: ambos casos que antes divergían ahora
convergen correctamente -- 5° en 3 iteraciones, 30° (el caso original) en
8 iteraciones, ambos con rotación y traslación recuperadas exactamente.

**Ajuste adicional**: se corrigieron también los 2 errores de tipo en
`point_to_plane.py` ya anticipados en el checkpoint anterior (`solution[:3]`/
`solution[3:]` de `np.linalg.lstsq` con dtype genérico, forzado a
`float64` explícito).

**Tests**: `tests/processing/registration/test_point_to_plane.py` (6 tests)
-- incluye la inspección directa de `A`/`b`/`x` para los dos casos puros
(la evidencia decisiva, no solo el resultado end-to-end), traslaciones
puras en cada eje por separado (según el plan de pruebas de Hernán), y
los dos casos de reproducción completos (5° y 30°).

**Estado: 828/828 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Estado de `processing.registration`

```
base.py            ✅ verificado, sin bugs
icp.py (bucle)      ✅ bug real de convergencia corregido
point_to_point.py     ✅ verificado end-to-end
point_to_plane.py      ✅ bug CRÍTICO de signo corregido y verificado
manager.py               ⏳ siguiente -- último archivo del módulo
```

## `processing.registration` -- CIERRE FORMAL

### `manager.py` -- auditado contra el checklist completo, sin bugs encontrados

- **Contrato público/APIs**: `register()`, `__call__`, `method` (getter/setter),
  `set_params()` -- todos verificados.
- **Sin caché por `id()`**: a diferencia de varios otros managers corregidos
  en esta sesión, `RegistrationManager` crea un registrador (`ICPBase`)
  **fresco en cada llamada** (`_create_registrar()`), sin ningún estado
  compartido entre llamadas -- nunca estuvo en riesgo de la clase de bug
  encontrada en `compute_pca()`/`FeatureManager`/etc.
- **Filtrado de parámetros por algoritmo**: confirmado con `inspect.signature`
  que `normal_k` llega a `PointToPlaneICP` pero se filtra correctamente
  para `PointToPointICP` (que ni siquiera tiene el atributo).
- **Reutilización con múltiples nubes**: mismo manager, dos pares de nubes
  distintos, sin contaminación cruzada -- cada registro recupera su propia
  traslación correcta de forma independiente.
- **Determinismo**: mismo par de nubes, dos llamadas, matriz de
  transformación idéntica bit a bit.
- **Ambos fixes matemáticos alcanzables desde la API pública**: se
  verificó explícitamente que tanto el fix de convergencia (`icp.py`)
  como el fix de signo (`point_to_plane.py`) se reflejan correctamente
  al pasar por `RegistrationManager`, no solo al llamar las clases
  directamente.
- **Manejo de errores**: método no soportado (constructor y setter),
  nubes vacías (source y target por separado) -- todos correctamente
  rechazados con `RegistrationError`.
- La validación de coordenadas del manager solo revisa `X` explícitamente
  (no `Y`/`Z`) -- investigado y confirmado **inofensivo**: `Chunk.__init__`
  ya exige X/Y/Z siempre, haciendo imposible construir una nube real con
  solo X (mismo patrón de invariante-más-abajo ya visto antes en esta
  sesión con `HeightFeatureComputer`).

**Tests**: `tests/processing/registration/test_manager.py` (14 tests).

### Verificación de calidad final del módulo completo

- `mypy` sin acotar sobre los 6 archivos de `registration`: **limpio** --
  confirma que los 2 errores de tipo anticipados en el checkpoint anterior
  (`point_to_plane.py:170-171`) quedaron resueltos junto con el fix del
  signo.
- `ruff` sobre todo el módulo: 2 hallazgos de estilo preexistentes
  (`__all__` sin ordenar en `__init__.py`/`base.py`) corregidos como parte
  del cierre formal.
- **Suite completa corrida 6 veces consecutivas: 842/842 en cada corrida,
  sin ninguna intermitencia detectada.**

### `processing.registration` -- CERRADO

```
base.py            ✅ sin bugs
icp.py               ✅ bug real de convergencia corregido
point_to_point.py      ✅ verificado end-to-end
point_to_plane.py        ✅ bug CRÍTICO de signo corregido
manager.py                 ✅ sin bugs, ambos fixes verificados desde la API pública
```

**Bugs reales encontrados y corregidos en `registration`: 2**
(convergencia en `icp.py`, signo invertido en `point_to_plane.py` --
este último CRÍTICO, causaba divergencia total del algoritmo con
cualquier offset real).

## Estado formal de PR19 actualizado

```
PR19 -- QA / Regression / Deep Audit

geodesy                 ✅
terrain                 ✅
processing
├── normals             ✅
├── features            ✅
├── ground              ✅
├── sampling            ✅
├── filters              ✅
├── segmentation           ✅
├── classification           ✅
└── registration                ✅ CERRADO

analysis                ⏳ sin empezar
io                      ⏳ sin empezar
export                  ⏳ sin empezar
workflow                ⏳ sin empezar
```

**Bugs reales totales encontrados y corregidos en PR19: 19**
(2 geodesy + 5 terrain + 2 features + 2 sampling + 3 segmentation +
2 classification/ml + 1 compute_pca transversal + 2 registration).

**842/842 tests en todo el sandbox, estable en 6 corridas consecutivas,
`ruff`/`mypy` limpios en todo lo tocado durante PR19.**

Con esto, **todo el árbol de `processing` queda cerrado dentro de PR19.**
Siguiente bloque pendiente: `analysis`, `io`, `export`, o `workflow`
(a decidir el orden).

# PR19 -- `analysis` arranca

## `_shared/surface.py` -- verificado sin bugs

Validación de coordenadas finitas e interpolación con manejo de errores
correctos.

## Bug real: `CutFillVolume`/`GridVolume` inutilizables con cualquier DTM real

`_shared/volume.py` (`validate_volume_arrays`) rechazaba por completo
cualquier grilla que contuviera **algún** valor `NaN`, sin importar cuántas
celdas fueran válidas. Confirmado con un DTM real generado por
`DTM.from_tin()` (el mismo código que corregimos antes en esta sesión para
dejar `NaN` legítimamente en celdas fuera del casco convexo del TIN
triangulado -- una forma común y esperada de cualquier superficie
triangulada real, no un error): `CutFillVolume.compute()` lanzaba
`VolumeError: Existing surface contains invalid elevations.` aunque la
mayoría de la grilla tuviera datos perfectamente válidos.

**Impacto**: cualquier cálculo de cut/fill sobre una superficie triangulada
real con borde irregular (esencialmente cualquier superficie topográfica
real, dado que los levantamientos perfectamente rectangulares son raros)
fallaba por completo, a menos que el usuario recortara manualmente su
grilla para eliminar toda celda NaN -- justo el tipo de problema que
convenía resolver antes de construir Surface Comparison/Cut-Fill de PR20
encima de este código.

**Corregido**: las celdas NaN ahora se **excluyen** del cálculo (se suma
solo sobre el área donde ambas superficies tienen datos válidos), en vez
de rechazar todo el cómputo. Se sigue rechazando lo genuinamente inválido:
valores infinitos (nunca un marcador legítimo de NoData en este código
base) y superficies completamente NaN (nada que calcular). Se agregaron
los campos `valid_cells`/`excluded_cells` a `VolumeResult` (con default
`None` para no romper los otros 3 constructores existentes de
`VolumeResult` en `average_end_area.py`/`prismoidal.py`/`tin_volume.py`,
que no fueron tocados) para que el llamador sepa cuánta cobertura real
tuvo el cálculo, en vez de recibir un volumen silenciosamente parcial sin
ninguna señal.

`compute_cut_fill()` cambió su firma de retorno de una tupla de 3 a una de
5 elementos -- actualizados los 2 llamadores existentes (`cut_fill.py`,
`grid_volume.py`).

**Tests**: `tests/analysis/_shared/test_volume.py` (14 tests) y
`tests/analysis/volume/test_cut_fill_grid_volume.py` (7 tests) -- incluye
el caso de reproducción exacto con DTM real, verificación de que NaN se
excluye (no se trata como cero, lo cual daría un resultado numérico
distinto e incorrecto), y confirmación de que los rechazos genuinos
(todo-NaN, infinito) se preservan.

**Estado: 863/863 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### Progreso de `analysis`

```
_shared/surface.py     ✅ sin bugs
_shared/volume.py        ✅ bug real corregido (NaN)
volume/cut_fill.py         ✅ bug real corregido (mismo fix)
volume/grid_volume.py        ✅ bug real corregido (mismo fix)
volume/tin_volume.py           ⏳ pendiente (¿mismo patrón de NaN?)
volume/average_end_area.py       ⏳ pendiente
volume/prismoidal.py               ⏳ pendiente
volume/manager.py                    ⏳ pendiente
distance/, profile/, visibility/,
statistics/, quality/, config.py,
protocols.py, types.py (resto)         ⏳ pendiente
```

## `analysis/volume` -- MÓDULO COMPLETO

Siguiendo la instrucción explícita de no asumir que los demás métodos
comparten el bug de NaN de `cut_fill`/`grid_volume` -- verificado
individualmente cada uno:

### `tin_volume.py` -- verificado sin bugs, y confirmado que NO comparte el problema de NaN

Dominio fundamentalmente distinto (triángulos discretos, no una grilla).
Confirmado que `TIN.from_points()` ya rechaza `NaN` en la construcción --
un `TriangulatedSurface` real **no puede** tener vértices NaN en este
código base, así que el chequeo defensivo de `TINVolume._validate_vertex`
es inalcanzable en la práctica (mismo patrón ya visto con `Chunk`
exigiendo Z siempre). Matemática de integración de prisma triangular
verificada con casos conocidos (área=6, altura=5 → volumen=30 exacto).

### `average_end_area.py` -- verificado sin bugs

Dominio 1D (pares estación/área a lo largo de un corredor), sin concepto
de celda NaN. Matemática verificada con casos constante y trapezoidal
conocidos.

### Hallazgo real (documentado, no corregido): `PrismoidalVolume` equivale exactamente a `AverageEndAreaVolume`

Confirmado con evidencia numérica: dado el mismo input, ambas clases
devuelven `cut_volume` **bit a bit idéntico**. Causa: el área de la
sección intermedia se aproxima como `(A1+A2)/2` (el promedio de los
extremos) en vez de aceptar una medición real de la sección intermedia --
lo que reduce algebraicamente la fórmula prismoidal (regla de Simpson) a
ser exactamente la fórmula del área media:
`L/6*(A1+4*(A1+A2)/2+A2) == L/2*(A1+A2)`.

Todo el propósito de la regla de Simpson es capturar curvatura/variación
no lineal mediante una sección intermedia genuinamente medida -- esta
implementación descarta exactamente esa información. Confirmado que el
problema llega hasta la API pública: `VolumeAnalysis.prismoidal()` (el
manager) usa la misma firma de pares que `average_end_area()`, sin forma
de recibir una sección intermedia real.

**Decisión de Hernán**: no cambiar la API en esta sesión (requeriría
aceptar tercias con áreas medidas reales en las 3 estaciones, un cambio
mayor que amerita su propia decisión de diseño separada). **Documentado
explícitamente** en el docstring del módulo, de la clase, y del método
`VolumeAnalysis.prismoidal()` -- para que no se elija "prismoidal" en vez
de "average_end_area" esperando resultados distintos sin saber que hoy
son equivalentes.

### `manager.py` (`VolumeAnalysis`) -- verificado sin bugs

Los 5 métodos (`cut_fill`, `grid_volume`, `average_end_area`,
`prismoidal`, `tin_volume`) dispatch correctamente vía `compute()`.
Confirmado que el fix de exclusión de NaN se propaga correctamente a
través del manager, no solo llamando las clases directamente.

**Tests**: `tests/analysis/volume/test_tin_average_prismoidal.py`
(13 tests) y `test_manager.py` (10 tests).

**Estado: 887/887 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### `analysis/volume` -- CERRADO

```
_shared/volume.py     ✅ bug real corregido (NaN)
cut_fill.py              ✅ bug real corregido (mismo fix)
grid_volume.py              ✅ bug real corregido (mismo fix)
tin_volume.py                   ✅ sin bugs, confirmado NO comparte el problema
average_end_area.py                 ✅ sin bugs
prismoidal.py                           ⚠️ limitación real documentada, no corregida
manager.py                                  ✅ sin bugs
```

**Siguiente, según el orden de Hernán**: `analysis/distance`.

## `analysis/distance` -- MÓDULO COMPLETO. Bug real de orden de parámetros corregido

### `euclidean.py`, `horizontal.py`, `vertical.py`, `slope.py`, `geodesic.py` -- verificados sin bugs

Matemática verificada con casos conocidos: triángulos 3-4-5 y 3-4-12-13
en 2D/3D, pendiente de 45° exacta (gradiente=100%, ángulo=45°),
distancia geodésica de 1° de longitud en el ecuador WGS84 (~111319m,
delegando en `geodesy.geodesic` ya auditado y cerrado esta sesión).

### Bug real: `DistanceAnalysis.compute()` daba distancias 3D equivocadas en silencio

`EuclideanDistance.compute()` tiene un orden de parámetros **distinto**
al resto del módulo: `(x1,y1,x2,y2,z1,z2)` ("por eje"), mientras que
`SlopeDistance.compute()`/`GeodesicDistance.compute()` usan el orden
natural "por punto" `(x1,y1,z1,x2,y2,z2)`. El dispatcher del manager
reenviaba los 6 argumentos posicionales directamente
(`self._euclidean_3d.compute(*args)`), asumiendo que coincidían --
confirmado con evidencia directa: `compute(0,0,0,3,4,12)` (triángulo
3-4-12-13, distancia esperada `13.0`) devolvía en silencio `8.544`, sin
ningún error.

**Pista decisiva que confirmó el enfoque de corrección**: `slope.py`
llama `SlopeDistance._ENGINE.compute(x1,y1,x2,y2,z1,z2)` -- **reordenando
explícitamente** antes de delegar en `EuclideanDistance`. Esto confirma
que el orden de `EuclideanDistance` es el "canónico" que el resto del
código ya sabe respetar, y que **solo el manager** tenía el reenvío
ingenuo. **Corregido replicando exactamente ese mismo patrón** en
`DistanceAnalysis.compute()` -- sin tocar la API pública de
`EuclideanDistance` (verificado que ningún otro sitio del repo la llama
posicionalmente con el orden equivocado).

**Decisión de Hernán sobre el rechazo total de NaN en `compute_many()`**
(presente en las 4 clases de punto-a-punto: euclidean/horizontal/
vertical/slope): se deja como está -- fallar rápido ante cualquier punto
inválido es un diseño razonable para arrays de puntos discretos, distinto
del caso de superficies/grillas (`volume/`) donde sí había evidencia
concreta y demostrada de un escenario real y común (bordes de casco
convexo de un DTM) que se rompía. No se cambia.

**Tests**: `tests/analysis/distance/` -- `test_euclidean.py` (8),
`test_horizontal_vertical_slope.py` (8), `test_geodesic.py` (5),
`test_manager.py` (14) = 35 tests.

**Estado: 922/922 tests en todo el sandbox (dos corridas consecutivas),
`ruff`/`mypy` limpios.**

### `analysis/distance` -- CERRADO

```
euclidean.py    ✅ sin bugs
horizontal.py     ✅ sin bugs
vertical.py         ✅ sin bugs
slope.py               ✅ sin bugs (además revela el patrón correcto de reordenación)
geodesic.py               ✅ sin bugs
manager.py                   ✅ bug real de orden de parámetros corregido
```

**Siguiente, según el orden de Hernán**: `analysis/profile`.

## `analysis/profile` -- MÓDULO COMPLETO. Dos bugs reales corregidos

### `longitudinal.py` -- verificado sin bugs

`_generate_stations()` verificado con división exacta, no exacta, y el
caso de riesgo clásico de precisión de punto flotante
(`interval=0.1` sobre `axis=10.0`, que en muchos lenguajes produciría
99 en vez de 100 pasos por error de representación) -- confirmado
correcto: 101 estaciones exactas. Perfil completo verificado contra un
plano inclinado conocido (`z=x`), interpolación exacta en todos los
puntos.

### Bug real (CRÍTICO para uso en ingeniería civil): `TransversalProfile` podía omitir el eje (offset=0)

`_generate_offsets()` avanzaba desde `-width` en incrementos de
`interval`, lo que solo tocaba exactamente `offset=0` (la línea de eje
misma -- el punto de referencia más importante de cualquier sección
transversal) cuando `width` era múltiplo exacto de `interval`. Confirmado
con `width=10, interval=3`: los offsets generados eran
`[-10,-7,-4,-1,2,5,8,10]`, con los puntos más cercanos a cero siendo
`-1` y `+2` -- rodeando el eje sin nunca tocarlo. Esto rompía la garantía
que `LongitudinalProfile` ya ofrece para su punto equivalente
(`station=0` siempre presente por construcción), y se propagaba a
`CrossSectionProfile` (que delega en `TransversalProfile` en cada vértice
de un eje de alineamiento -- el flujo de trabajo más usado en topografía
vial).

**Confirmado que `MultiProfile` NO comparte el problema** -- usa offsets
explícitos dados por el llamador (por defecto `(0.0,)`), no auto-genera
una grilla.

**Corregido** construyendo los offsets hacia afuera desde 0 en el lado
positivo (0, interval, 2·interval, ..., hasta width, con el extremo
exacto agregado si no se alcanzó ya), luego reflejando los valores
estrictamente positivos al lado negativo -- el 0 queda incluido por
construcción, exactamente una vez, sin importar si `width` es múltiplo
exacto de `interval`. Esto también corrigió de paso el caso especial
`interval >= width` (que antes devolvía solo `[-width, width]`, también
sin el eje) -- ahora manejado por la misma lógica general, sin rama
separada.

**Verificado con todos los criterios pedidos**: eje presente exactamente
una vez, ambos extremos `±width` preservados, orden creciente, sin
duplicados, tolerancia numérica apropiada, comportamiento existente
preservado cuando `width` sí es múltiplo exacto del intervalo, y
`CrossSectionProfile` heredando la corrección en los 3 vértices de un eje
de prueba.

### Bug real (encontrado al escribir el propio test suite): `ProfileAnalysis.compute()` dejaba escapar `ValueError` crudo

`compute()` construía el enum `ProfileMethod` directamente
(`ProfileMethod(method or ...)`), sin el mismo chequeo que `__init__()`
ya hacía correctamente -- dejando escapar un `ValueError` crudo de Python
para un método inválido, en vez de la excepción propia del módulo
(`ProfileError`), rompiendo el contrato de excepciones de `analysis` (e
inconsistente con `DistanceAnalysis`/`VolumeAnalysis`, que ya envuelven
esto correctamente, verificado antes en esta sesión). Corregido con
`try/except` envolviendo la construcción del enum.

### Hallazgo de tipado (no un bug de código fuente): `LinearInterpolator` no satisface `TerrainSurface`

El protocolo `TerrainSurface` (usado por todo `analysis.profile`) exige
tanto `interpolate()` como `contains()`. `topocore.terrain.linear.
LinearInterpolator` (usado extensamente en mis pruebas de esta sesión)
solo implementa `interpolate()` -- funcionaba en tiempo de ejecución
porque ningún camino de código ejercitado llama `contains()`, pero
`mypy` correctamente señala la discrepancia estructural. No es un bug
del código fuente de `terrain` ni de `analysis` -- es que
`LinearInterpolator` nunca fue pensado para SER un `TerrainSurface`
completo. Se creó `tests/analysis/profile/_helpers.py::SurfaceAdapter`,
un adaptador mínimo de test que delega `contains()` en `TIN.contains()`
(ya existente), para tener pruebas genuinamente conformes al protocolo.

**Tests**: `tests/analysis/profile/` -- `test_longitudinal.py` (9),
`test_transversal.py` (9), `test_cross_section_multi.py` (7),
`test_manager.py` (6) = 31 tests.

**Estado: 953/953 tests en todo el sandbox, estable en 4 corridas
consecutivas, `ruff`/`mypy` limpios.**

### `analysis/profile` -- CERRADO

```
longitudinal.py    ✅ sin bugs
transversal.py        ✅ bug real CRÍTICO corregido (eje ausente)
cross_section.py         ✅ hereda la corrección
multi_profile.py            ✅ confirmado sin el problema (offsets explícitos)
manager.py                     ✅ bug real corregido (ValueError crudo)
```

**Siguiente, según el orden de Hernán**: `analysis/visibility`.

## `analysis/visibility` -- MÓDULO COMPLETO. Dos bugs reales corregidos

### Bug real (CRÍTICO): fórmula de curvatura terrestre matemáticamente equivocada en `LineOfSight`

`_curvature_correction()` calculaba `d1²/(2R)` usando **solo** la distancia
desde el observador -- una fórmula que **crece sin límite** al acercarse
al objetivo, en vez de volver correctamente a cero en ambos extremos del
trayecto (tanto la elevación dada del observador como la del objetivo ya
son datos correctos, no necesitan corrección adicional en su propia
posición). La fórmula estándar y correcta de "abultamiento terrestre"
(usada en ingeniería de radioenlaces/microondas y estudios de visibilidad
topográfica) es la forma de **producto** `d1·d2/(2R)` -- cero cuando
`d1=0` (en el observador) o `d2=0` (en el objetivo), máxima en el punto
medio.

**Confirmado directamente contra la fórmula clásica de distancia al
horizonte** (`d=√(2Rh)`, ~4.65km para altura de ojos de 1.7m): la fórmula
anterior reportaba un objetivo a nivel de suelo como ya invisible a solo
1km. **Corregido y verificado**: el objetivo permanece visible hasta
~4.65km y se vuelve invisible justo después (verificado en 5km), con
precisión notable respecto a la fórmula clásica. La corrección ahora es
simétrica, máxima en el punto medio, y exactamente cero en ambos extremos
del trayecto.

**Alcance confirmado**: tanto `Viewshed` como `Intervisibility` delegan
directamente en `LineOfSight.compute()` internamente sin implementación
propia de curvatura -- el fix en un solo lugar (`los.py`) corrige
correctamente los 3 consumidores.

### Bug real: `VisibilityAnalysis.viewshed()` ignoraba la configuración de curvatura

`viewshed()` del manager nunca pasaba `earth_curvature` al construir
`Viewshed(...)`, usando siempre el valor por defecto (`True`) sin importar
lo que `VisibilityConfig.earth_curvature_correction` realmente dijera --
a diferencia de `line_of_sight()`/`intervisibility()`, que sí lo propagan
correctamente. Confirmado con evidencia dramática: configurando
`earth_curvature_correction=False` sobre una superficie plana enorme a
6km de distancia máxima, el manager daba `69/113` celdas visibles (la
curvatura se seguía aplicando) en vez de las `113/113` correctas
(coincidiendo con un `Viewshed(earth_curvature=False)` construido
directamente). **Corregido** propagando el parámetro.

### Resto del módulo -- verificado sin bugs adicionales

`LineOfSight`: detección de obstáculos (muro), terreno plano visible,
determinismo, puntos coincidentes, puntos fuera del TIN -- todo correcto.
`Viewshed`: terreno plano 100% visible, muro reduce visibilidad
sensatamente, celdas fuera del TIN correctamente excluidas del conteo.
`Intervisibility`: matriz simétrica confirmada (válido porque esta clase
siempre usa la misma altura para observador y objetivo), geometría de
obstáculo verificada incluyendo un caso diagonal donde mi expectativa
inicial estaba equivocada, no el código (la diagonal también cruza la
posición del muro, confirmado analíticamente).

**Tests**: `tests/analysis/visibility/` -- `test_los.py` (13, incluye
prueba directa de la fórmula de curvatura), `test_viewshed.py` (6),
`test_intervisibility.py` (6), `test_manager.py` (8) = 33 tests. Se creó
`_helpers.py::SurfaceAdapter` (mismo patrón que `profile/_helpers.py`,
extendido con `triangle_count`/`find_triangle`/`triangle_vertices`/
`bounds` para satisfacer completamente `TriangulatedSurface`).

**Estado: 986/986 tests en todo el sandbox, estable en 4 corridas
consecutivas, `ruff`/`mypy` limpios.**

### `analysis/visibility` -- CERRADO

```
los.py             ✅ bug real CRÍTICO corregido (fórmula de curvatura)
viewshed.py           ✅ sin bugs propios
intervisibility.py       ✅ sin bugs
manager.py                  ✅ bug real corregido (curvatura no propagada)
```

**Siguiente, según el orden de Hernán**: `analysis/statistics`.

## `analysis/statistics` -- MÓDULO COMPLETO. Un bug real corregido

### `area.py` -- verificado sin bugs

Fórmula de área 3D (producto cruz) y proyectada (shoelace) verificadas
con triángulo horizontal (superficie=proyectada exactamente) y plano
inclinado a 45° (relación conocida `superficie/proyectada = √2`) --
ambas exactas.

### `density.py` -- verificado sin bugs

Densidad = puntos/área de celda verificada con casos conocidos (4 puntos
en 1 celda → densidad 4.0) y el caso degenerado de puntos coincidentes.

### Bug real (SEVERO para muestras pequeñas): `skewness`/`kurtosis` mezclaban convenciones estadísticas

*(Corregido en un tramo anterior de esta sesión; documentado y cubierto
con tests formales en este cierre.)*

La fórmula mezclaba desviación estándar **muestral** (`ddof=1`, corrección
de Bessel) en el denominador con momentos **poblacionales** (dividir por
N, no N-1) en el numerador -- una convención híbrida que no coincide con
ningún estándar reconocido (ni `scipy.stats.skew/kurtosis(bias=True)` ni
`bias=False`). Para muestras grandes la discrepancia era pequeña (~0.3%
con n=1000), pero para muestras pequeñas -- comunes en estadística
topográfica, p.ej. un puñado de puntos de levantamiento -- se volvía
severa: **confirmado con n=6, la kurtosis difería en 1676% y cambiaba de
signo** (`-0.88` vs `+0.056`), lo que habría reportado una forma de
distribución cualitativamente equivocada (platicúrtica vs leptocúrtica).

Confirmado por búsqueda que ningún otro código del repositorio lee
`DistributionStats.skewness`/`.kurtosis`, así que no había una convención
existente que preservar. **Corregido** usando estadística poblacional de
forma consistente en todo (`ddof=0`), coincidiendo exactamente con
`scipy.stats.skew/kurtosis(bias=True)` -- verificado con n=6 y n=1000,
coincidencia exacta hasta 1e-9 en ambos casos.

### `elevation.py` -- verificado sin bugs

Estadísticas descriptivas con exclusión correcta de NaN/infinito (patrón
compatible con DTMs de casco convexo irregular).

### `slope.py` -- verificado sin bugs, grados confirmados (no radianes)

Verificado con plano de pendiente conocida (`z=x`, resolución=1.0) dando
exactamente `45.0°` en **toda** la grilla (no solo el centro). Confirmado
que aunque `dy`/`dx` de `np.gradient` pudieran estar "invertidos" en el
nombre, el resultado final usa `hypot(dx,dy)` -- simétrico en sus dos
argumentos -- así que no habría impacto real en la magnitud de la
pendiente calculada. Propagación de NaN verificada (patrón DTM).

### `manager.py` -- verificado sin bugs

Los 5 métodos (`elevation`, `slope`, `area`, `density`, `distribution`)
dispatch correctamente, con propagación correcta de parámetros
(`num_bins`, `resolution`) -- a diferencia de otros managers auditados
antes en esta sesión (`distance`, `visibility`), aquí no se encontró
ningún patrón de "parámetro olvidado" ni reordenamiento posicional
ingenuo.

**Tests**: `tests/analysis/statistics/` -- `test_area.py` (4),
`test_density.py` (6), `test_distribution.py` (9, incluye la regresión
exacta del bug de skewness/kurtosis con n=6 y n=1000 contra scipy),
`test_elevation.py` (4), `test_slope.py` (7), `test_manager.py` (9) = 39
tests.

**Estado: 1025/1025 tests en todo el sandbox, estable en 4 corridas
consecutivas, `ruff`/`mypy` limpios.** (10 warnings de `DeprecationWarning`
en `joblib.numpy_pickle` -- confirmado que provienen de los tests de
`ClassificationManager.save()`/`.load()` ya cerrados, biblioteca de
terceros, fuera del alcance de esta sesión.)

### `analysis/statistics` -- CERRADO

```
area.py          ✅ sin bugs
density.py          ✅ sin bugs
distribution.py        ✅ bug real SEVERO corregido (skewness/kurtosis)
elevation.py               ✅ sin bugs
slope.py                       ✅ sin bugs, grados confirmados
manager.py                        ✅ sin bugs
```

**Siguiente, según el orden de Hernán**: `analysis/quality`.

## `analysis/quality` -- MÓDULO COMPLETO. Síntesis honesta: hallazgos que se me escaparon, corregidos por Hernán, más un bug adicional encontrado en la corrección

### Contexto: sincronización a mitad de auditoría

A mitad de la auditoría de `quality/` (tras verificar `rmse.py`,
`hausdorff.py`, `chamfer.py`, `c2c.py`, `c2m.py`, `completeness.py`,
`correctness.py`, `precision.py`, `gps_control.py`, `registration.py` y
`manager.py` como "sin bugs"), Hernán subió un volcado fresco de su
repositorio real que revelaba correcciones **independientes, ya
aplicadas** a varios de estos archivos -- corrigiendo errores reales que
mi propia auditoría no detectó, o que detectó y clasificó
**incorrectamente** como comportamiento correcto.

### Hallazgo #1 (GRAVE, me equivoqué): confundí "sin correspondencia" con "excluible", exactamente lo que se pidió evitar

`c2c.py`, `c2m.py` y `registration.py` originalmente **excluían**
silenciosamente los puntos sin correspondencia dentro de `max_distance`
(o residuos NaN) de las estadísticas resumen. Yo verifiqué este
comportamiento y lo clasifiqué como "diseño correcto, apropiado al
dominio" -- razonando que "sin correspondencia" es distinto de "dato
corrupto".

**Estaba equivocado.** El repo real corrige esto: ahora **rechaza por
completo** el cómputo si algún punto queda sin correspondencia, con el
razonamiento explícito en el código: *"Silently dropping points outside
max_distance would make the quality result look better than the actual
coverage of the correspondence"* / *"Silently removing invalid values
can hide failures and artificially increase the fitness score."*

La distinción correcta, que se me escapó: para `volume/` (dominio de
superficies), NaN/sin-dato significa genuinamente "no hay nada físico
que medir ahí" -- excluir es correcto. Pero para `quality/`
(dominio de evaluación de calidad), "sin correspondencia dentro del
radio de búsqueda" **es en sí mismo una señal de mala calidad**
(cobertura pobre, mal registro) que debe quedar **visible** en el
resultado, no promediada silenciosamente fuera de la estadística --
ocultarla infla artificialmente la calidad reportada. Aplicé la
distinción correcta en `volume/` pero fallé en aplicarla un nivel más
profundo aquí, exactamente en el punto que Hernán pidió vigilar
explícitamente.

### Hallazgo #2 (real, se me escapó por completo): fórmula de intervalo de confianza incorrecta en `precision.py`

Mi versión usaba `margen = t_value * desviación_estándar`. La fórmula
correcta para un intervalo de confianza sobre la **media estimada** es
`margen = t_value * desviación_estándar / √n` (error estándar de la
media, no la desviación estándar cruda) -- un error estadístico clásico
que confunde el intervalo de confianza de la media con un intervalo de
predicción para una observación individual. Yo verifiqué el valor de
`t.ppf()` contra `scipy` pero nunca verifiqué la fórmula completa contra
el estándar de un intervalo de confianza para la media -- un vacío real
en mi propia verificación. Confirmado numéricamente: para n=3,
s=√2, mi fórmula daba margen=6.08; la correcta da margen=3.51.

También se agregó un parámetro `confidence_level` configurable
(propagado correctamente desde `QualityConfig.confidence_level`, que ya
existía en `config.py` pero nunca se conectaba -- el mismo patrón de
"parámetro no propagado" encontrado varias veces antes en esta sesión).

### Hallazgo #3 (nuevo, encontrado por mí en la propia corrección de Hernán): `PrecisionResult.confidence_level` hardcodeado

Al re-verificar la corrección de Hernán con `confidence_level=0.90`, el
margen se calculaba correctamente, pero el campo `confidence_level` del
resultado **seguía reportando `0.95` hardcodeado**, ignorando por
completo `self._confidence_level`. **Confirmado y corregido**: ahora usa
`self._confidence_level` en la construcción de `PrecisionResult`.

### Resto de la sincronización

`chamfer.py`/`hausdorff.py`/`completeness.py` ganaron validación
NaN/Inf/infinito en las entradas (endurecimiento razonable, sin
controversia). `c2m.py` ganó un caché privado por instancia
(`_cache_tin_id`/`_cache_index`, clave `id(tin)`) -- revisado con cuidado
dado el patrón de bugs de `id()` encontrado varias veces esta sesión;
juzgado de **menor riesgo** y explícitamente documentado en el propio
código (TINs son objetos de larga vida mantenidos por el llamador, a
diferencia de los objetos efímeros que causaron el bug de
`compute_pca()`). Verificado directamente que el caché reconstruye
correctamente al cambiar de TIN, sin reutilización obsoleta.

### Archivos NO afectados por la sincronización, confirmados sin bugs

`rmse.py`, `correctness.py`, `gps_control.py` -- idénticos entre mi
sandbox y el repo real, ambos ya correctos.

**Tests**: reescrito por completo `tests/analysis/quality/` reflejando
el comportamiento corregido -- `test_rmse.py` (7), `test_hausdorff_chamfer.py`
(8), `test_completeness_correctness.py` (8), `test_c2c_c2m.py` (13,
incluye verificación explícita del nuevo caché de TIN),
`test_precision_gps_registration.py` (14, incluye ambos hallazgos #2 y
#3), `test_manager.py` (12) = 62 tests.

**Estado: 1087/1087 tests en todo el sandbox, estable en 4 corridas
consecutivas, `ruff`/`mypy` limpios.**

### `analysis/quality` -- CERRADO

```
rmse.py            ✅ sin bugs (confirmado idéntico al repo real)
hausdorff.py          ✅ endurecido (validación NaN/Inf agregada)
chamfer.py               ✅ endurecido (validación NaN/Inf agregada)
completeness.py             ✅ endurecido (validación NaN/Inf agregada)
correctness.py                 ✅ sin bugs (confirmado idéntico)
gps_control.py                    ✅ sin bugs (confirmado idéntico)
c2c.py                                ✅ corregido: rechaza en vez de excluir
c2m.py                                   ✅ corregido: rechaza + caché nuevo revisado
precision.py                                ✅ 2 bugs reales corregidos (fórmula + hardcodeo)
registration.py                                ✅ corregido: rechaza en vez de excluir
manager.py                                        ✅ confidence_level ahora propagado
```

### Estado formal de PR19 actualizado

```
PR19 -- QA / Regression / Deep Audit

geodesy                 ✅
terrain                 ✅
processing              ✅
analysis
├── volume              ✅
├── distance             ✅
├── profile               ✅
├── visibility              ✅
├── statistics                ✅
└── quality                      ✅ CERRADO

analysis (resto)        ⏳ config.py, protocols.py, types.py
io                       ⏳
export                   ⏳
workflow                 ⏳
```

**Bugs reales totales encontrados y corregidos en PR19: 23**
(19 anteriores + 4 en `analysis/quality`: exclude→reject en
c2c/c2m/registration -- contado como 1 patrón corregido en 3 archivos --,
fórmula de intervalo de confianza, `confidence_level` hardcodeado,
`confidence_level` no propagado en el manager).

**1087/1087 tests en todo el sandbox, estable en 4 corridas
consecutivas, `ruff`/`mypy` limpios.**

Con esto, quedan únicamente `config.py`, `protocols.py`, `types.py` para
cerrar formalmente todo `analysis` dentro de PR19.

## `analysis/config.py`, `protocols.py`, `types.py` -- CIERRE FINAL DE `analysis`

Nota: el mismo volcado real (`repo-to-text_2026-08-19-18-13-42-UTC.txt`)
fue reenviado por Hernán; verificado con hash MD5 idéntico al ya
procesado -- confirmado que no había contenido nuevo que sincronizar más
allá de lo ya aplicado a `quality/`. Se confirmó además, comparando AST
(no solo texto), que las diferencias restantes entre el sandbox y el
volcado eran puramente de formato (`ruff format` aplicado por mí), no de
sustancia -- excepto el fix ya propio de `precision.py`
(`confidence_level` hardcodeado), que aún no está reflejado en el
repositorio real de Hernán.

### `config.py` -- verificado sin bugs

Los 6 sub-configs (`Distance`, `Volume`, `Profile`, `Visibility`,
`Statistics`, `Quality`) y `AnalysisConfig` verificados con casos límite
de validación (todos los `ValueError` esperados en valores negativos,
cero, o fuera de rango). Confirmado que los valores por defecto de
`AnalysisConfig` (instancias directas de sub-configs `frozen=True`) son
seguros de compartir -- sin el riesgo clásico de "default mutable" de
Python, dado que las sub-configs son inmutables.

### `protocols.py` -- verificado, sin lógica propia que auditar

Contratos `Protocol` puros (tipificación estructural), sin estado ni
cómputo -- ya verificados implícitamente en cada `mypy` limpio de todo
`analysis` a lo largo de la sesión.

### `types.py` -- verificado, un bug real menor corregido

Revisadas todas las propiedades/métodos con lógica real (no solo acceso
a campos): `DistanceResult.meters`, `ProfileResult.distances/elevations/
offsets`, `ViewshedResult.visibility_ratio`, `IntervisibilityResult.
visibility_ratio` (ambas con guarda correcta contra división por cero),
`ElevationStats.percentiles()` (verificado con valores conocidos y
exclusión de NaN), `HausdorffResult.symmetric`/`ChamferResult.symmetric`
(alias correctos de campos ya verificados).

**Bug real encontrado**: `QualitySummary` es una clase pública real y
funcional, pero estaba **ausente de `__all__`** -- `from
topocore.analysis.types import *` la omitía silenciosamente, y
cualquier herramienta que dependa de `__all__` para enumerar la API
pública del módulo no la vería. No se usa aún en ningún otro lugar del
código, así que es una omisión pura, no un bug funcional alcanzable.
**Corregido**, agregada a `__all__` con comentario explicativo.

**Tests**: `tests/analysis/test_config.py` (15), `test_types.py` (11,
incluye la regresión de `QualitySummary` en `__all__`),
`test_protocols.py` (3) = 29 tests.

**Estado: 1116/1116 tests en todo el sandbox, estable en 4 corridas
consecutivas, `ruff`/`mypy` limpios.**

## `analysis` -- CERRADO FORMALMENTE EN PR19

```
analysis/
├── volume        ✅
├── distance         ✅
├── profile             ✅
├── visibility             ✅
├── statistics                ✅
├── quality                      ✅
├── config.py                       ✅
├── protocols.py                       ✅
└── types.py                              ✅
```

## Estado formal de PR19 actualizado

```
PR19 -- QA / Regression / Deep Audit

geodesy        ✅
terrain        ✅
processing     ✅
analysis       ✅ CERRADO
io             ⏳ siguiente
export         ⏳
workflow       ⏳
```

**Bugs reales totales encontrados y corregidos en PR19: 24**
(23 anteriores + 1 en `types.py`: `QualitySummary` ausente de `__all__`).

**1116/1116 tests en todo el sandbox, estable en 4 corridas consecutivas,
`ruff`/`mypy` limpios en todo lo tocado durante PR19.**

Con `analysis` cerrado por completo, siguiente bloque de PR19: `io`.
