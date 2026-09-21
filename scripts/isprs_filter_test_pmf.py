"""
Corre PMF (TopoCore, implementacion nativa) sobre un sitio real del benchmark ISPRS Filter
Test (Sithole & Vosselman, 2004) y calcula las metricas EXACTAS de
la literatura -- Type I y Type II error -- no precision/recall
genericas.

Definiciones (segun el reporte oficial del benchmark, ITC/TU Delft):
    Type I error  = % de puntos de TERRENO real que PMF clasifico
                    mal como objeto (omision -- terreno perdido)
    Type II error = % de puntos de OBJETO real que PMF clasifico
                    mal como terreno (comision -- falso terreno)
    Total error   = % de puntos totales mal clasificados
    Kappa         = Indice Kappa de Cohen (Jensen, 2005) -- el mismo
                    usado en la literatura de seguimiento del
                    benchmark (Meng et al. 2009) para comparar contra
                    los valores YA PUBLICADOS por sitio -- confirmado
                    necesario para una comparacion directa, en las
                    mismas unidades que la literatura (Type I/II por
                    si solos no son directamente comparables contra
                    valores Kappa publicados).

Formato real de los datos (confirmado en la pagina oficial del
benchmark):
    Nube:       X1 Y1 Z1 I1 X2 Y2 Z2 I2  (primer/ultimo retorno)
    Referencia: X Y Z 0/1                  (0=terreno, 1=objeto)

Uso:
    python isprs_filter_test.py nube.txt referencia.txt [--cloth-resolution N]
"""

import argparse
from pathlib import Path

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.ground.pmf import PMFGroundClassifier


def _leer_nube_isprs(ruta: Path) -> PointCloud:
    """
    Lee un archivo de nube en el formato real del benchmark ISPRS --
    cada linea trae 2 puntos (primer y ultimo retorno). Se usa el
    ULTIMO retorno de cada par (X2,Y2,Z2) -- el mas probable de
    corresponder al terreno real bajo vegetacion, consistente con
    el objetivo de filtrado de terreno.
    """
    xs, ys, zs = [], [], []
    with open(ruta) as f:
        for linea in f:
            valores = linea.split()
            if len(valores) < 8:
                continue
            xs.append(float(valores[4]))
            ys.append(float(valores[5]))
            zs.append(float(valores[6]))

    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    z = np.array(zs, dtype=np.float64)

    chunk = Chunk(size=len(x), attributes=(PointAttribute.X, PointAttribute.Y, PointAttribute.Z))
    chunk._data[PointAttribute.X][:] = x
    chunk._data[PointAttribute.Y][:] = y
    chunk._data[PointAttribute.Z][:] = z

    nube = PointCloud()
    nube.add_chunk(chunk)
    nube.update_bounds()
    return nube


def _leer_referencia_isprs(ruta: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lee el archivo de referencia real: X Y codigo (0=terreno, 1=objeto)."""
    xs, ys, es_terreno = [], [], []
    with open(ruta) as f:
        for linea in f:
            valores = linea.split()
            if len(valores) < 4:
                continue
            xs.append(float(valores[0]))
            ys.append(float(valores[1]))
            es_terreno.append(int(valores[3]) == 0)

    return np.array(xs), np.array(ys), np.array(es_terreno, dtype=bool)


def correr_isprs(
    ruta_nube: str,
    ruta_referencia: str,
    cell_size: float,
    initial_distance: float,
    max_distance: float,
    slope: float,
) -> None:
    ruta_n = Path(ruta_nube)
    ruta_r = Path(ruta_referencia)
    print(f"\n=== {ruta_n.name} vs {ruta_r.name} (PMF) ===")

    nube = _leer_nube_isprs(ruta_n)
    print(f"Puntos leidos (nube completa): {nube.point_count:,}")

    x_ref, y_ref, es_terreno_real = _leer_referencia_isprs(ruta_r)
    print(f"Puntos de referencia (verdad de campo real): {len(x_ref):,}")

    clasificador = PMFGroundClassifier(
        cell_size=cell_size,
        initial_distance=initial_distance,
        max_distance=max_distance,
        slope=slope,
    )
    es_terreno_predicho_completo = clasificador.classify(nube)

    # Emparejar cada punto de referencia con el punto mas cercano de
    # la nube clasificada (por XY) -- confirmado necesario: el
    # archivo de referencia no necesariamente preserva el orden
    # exacto de la nube de entrada. Solo hay 1 chunk (ver
    # _leer_nube_isprs), asi que se toma directamente.
    primer_chunk = next(iter(nube))
    x_nube = np.asarray(primer_chunk[PointAttribute.X])
    y_nube = np.asarray(primer_chunk[PointAttribute.Y])

    from scipy.spatial import cKDTree

    arbol = cKDTree(np.column_stack([x_nube, y_nube]))
    _, indices = arbol.query(np.column_stack([x_ref, y_ref]))
    es_terreno_predicho = es_terreno_predicho_completo[indices]

    terreno_real_total = int(es_terreno_real.sum())
    objeto_real_total = int((~es_terreno_real).sum())

    terreno_mal_clasificado = int(np.sum(es_terreno_real & ~es_terreno_predicho))
    objeto_mal_clasificado = int(np.sum(~es_terreno_real & es_terreno_predicho))
    total_mal_clasificado = terreno_mal_clasificado + objeto_mal_clasificado

    type_i = terreno_mal_clasificado / terreno_real_total * 100 if terreno_real_total else float("nan")
    type_ii = objeto_mal_clasificado / objeto_real_total * 100 if objeto_real_total else float("nan")
    total_error = total_mal_clasificado / len(x_ref) * 100

    # Indice Kappa de Cohen -- confirmado con un caso conocido antes
    # de entregarse: permite comparar directamente contra los
    # valores YA PUBLICADOS en la literatura de seguimiento del
    # benchmark (ej. Meng et al. 2009), que reportan Kappa por sitio,
    # no Type I/II.
    tp = terreno_real_total - terreno_mal_clasificado
    fn = terreno_mal_clasificado
    fp = objeto_mal_clasificado
    tn = objeto_real_total - objeto_mal_clasificado
    total = terreno_real_total + objeto_real_total

    p_o = (tp + tn) / total
    predicho_terreno = tp + fp
    predicho_objeto = fn + tn
    p_e = (terreno_real_total * predicho_terreno + objeto_real_total * predicho_objeto) / (total**2)
    kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else float("nan")

    print(
        f"\n=== Resultado PMF (cell_size={cell_size}, initial_distance={initial_distance}, "
        f"max_distance={max_distance}, slope={slope}) ==="
    )
    print(f"Type I error  (terreno real perdido):    {type_i:.2f}%  ({terreno_mal_clasificado}/{terreno_real_total})")
    print(f"Type II error (falso terreno):              {type_ii:.2f}%  ({objeto_mal_clasificado}/{objeto_real_total})")
    print(f"Total error:                                    {total_error:.2f}%")
    print(f"Kappa (Cohen, comparable con literatura):          {kappa * 100:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nube", help="Ruta al archivo de la nube (ej. FSite7.txt)")
    parser.add_argument("referencia", help="Ruta al archivo de referencia correspondiente")
    parser.add_argument("--cell-size", type=float, default=1.0)
    parser.add_argument("--initial-distance", type=float, default=0.15)
    parser.add_argument("--max-distance", type=float, default=2.5)
    parser.add_argument("--slope", type=float, default=1.0, help="Crecimiento del umbral por unidad horizontal.")
    args = parser.parse_args()

    correr_isprs(
        args.nube,
        args.referencia,
        args.cell_size,
        args.initial_distance,
        args.max_distance,
        args.slope,
    )
