"""
Barrido sistematico de parametros de PMF contra un sitio real del
benchmark ISPRS -- misma logica que isprs_barrido.py (CSF), aplicada
a PMF.

Uso:
    python isprs_barrido_pmf.py nube.txt referencia.txt
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.ground.pmf import PMFGroundClassifier

REJILLA_INITIAL_DISTANCE = (0.1, 0.15, 0.3)
REJILLA_MAX_DISTANCE = (1.5, 2.5, 4.0)
REJILLA_SLOPE = (0.5, 1.0, 2.0)
REJILLA_MAX_WINDOW_SIZE = (17, 33, 65)


def _leer_nube_isprs(ruta: Path) -> PointCloud:
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


def _calcular_kappa(es_terreno_real: np.ndarray, es_terreno_predicho: np.ndarray) -> tuple[float, float, float]:
    terreno_real_total = int(es_terreno_real.sum())
    objeto_real_total = int((~es_terreno_real).sum())

    terreno_mal_clasificado = int(np.sum(es_terreno_real & ~es_terreno_predicho))
    objeto_mal_clasificado = int(np.sum(~es_terreno_real & es_terreno_predicho))

    type_i = terreno_mal_clasificado / terreno_real_total * 100 if terreno_real_total else float("nan")
    type_ii = objeto_mal_clasificado / objeto_real_total * 100 if objeto_real_total else float("nan")

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

    return kappa * 100, type_i, type_ii


def barrer(ruta_nube: str, ruta_referencia: str, cell_size: float) -> None:
    ruta_n = Path(ruta_nube)
    ruta_r = Path(ruta_referencia)
    print(f"\n=== Barrido PMF: {ruta_n.name} vs {ruta_r.name} ===")

    nube = _leer_nube_isprs(ruta_n)
    x_ref, y_ref, es_terreno_real = _leer_referencia_isprs(ruta_r)
    print(f"Puntos: {nube.point_count:,}  Referencia: {len(x_ref):,}")

    primer_chunk = next(iter(nube))
    x_nube = np.asarray(primer_chunk[PointAttribute.X])
    y_nube = np.asarray(primer_chunk[PointAttribute.Y])

    from scipy.spatial import cKDTree

    arbol = cKDTree(np.column_stack([x_nube, y_nube]))
    _, indices = arbol.query(np.column_stack([x_ref, y_ref]))

    combinaciones = list(
        itertools.product(REJILLA_INITIAL_DISTANCE, REJILLA_MAX_DISTANCE, REJILLA_SLOPE, REJILLA_MAX_WINDOW_SIZE)
    )
    # Descartar combinaciones invalidas (initial_distance > max_distance)
    combinaciones = [c for c in combinaciones if c[0] <= c[1]]
    print(f"Probando {len(combinaciones)} combinaciones reales...\n")

    resultados = []
    for i, (initial_distance, max_distance, slope, max_window_size) in enumerate(combinaciones, start=1):
        clasificador = PMFGroundClassifier(
            cell_size=cell_size,
            initial_distance=initial_distance,
            max_distance=max_distance,
            slope=slope,
            max_window_size=max_window_size,
        )
        es_terreno_predicho_completo = clasificador.classify(nube)
        es_terreno_predicho = es_terreno_predicho_completo[indices]

        kappa, type_i, type_ii = _calcular_kappa(es_terreno_real, es_terreno_predicho)
        resultados.append((kappa, initial_distance, max_distance, slope, max_window_size, type_i, type_ii))
        print(
            f"[{i}/{len(combinaciones)}] initial_distance={initial_distance} max_distance={max_distance} "
            f"slope={slope} max_window_size={max_window_size}  ->  Kappa={kappa:.2f}  "
            f"(Type I={type_i:.2f}%, Type II={type_ii:.2f}%)"
        )

    resultados.sort(key=lambda r: r[0], reverse=True)
    mejor = resultados[0]

    print("\n=== Mejor combinacion encontrada ===")
    print(
        f"initial_distance={mejor[1]}  max_distance={mejor[2]}  slope={mejor[3]}  max_window_size={mejor[4]}"
    )
    print(f"Kappa={mejor[0]:.2f}  (Type I={mejor[5]:.2f}%, Type II={mejor[6]:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nube", help="Ruta al archivo de la nube")
    parser.add_argument("referencia", help="Ruta al archivo de referencia")
    parser.add_argument("--cell-size", type=float, default=0.5)
    args = parser.parse_args()

    barrer(args.nube, args.referencia, args.cell_size)
