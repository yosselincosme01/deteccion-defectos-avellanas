"""Experimentos de rendimiento E1 y E2 del método clásico.

  E1  Secuencial (1 proceso), a 256 y 512 px          -> línea base T(1)
  E2  Paralelo con 2, 4 y 8 procesos, a 256 y 512 px  -> T(p), speedup, eficiencia

Diseño de la medición:
  * Cada configuración se repite N veces (por defecto 3) para medir la
    variación entre ejecuciones (media ± desviación).
  * Las configuraciones se INTERCALAN dentro de cada repetición
    (sec, p2, p4, p8, sec, p2, ...): si la PC se calienta o algo corre en
    segundo plano, afecta a todas por igual y no solo a una.
  * Cada ejecución paralela se compara con la secuencial de la misma
    repetición para comprobar que los resultados son idénticos.
  * Tiempos con time.perf_counter() (reloj de alta resolución), en segundos.

Resultados:
  resultados/rendimiento.csv   una fila por ejecución
  resultados/entorno.txt       hardware y versiones de software
  resultados/mapas_clasico_256.npz, puntajes_clasico_256.csv  (para figuras)

Uso (desde la raíz del proyecto, con la PC preparada):
    python -m experimentos.rendimiento                 # completo (~35 min)
    python -m experimentos.rendimiento --rapido        # prueba (~2 min)
"""
import src.comun.configuracion  # noqa: F401  (1 hilo por proceso; va primero)

import argparse
import csv
import os
import platform
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from src.comun.modelo_clasico import ejecutar, resumen
from src.paralelo.clasico_paralelo import extraer_paralelo, verificar_igualdad
from src.secuencial.clasico_secuencial import extraer_secuencial

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
CAMPOS = ["experimento", "repeticion", "tam", "procesos", "chunksize",
          "t_crear_pool", "t_calculo", "t_cerrar_pool", "t_extraccion",
          "t_gmm", "t_puntuacion", "t_total", "auroc_imagen",
          "atributos_identicos", "puntajes_identicos", "fecha"]


def guardar_entorno() -> None:
    """Hardware y software usados (la guía pide reportarlos)."""
    import cv2, sklearn, skimage  # noqa: E401
    lineas = {
        "sistema": platform.platform(),
        "procesador": platform.processor(),
        "nucleos_logicos": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__, "opencv": cv2.__version__,
        "scikit-learn": sklearn.__version__,
        "scikit-image": skimage.__version__,
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    with (RESULTADOS / "entorno.txt").open("w", encoding="utf-8") as f:
        f.writelines(f"{k}: {v}\n" for k, v in lineas.items())


def guardar_fila(fila: dict, archivo: Path) -> None:
    """Agrega una fila al CSV apenas termina cada ejecución
    (si algo falla a la mitad, no se pierde lo ya medido)."""
    nuevo = not archivo.exists()
    with archivo.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def guardar_para_figuras(r: dict) -> None:
    """Mapas y puntajes de una corrida secuencial a 256 px (para Emmy y figuras)."""
    np.savez_compressed(
        RESULTADOS / "mapas_clasico_256.npz",
        mapas=np.stack(r["mapas"]).astype(np.float32),
        rutas=np.array([Path(m.ruta).parent.name + "/" + Path(m.ruta).name
                        for m in r["muestras"]]))
    with (RESULTADOS / "puntajes_clasico_256.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["imagen", "defecto", "etiqueta", "puntaje"])
        for m, p in zip(r["muestras"], r["puntajes"]):
            w.writerow([Path(m.ruta).parent.name + "/" + Path(m.ruta).name,
                        m.defecto, m.etiqueta, f"{p:.6f}"])


def main():
    ap = argparse.ArgumentParser(description="Experimentos E1 y E2")
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--rapido", action="store_true",
                    help="prueba corta: 1 repetición, 256 px, 1 y 2 procesos")
    args = ap.parse_args()

    tams, procesos, reps = (256, 512), (2, 4, 8), args.repeticiones
    archivo = RESULTADOS / "rendimiento.csv"
    if args.rapido:
        tams, procesos, reps = (256,), (2,), 1
        archivo = RESULTADOS / "rendimiento_prueba.csv"
    RESULTADOS.mkdir(exist_ok=True)
    if archivo.exists():
        archivo.unlink()          # empezamos un archivo limpio
    guardar_entorno()

    total = len(tams) * reps * (1 + len(procesos))
    n = 0
    for tam in tams:
        for rep in range(reps):
            n += 1
            base = ejecutar(extraer_secuencial, tam=tam)
            print(f"[{n}/{total}] E1 rep={rep}", resumen(base), flush=True)
            guardar_fila({"experimento": "E1", "repeticion": rep,
                          "fecha": datetime.now().isoformat(timespec="seconds"),
                          **base}, archivo)
            if tam == 256 and rep == 0:
                guardar_para_figuras(base)

            for p in procesos:
                n += 1
                r = ejecutar(partial(extraer_paralelo, procesos=p), tam=tam)
                v = verificar_igualdad(base, r)
                print(f"[{n}/{total}] E2 rep={rep}", resumen(r),
                      "| idénticos" if all(v.values()) else f"| ¡DIFERENTES! {v}",
                      flush=True)
                guardar_fila({"experimento": "E2", "repeticion": rep,
                              "fecha": datetime.now().isoformat(timespec="seconds"),
                              **r, **v}, archivo)
    print(f"Listo. Resultados en {archivo}")


if __name__ == "__main__":
    main()
