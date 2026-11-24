"""Experimentos de calidad E3, E4 y E5: método clásico frente a profundo.

  E3  ResNet18 en CPU a 256 px, 3 repeticiones   -> calidad y costo del profundo
  E4  Ambos con 10, 25, 50 y 100 % de las sanas  -> ¿cuál resiste mejor con pocos datos?
  E5  Ambos a 512 px (y 256 px de E4)            -> efecto de la resolución

Notas de diseño:
  * El método clásico se corre en su versión paralela con 8 procesos: da
    resultados idénticos a la secuencial (verificado en E2) y es más rápido.
  * Las métricas de calidad son deterministas (misma semilla): repetirlas no
    cambia el resultado, por eso E4 y E5 se corren una sola vez.
  * Los tiempos se miden en la misma PC que E1 y E2, para que el costo de
    ambos métodos sea comparable.

Resultados:
  resultados/calidad.csv           una fila por ejecución
  resultados/mapas_profundo_256.npz  mapas de defectos (para la figura)

Uso (desde la raíz del proyecto, con la PC libre):
    python -m experimentos.calidad              # completo (~10 min)
    python -m experimentos.calidad --rapido     # prueba (~1 min)
"""
import src.comun.configuracion  # noqa: F401  (va primero)

import argparse
import csv
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from src.comun.metricas import resumen_calidad
from src.comun.modelo_clasico import ejecutar as ejecutar_clasico
from src.paralelo.clasico_paralelo import extraer_paralelo
from src.profundo.resnet_knn import ejecutar as ejecutar_profundo

RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"
CAMPOS = ["experimento", "metodo", "repeticion", "tam", "fraccion",
          "n_entrenamiento", "dispositivo", "hilos", "procesos", "t_total",
          "auroc_imagen", "auroc_pixel", "precision", "recall", "f1",
          "falsos_positivos", "falsos_negativos", "auroc_crack", "auroc_cut",
          "auroc_hole", "auroc_print", "fecha"]


def guardar_fila(fila: dict, archivo: Path) -> None:
    nuevo = not archivo.exists()
    with archivo.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, extrasaction="ignore")
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def clasico(tam: int, fraccion: float) -> dict:
    """Método clásico (paralelo, 8 procesos) con métricas completas."""
    r = ejecutar_clasico(partial(extraer_paralelo, procesos=8),
                         tam=tam, fraccion=fraccion)
    r.update(resumen_calidad(r["muestras"], r["puntajes"], r["mapas"]))
    r["metodo"] = "clasico"
    return r


def profundo(tam: int, fraccion: float) -> dict:
    return ejecutar_profundo(tam=tam, fraccion=fraccion, dispositivo="cpu")


def registrar(r: dict, experimento: str, rep: int, archivo: Path) -> None:
    guardar_fila({"experimento": experimento, "repeticion": rep,
                  "fecha": datetime.now().isoformat(timespec="seconds"), **r},
                 archivo)
    print(f"{experimento} {r['metodo']:8s} tam={r['tam']} "
          f"datos={r['fraccion']:.0%} ({r['n_entrenamiento']} sanas) | "
          f"AUROC img={r['auroc_imagen']:.3f} píxel={r['auroc_pixel']:.3f} "
          f"F1={r['f1']:.3f} | t={r['t_total']:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Experimentos E3, E4 y E5")
    ap.add_argument("--rapido", action="store_true",
                    help="prueba corta: E3 1 vez y E4 solo con 10 %%")
    args = ap.parse_args()

    archivo = RESULTADOS / ("calidad_prueba.csv" if args.rapido else "calidad.csv")
    RESULTADOS.mkdir(exist_ok=True)
    if archivo.exists():
        archivo.unlink()

    # E3: costo y calidad del profundo en CPU
    for rep in range(1 if args.rapido else 3):
        r = profundo(256, 1.0)
        registrar(r, "E3", rep, archivo)
        if rep == 0:
            np.savez_compressed(RESULTADOS / "mapas_profundo_256.npz",
                                mapas=np.stack(r["mapas"]).astype(np.float32))

    # E4: pocos datos
    for fr in ((0.10,) if args.rapido else (0.10, 0.25, 0.50, 1.00)):
        registrar(clasico(256, fr), "E4", 0, archivo)
        registrar(profundo(256, fr), "E4", 0, archivo)

    # E5: resolución (los valores a 256 px salen de E4 con 100 %)
    if not args.rapido:
        registrar(clasico(512, 1.0), "E5", 0, archivo)
        registrar(profundo(512, 1.0), "E5", 0, archivo)

    print(f"Listo. Resultados en {archivo}")


if __name__ == "__main__":
    main()
