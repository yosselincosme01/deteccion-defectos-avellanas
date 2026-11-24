"""Método clásico - VERSIÓN SECUENCIAL (línea base).

Un solo proceso y un solo hilo: las imágenes se procesan una detrás de otra.
Sirve para:
  1. comprobar que el algoritmo funciona y da buenos resultados;
  2. medir T(1), el tiempo con el que se calcula el speedup del paralelo.

Uso (desde la raíz del proyecto):
    python -m src.secuencial.clasico_secuencial
    python -m src.secuencial.clasico_secuencial --tam 512
    python -m src.secuencial.clasico_secuencial --fraccion 0.25
"""
import src.comun.configuracion  # noqa: F401  (1 hilo; debe ir primero)

import argparse
import time

from src.comun.caracteristicas import caracteristicas_imagen
from src.comun.modelo_clasico import ejecutar, resumen


def extraer_secuencial(rutas: list[str], tam: int):
    """Procesa las imágenes una por una, en orden."""
    t0 = time.perf_counter()
    atributos = [caracteristicas_imagen(r, tam) for r in rutas]
    return atributos, {"procesos": 1, "t_calculo": time.perf_counter() - t0}


def main():
    ap = argparse.ArgumentParser(description="Método clásico, versión secuencial")
    ap.add_argument("--tam", type=int, default=256,
                    help="lado de la imagen en píxeles (256 o 512)")
    ap.add_argument("--fraccion", type=float, default=1.0,
                    help="fracción de imágenes sanas para aprender (0 a 1)")
    args = ap.parse_args()

    r = ejecutar(extraer_secuencial, tam=args.tam, fraccion=args.fraccion)
    print("[secuencial]", resumen(r))

    # Las 5 avellanas más sospechosas, para ver que tenga sentido
    orden = sorted(range(len(r["puntajes"])), key=lambda i: -r["puntajes"][i])
    print("Más sospechosas:")
    for i in orden[:5]:
        m = r["muestras"][i]
        print(f"  {m.defecto:6s} {m.ruta[-7:]}  puntaje={r['puntajes'][i]:.1f}")


if __name__ == "__main__":
    main()
