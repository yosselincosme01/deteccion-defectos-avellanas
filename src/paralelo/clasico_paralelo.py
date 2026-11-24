"""Método clásico - VERSIÓN PARALELA (multiprocessing).

QUÉ se paraleliza: la extracción de atributos (color + GLCM), que ocupa ~85 %
del tiempo. Cada imagen es independiente de las demás -> paralelismo de datos.

CÓMO se reparte:
  * Se crea un Pool con p procesos trabajadores. En Windows cada uno arranca
    su propio intérprete de Python (eso es sobrecarga: tiempo de arranque).
  * pool.map envía a los trabajadores lotes ("chunks") de RUTAS de imagen,
    que son textos cortos; cada trabajador lee su imagen del disco.
    Así se comunica muy poco.
  * Cada trabajador devuelve la matriz de atributos de su imagen. pool.map
    entrega los resultados EN EL MISMO ORDEN que las rutas, así que el
    resultado es idéntico al secuencial.
  * Sincronización: pool.map espera a que todos terminen (barrera implícita).
    No hay datos compartidos que se modifiquen, así que no hacen falta
    candados (locks).

QUÉ queda secuencial: el ajuste del GMM y la puntuación (ley de Amdahl).

POR QUÉ procesos y no hilos: en Python el GIL deja ejecutar código Python a un
solo hilo a la vez, así que los hilos no aceleran cálculos pesados.

Uso (desde la raíz del proyecto):
    python -m src.paralelo.clasico_paralelo --procesos 4
    python -m src.paralelo.clasico_paralelo --procesos 4 --verificar
"""
import src.comun.configuracion  # noqa: F401  (1 hilo por proceso; va primero)

import argparse
import time
from functools import partial
from multiprocessing import Pool

import numpy as np

from src.comun.caracteristicas import caracteristicas_imagen
from src.comun.modelo_clasico import ejecutar, resumen


def extraer_paralelo(rutas: list[str], tam: int, procesos: int = 4,
                     chunksize: int | None = None):
    """Reparte las imágenes entre `procesos` trabajadores."""
    if chunksize is None:
        # Lotes medianos: ~4 lotes por trabajador. Pocos mensajes, pero si un
        # trabajador termina antes, todavía quedan lotes para él (balance).
        chunksize = max(1, len(rutas) // (procesos * 4))
    tarea = partial(caracteristicas_imagen, tam=tam)   # fija tam para todas

    t0 = time.perf_counter()
    with Pool(processes=procesos) as pool:
        t1 = time.perf_counter()
        atributos = pool.map(tarea, rutas, chunksize=chunksize)
        t2 = time.perf_counter()
    t3 = time.perf_counter()
    return atributos, {
        "procesos": procesos, "chunksize": chunksize,
        "t_crear_pool": t1 - t0,    # arrancar los procesos
        "t_calculo": t2 - t1,       # repartir + calcular + recoger
        "t_cerrar_pool": t3 - t2,   # terminar los procesos
    }


def verificar_igualdad(r_sec: dict, r_par: dict) -> dict:
    """Comprueba que ambas versiones dan EXACTAMENTE el mismo resultado."""
    atributos_iguales = all(np.array_equal(a, b) for a, b in
                            zip(r_sec["atributos"], r_par["atributos"]))
    puntajes_iguales = r_sec["puntajes"] == r_par["puntajes"]
    return {"atributos_identicos": atributos_iguales,
            "puntajes_identicos": puntajes_iguales,
            "auroc_identico": r_sec["auroc_imagen"] == r_par["auroc_imagen"]}


def main():
    ap = argparse.ArgumentParser(description="Método clásico, versión paralela")
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--fraccion", type=float, default=1.0)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--verificar", action="store_true",
                    help="corre también la secuencial y compara resultados")
    args = ap.parse_args()

    if args.verificar:   # la secuencial va PRIMERO, con la CPU "en frío"
        from src.secuencial.clasico_secuencial import extraer_secuencial
        r_sec = ejecutar(extraer_secuencial, tam=args.tam,
                         fraccion=args.fraccion)
        print("[secuencial]   ", resumen(r_sec))

    extraer = partial(extraer_paralelo, procesos=args.procesos)
    r = ejecutar(extraer, tam=args.tam, fraccion=args.fraccion)
    print(f"[paralelo p={args.procesos}]", resumen(r))
    print(f"  crear procesos={r['t_crear_pool']:.2f} s | "
          f"cálculo={r['t_calculo']:.2f} s | cerrar={r['t_cerrar_pool']:.2f} s")

    if args.verificar:
        print("Verificación:", verificar_igualdad(r_sec, r))
        s_ext = r_sec["t_extraccion"] / r["t_extraccion"]
        s_tot = r_sec["t_total"] / r["t_total"]
        print(f"Speedup extracción = {s_ext:.2f}  (eficiencia {s_ext / args.procesos:.0%})")
        print(f"Speedup total      = {s_tot:.2f}  (eficiencia {s_tot / args.procesos:.0%})")


# En Windows este bloque es OBLIGATORIO con multiprocessing: cada proceso
# nuevo vuelve a importar este archivo, y sin el "if" volvería a crear
# procesos sin fin.
if __name__ == "__main__":
    main()
