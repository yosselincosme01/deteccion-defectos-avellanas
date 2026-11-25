"""Experimentos E6 y E7 (equipo B): memoria, IoU y CPU frente a GPU.

  E6  Memoria pico e IoU de la localización, a 256 y 512 px:
        clásico secuencial, clásico con 8 procesos y profundo en CPU.
  E7  Método profundo en CPU frente a GPU (CUDA), a 256 y 512 px,
      separando el tiempo de transferencia del de cómputo.

IMPORTANTE: estos resultados se miden en OTRA computadora (equipo B) y se
guardan en archivos PROPIOS. Este script nunca toca rendimiento.csv,
calidad.csv ni entorno.txt (equipo A). Los tiempos de E6 son solo
informativos: el speedup oficial es el de E1/E2.

Diseño de la medición:
  * Cada configuración corre en un PROCESO NUEVO de Python. Así la memoria
    medida empieza limpia (por ejemplo, el clásico no carga torch).
  * Memoria pico: un hilo mide cada 20 ms la memoria residente (RSS) del
    proceso principal MÁS la de sus procesos hijos (los trabajadores del
    Pool), y guarda el máximo de la suma. En GPU se reporta además la
    memoria pico reservada por PyTorch.
  * IoU: src/comun/metricas.py::iou_pixel (umbral que maximiza F1 por píxel).
  * E7: 1 ejecución de calentamiento en GPU (no se guarda) + 3 repeticiones.

Resultados:
  resultados/e6_memoria_iou.csv
  resultados/e7_cpu_gpu.csv
  resultados/entorno_equipo_b.txt

Uso (desde la raíz del proyecto, con la PC libre y conectada a la corriente):
    python -m experimentos.memoria_iou              # E6 y E7 (~20 min)
    python -m experimentos.memoria_iou --solo e6
    python -m experimentos.memoria_iou --solo e7
    python -m experimentos.memoria_iou --rapido     # prueba corta, solo 256 px
"""
import src.comun.configuracion  # noqa: F401  (1 hilo por proceso; va primero)

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import psutil

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"
MARCA = "RESULTADO_JSON:"

CAMPOS_E6 = ["experimento", "metodo", "tam", "procesos", "dispositivo",
             "hilos", "mem_base_mb", "mem_pico_mb", "mem_gpu_pico_mb",
             "t_total", "auroc_imagen", "auroc_pixel", "f1",
             "iou_global", "iou_medio", "umbral_pixel", "fecha"]
CAMPOS_E7 = ["experimento", "repeticion", "tam", "dispositivo", "hilos",
             "t_banco", "t_red_prueba", "t_knn", "t_total", "t_hacia_disp",
             "t_desde_disp", "mem_pico_mb", "mem_gpu_pico_mb",
             "auroc_imagen", "auroc_pixel", "fecha"]


# --------------------------------------------------------------------------- #
# Medición de memoria
# --------------------------------------------------------------------------- #
class MonitorMemoria:
    """Mide en segundo plano la memoria RSS del proceso + sus hijos."""

    def __init__(self, intervalo: float = 0.02):
        self.proc = psutil.Process(os.getpid())
        self.intervalo = intervalo
        self.pico = 0
        self._parar = threading.Event()
        self._hilo = threading.Thread(target=self._medir, daemon=True)

    def _total(self) -> int:
        total = self.proc.memory_info().rss
        for hijo in self.proc.children(recursive=True):
            try:
                total += hijo.memory_info().rss
            except psutil.Error:        # el hijo terminó mientras medíamos
                pass
        return total

    def _medir(self) -> None:
        while not self._parar.is_set():
            self.pico = max(self.pico, self._total())
            time.sleep(self.intervalo)

    def __enter__(self):
        self.base = self._total()
        self._hilo.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        self._hilo.join()
        self.pico = max(self.pico, self._total())


# --------------------------------------------------------------------------- #
# Una configuración (se ejecuta en un proceso hijo)
# --------------------------------------------------------------------------- #
def correr_una(metodo: str, tam: int, procesos: int, dispositivo: str) -> dict:
    from src.comun.metricas import iou_pixel, resumen_calidad

    with MonitorMemoria() as mon:
        if metodo == "clasico":
            from src.comun.modelo_clasico import ejecutar
            if procesos == 1:
                from src.secuencial.clasico_secuencial import extraer_secuencial
                r = ejecutar(extraer_secuencial, tam=tam)
            else:
                from src.paralelo.clasico_paralelo import extraer_paralelo
                r = ejecutar(partial(extraer_paralelo, procesos=procesos), tam=tam)
            r.update(resumen_calidad(r["muestras"], r["puntajes"], r["mapas"]))
            r["dispositivo"], r["hilos"] = "cpu", 1
        else:
            from src.profundo.resnet_knn import ejecutar
            r = ejecutar(tam=tam, dispositivo=dispositivo)

    r.update(iou_pixel(r["muestras"], r["mapas"]))
    r.update({"metodo": metodo, "procesos": procesos,
              "mem_base_mb": mon.base / 2**20, "mem_pico_mb": mon.pico / 2**20})
    r.setdefault("mem_gpu_pico_mb", 0.0)
    # Solo valores simples (sin mapas ni atributos) para mandar al padre
    return {k: v for k, v in r.items()
            if isinstance(v, (int, float, str, bool)) or v is None}


def lanzar(metodo: str, tam: int, procesos: int = 1,
           dispositivo: str = "cpu") -> dict:
    """Corre una configuración en un proceso NUEVO y devuelve su resultado."""
    cmd = [sys.executable, "-m", "experimentos.memoria_iou", "--uno",
           metodo, str(tam), str(procesos), dispositivo]
    salida = subprocess.run(cmd, cwd=RAIZ, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    for linea in salida.stdout.splitlines():
        if linea.startswith(MARCA):
            return json.loads(linea[len(MARCA):])
    raise RuntimeError(f"Falló {cmd}:\n{salida.stdout}\n{salida.stderr}")


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #
def guardar_entorno() -> None:
    import cv2, numpy, sklearn, skimage, torch, torchvision  # noqa: E401
    gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
           else "no disponible")
    try:
        cpu = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:
        cpu = platform.processor()
    lineas = {
        "equipo": "B",
        "sistema": platform.platform(),
        "procesador": cpu,
        "nucleos_fisicos": psutil.cpu_count(logical=False),
        "nucleos_logicos": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "gpu": gpu,
        "cuda_torch": torch.version.cuda,
        "python": platform.python_version(),
        "numpy": numpy.__version__, "opencv": cv2.__version__,
        "scikit-learn": sklearn.__version__, "scikit-image": skimage.__version__,
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "psutil": psutil.__version__,
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    with (RESULTADOS / "entorno_equipo_b.txt").open("w", encoding="utf-8") as f:
        f.writelines(f"{k}: {v}\n" for k, v in lineas.items())


def guardar_fila(fila: dict, archivo: Path, campos: list[str]) -> None:
    nuevo = not archivo.exists()
    with archivo.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def ahora() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Experimentos
# --------------------------------------------------------------------------- #
def e6(tams, archivo: Path) -> None:
    configs = [("clasico", 1, "cpu"), ("clasico", 8, "cpu"),
               ("profundo", 1, "cpu")]
    for tam in tams:
        for metodo, p, disp in configs:
            r = lanzar(metodo, tam, p, disp)
            guardar_fila({"experimento": "E6", "fecha": ahora(), **r},
                         archivo, CAMPOS_E6)
            print(f"E6 {metodo:8s} tam={tam} p={p} | mem pico="
                  f"{r['mem_pico_mb']:.0f} MB | IoU global={r['iou_global']:.3f} "
                  f"medio={r['iou_medio']:.3f} | AUROC img={r['auroc_imagen']:.3f} "
                  f"| t={r['t_total']:.1f} s", flush=True)


def e7(tams, reps: int, archivo: Path) -> None:
    import torch
    disps = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if len(disps) == 1:
        print("E7: no hay GPU CUDA disponible; solo se mide CPU.")
    for tam in tams:
        for disp in disps:
            if disp == "cuda":   # calentamiento: inicializar CUDA/cuDNN
                lanzar("profundo", tam, 1, disp)
            for rep in range(reps):
                r = lanzar("profundo", tam, 1, disp)
                guardar_fila({"experimento": "E7", "repeticion": rep,
                              "fecha": ahora(), **r}, archivo, CAMPOS_E7)
                print(f"E7 {disp:4s} tam={tam} rep={rep} | total="
                      f"{r['t_total']:.2f} s (red={r['t_red_prueba']:.2f}, "
                      f"vecino={r['t_knn']:.2f}, ->disp={r['t_hacia_disp']:.3f}, "
                      f"disp->={r['t_desde_disp']:.3f}) | GPU mem="
                      f"{r['mem_gpu_pico_mb']:.0f} MB | AUROC={r['auroc_imagen']:.3f}",
                      flush=True)


def main():
    ap = argparse.ArgumentParser(description="Experimentos E6 y E7 (equipo B)")
    ap.add_argument("--uno", nargs=4, metavar=("METODO", "TAM", "PROCESOS", "DISP"),
                    help=argparse.SUPPRESS)     # uso interno (proceso hijo)
    ap.add_argument("--solo", choices=["e6", "e7"])
    ap.add_argument("--repeticiones", type=int, default=3)
    ap.add_argument("--rapido", action="store_true",
                    help="prueba corta: solo 256 px y 1 repetición")
    args = ap.parse_args()

    if args.uno:                                  # proceso hijo
        metodo, tam, p, disp = args.uno
        r = correr_una(metodo, int(tam), int(p), disp)
        print(MARCA + json.dumps(r), flush=True)
        return

    tams, reps = ((256,), 1) if args.rapido else ((256, 512), args.repeticiones)
    sufijo = "_prueba" if args.rapido else ""
    RESULTADOS.mkdir(exist_ok=True)
    guardar_entorno()

    if args.solo in (None, "e6"):
        archivo = RESULTADOS / f"e6_memoria_iou{sufijo}.csv"
        archivo.unlink(missing_ok=True)           # solo borra SU propio archivo
        e6(tams, archivo)
    if args.solo in (None, "e7"):
        archivo = RESULTADOS / f"e7_cpu_gpu{sufijo}.csv"
        archivo.unlink(missing_ok=True)
        e7(tams, reps, archivo)
    print("Listo.")


# Obligatorio en Windows con multiprocessing (ver clasico_paralelo.py)
if __name__ == "__main__":
    main()
