"""Configuración que debe cargarse ANTES que numpy / scikit-learn / OpenCV.

1) UN HILO POR PROCESO. NumPy y scikit-learn pueden usar varios núcleos por
   su cuenta sin que lo pidamos: la "versión secuencial" ya sería paralela a
   escondidas y el speedup medido sería engañoso. Con 1 hilo por proceso,
   T(1) es de verdad un solo núcleo y en la versión paralela cada proceso
   ocupa un núcleo.

2) SIN MODO AHORRO DE ENERGÍA (Windows, procesadores híbridos). Los Intel de
   13.ª generación tienen núcleos rápidos (P) y núcleos de ahorro (E).
   Windows puede marcar un proceso como "de fondo" (EcoQoS) y mandarlo a los
   núcleos E, que son mucho más lentos: el mismo programa tardaba entre 59 s
   y 139 s según dónde cayera. Desactivamos ese modo para nuestro proceso y
   para cada trabajador, de modo que las mediciones sean comparables.

Uso: debe ser la PRIMERA importación del script principal:
    import src.comun.configuracion  # noqa: F401
"""
import os
import sys

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

SEMILLA = 0   # misma semilla en todo el proyecto -> resultados reproducibles


def preparar_proceso() -> None:
    """Se ejecuta al arrancar cada proceso trabajador del Pool
    (initializer), para que también quede sin modo ahorro de energía."""
    _desactivar_ahorro_energia_windows()


def _desactivar_ahorro_energia_windows() -> None:
    """Pide a Windows que no aplique EcoQoS (estrangulamiento) a este proceso."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class _ESTADO(ctypes.Structure):   # PROCESS_POWER_THROTTLING_STATE
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        estado = _ESTADO(1, 0x1, 0x0)  # controlar EXECUTION_SPEED -> apagado
        k32 = ctypes.windll.kernel32
        k32.SetProcessInformation(k32.GetCurrentProcess(), 4,  # ProcessPowerThrottling
                                  ctypes.byref(estado), ctypes.sizeof(estado))
        # Prioridad "por encima de lo normal" para que no compita con
        # programas en segundo plano durante la medición.
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000)
    except Exception:
        pass   # si falla, el programa sigue funcionando igual


_desactivar_ahorro_energia_windows()
