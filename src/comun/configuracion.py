"""Configuración que debe cargarse ANTES que numpy / scikit-learn / OpenCV.

Fija las librerías numéricas a 1 hilo por proceso. Sin esto, NumPy y
scikit-learn pueden usar varios núcleos por su cuenta sin que lo pidamos:
la "versión secuencial" ya sería paralela a escondidas y el speedup medido
sería engañoso. Con 1 hilo por proceso, T(1) es de verdad un solo núcleo y
en la versión paralela cada proceso ocupa exactamente un núcleo.

Uso: debe ser la PRIMERA importación del script principal:
    import src.comun.configuracion  # noqa: F401
"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

SEMILLA = 0   # misma semilla en todo el proyecto -> resultados reproducibles
