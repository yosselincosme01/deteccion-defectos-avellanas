"""Lectura del dataset MVTec AD, categoría hazelnut (avellanas).

Estructura en disco:
    datos/hazelnut/
        train/good/                   -> 391 avellanas sanas (para aprender)
        test/good/                    -> 40 sanas (para evaluar)
        test/crack|cut|hole|print/    -> 70 con defecto (para evaluar)
        ground_truth/<defecto>/000_mask.png -> dónde está cada defecto
"""
from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np

# Carpeta del dataset, calculada desde la ubicación de este archivo:
# src/comun/datos.py -> sube 3 niveles -> raíz del proyecto -> datos/hazelnut
RUTA_DATOS = Path(__file__).resolve().parents[2] / "datos" / "hazelnut"


@dataclass(frozen=True)
class Muestra:
    """Una imagen de prueba junto con su respuesta correcta."""
    ruta: str                 # ubicación de la imagen
    defecto: str              # "good", "crack", "cut", "hole" o "print"
    etiqueta: int             # 0 = sana, 1 = con defecto
    ruta_mascara: str | None  # máscara del defecto (None si está sana)


def _pngs(carpeta: Path) -> list[str]:
    """Lista ordenada de las imágenes .png de una carpeta."""
    return sorted(str(p) for p in carpeta.glob("*.png"))


def imagenes_entrenamiento(fraccion: float = 1.0, semilla: int = 0) -> list[str]:
    """Rutas de las avellanas sanas para aprender.

    fraccion < 1 toma solo una parte, al azar pero siempre la misma
    (gracias a la semilla). Se usa en el experimento E4: pocos datos.
    """
    rutas = _pngs(RUTA_DATOS / "train" / "good")
    if fraccion < 1.0:
        n = max(1, round(len(rutas) * fraccion))
        rutas = sorted(random.Random(semilla).sample(rutas, n))
    return rutas


def muestras_prueba() -> list[Muestra]:
    """Todas las imágenes de prueba (sanas y defectuosas) con su etiqueta."""
    muestras = []
    for carpeta in sorted((RUTA_DATOS / "test").iterdir()):
        defecto = carpeta.name
        for ruta in _pngs(carpeta):
            mascara = None
            if defecto != "good":
                nombre = Path(ruta).stem + "_mask.png"   # 000.png -> 000_mask.png
                mascara = str(RUTA_DATOS / "ground_truth" / defecto / nombre)
            muestras.append(Muestra(ruta, defecto, int(defecto != "good"), mascara))
    return muestras


def cargar_imagen(ruta: str, tam: int) -> np.ndarray:
    """Lee una imagen en color (RGB) y la reduce a tam x tam píxeles."""
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)        # OpenCV lee en orden BGR
    if img is None:
        raise IOError(f"No se pudo leer {ruta}")
    img = cv2.resize(img, (tam, tam), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # lo pasamos a RGB


def cargar_mascara(ruta: str | None, tam: int) -> np.ndarray:
    """Máscara del defecto: True donde hay defecto, False donde no."""
    if ruta is None:                                 # avellana sana: todo False
        return np.zeros((tam, tam), dtype=bool)
    m = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (tam, tam), interpolation=cv2.INTER_NEAREST)
    return m > 127                                   # blanco = defecto


if __name__ == "__main__":
    # Prueba rápida: python -m src.comun.datos
    ent = imagenes_entrenamiento()
    prueba = muestras_prueba()
    print(f"Entrenamiento: {len(ent)} sanas")
    print(f"Prueba: {len(prueba)} imágenes "
          f"({sum(m.etiqueta == 0 for m in prueba)} sanas, "
          f"{sum(m.etiqueta == 1 for m in prueba)} con defecto)")
    img = cargar_imagen(prueba[0].ruta, 256)
    print(f"Una imagen cargada: forma {img.shape}, tipo {img.dtype}")
    print(f"Con 10 % de datos: {len(imagenes_entrenamiento(0.10))} sanas")
