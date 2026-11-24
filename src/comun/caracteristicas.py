"""Atributos del método clásico: color + textura, por zona.

Cada imagen se corta en zonas (parches) de TAM_ZONA x TAM_ZONA píxeles que se
solapan a la mitad. Cada zona se describe con un vector de 34 números:

* COLOR (Sucar, cap. 4): histograma de cada canal H, S y V con BINS_COLOR
  barras -> 3 x 8 = 24 números. Detecta manchas o zonas con otro tono.

* TEXTURA (Sucar, cap. 5.4.2, "matrices de dependencia espacial" = GLCM):
  para cada distancia d se promedian los 4 ángulos y se calculan energía,
  entropía, correlación, inercia (contraste) y homogeneidad
  -> 2 distancias x 5 atributos = 10 números.
  Detecta grietas, cortes y agujeros, que rompen la textura de la cáscara.

Esta es la parte MÁS COSTOSA del método y la que se paralelizará: cada imagen
se procesa de forma totalmente independiente de las demás.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from src.comun.datos import cargar_imagen

# --- Parámetros -------------------------------------------------------------
TAM_ZONA = 32         # lado de cada zona en píxeles
PASO_ZONA = 16        # cuánto se mueve la ventana (16 = se solapan a la mitad)
NIVELES_GRIS = 16     # la GLCM usa 16 niveles de gris en vez de 256
DISTANCIAS = (1, 2)                          # d: vecino a 1 y a 2 píxeles
ANGULOS = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)  # 0°, 45°, 90°, 135°
BINS_COLOR = 8        # barras del histograma por canal

NOMBRES_TEXTURA = ("energia", "entropia", "correlacion", "inercia",
                   "homogeneidad")
DIM = 3 * BINS_COLOR + len(DISTANCIAS) * len(NOMBRES_TEXTURA)   # 34


def posiciones(tam: int) -> list[int]:
    """Coordenadas donde empieza cada zona a lo largo de un eje."""
    return list(range(0, tam - TAM_ZONA + 1, PASO_ZONA))


def forma_rejilla(tam: int) -> tuple[int, int]:
    """Cuántas zonas hay por fila y por columna (15x15 a 256 px)."""
    n = len(posiciones(tam))
    return n, n


def atributos_color(zona_hsv: np.ndarray) -> np.ndarray:
    """Histograma normalizado de los canales H (tono), S (saturación), V (brillo)."""
    maximos = (180, 256, 256)        # en OpenCV el tono H va de 0 a 179
    hist = [np.histogram(zona_hsv[..., c], bins=BINS_COLOR,
                         range=(0, maximos[c]))[0] for c in range(3)]
    return np.concatenate(hist) / zona_hsv[..., 0].size   # proporciones 0-1


def atributos_textura(zona_gris: np.ndarray) -> np.ndarray:
    """Atributos de la GLCM de una zona ya reducida a NIVELES_GRIS niveles.

    La GLCM cuenta cuántas veces un píxel de nivel i tiene a su vecino
    (a distancia d y ángulo a) con nivel j. Una superficie lisa concentra los
    conteos cerca de la diagonal (vecinos parecidos); una grieta los dispersa.
    """
    glcm = graycomatrix(zona_gris, distances=DISTANCIAS, angles=ANGULOS,
                        levels=NIVELES_GRIS, symmetric=True, normed=True)
    # glcm tiene forma (16, 16, n_distancias, n_angulos)
    energia = graycoprops(glcm, "ASM")              # suma P^2: uniformidad
    correlacion = graycoprops(glcm, "correlation")  # dependencia lineal
    inercia = graycoprops(glcm, "contrast")         # suma (i-j)^2 P: cambios bruscos
    homogeneidad = graycoprops(glcm, "homogeneity") # vecinos parecidos
    entropia = -np.sum(glcm * np.log2(glcm + 1e-12), axis=(0, 1))  # desorden
    # Promedio de los 4 ángulos: la avellana puede estar girada en la foto,
    # así no importa la orientación de la grieta.
    atributos = [m.mean(axis=1) for m in
                 (energia, entropia, correlacion, inercia, homogeneidad)]
    return np.nan_to_num(np.stack(atributos, axis=1).ravel())


def caracteristicas_imagen(ruta: str, tam: int) -> np.ndarray:
    """Vector de 34 atributos para cada zona de una imagen.

    Recibe la RUTA y no la imagen: en la versión paralela, a cada proceso se le
    envía un texto corto en lugar de una matriz grande (menos comunicación).

    Devuelve una matriz (n_zonas, 34) en float32.
    """
    rgb = cargar_imagen(ruta, tam)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gris_16 = (gris.astype(np.uint16) * NIVELES_GRIS // 256).astype(np.uint8)

    pos = posiciones(tam)
    salida = np.empty((len(pos) ** 2, DIM), dtype=np.float32)
    k = 0
    for y in pos:                        # recorre las zonas fila por fila
        for x in pos:
            zona = (slice(y, y + TAM_ZONA), slice(x, x + TAM_ZONA))
            salida[k, :3 * BINS_COLOR] = atributos_color(hsv[zona])
            salida[k, 3 * BINS_COLOR:] = atributos_textura(gris_16[zona])
            k += 1
    return salida


if __name__ == "__main__":
    # Prueba rápida: python -m src.comun.caracteristicas
    import time
    from src.comun.datos import muestras_prueba

    muestras = muestras_prueba()
    sana = next(m for m in muestras if m.defecto == "good")
    grieta = next(m for m in muestras if m.defecto == "crack")

    for tam in (256, 512):
        t0 = time.perf_counter()
        f = caracteristicas_imagen(sana.ruta, tam)
        t = time.perf_counter() - t0
        print(f"{tam} px: {f.shape[0]} zonas x {f.shape[1]} atributos "
              f"-> {t:.2f} s por imagen")

    # ¿La textura cambia donde hay defecto? Comparamos la inercia (contraste)
    # máxima de una avellana sana y de una con grieta.
    i_inercia = 3 * BINS_COLOR + NOMBRES_TEXTURA.index("inercia")
    for nombre, m in (("sana", sana), ("con grieta", grieta)):
        f = caracteristicas_imagen(m.ruta, 256)
        print(f"Inercia máxima, avellana {nombre}: {f[:, i_inercia].max():.2f}")
