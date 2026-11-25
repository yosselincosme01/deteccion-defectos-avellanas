"""Modelo del método clásico (GMM) y flujo completo, compartido por la
versión secuencial y la paralela.

La ÚNICA diferencia entre ambas versiones es la función `extraer` que recibe
`ejecutar()`: una procesa las imágenes una tras otra y la otra las reparte
entre procesos. Todo lo demás (GMM, puntajes, mapas) es el mismo código, así
que si los atributos son iguales, los resultados también lo son.

Etapas del flujo:
  1. Extraer atributos de todas las imágenes (entrenamiento + prueba)  <- paralelizable
  2. Ajustar el GMM con las zonas de avellanas sanas                   <- secuencial
  3. Puntuar cada zona de prueba y armar el mapa de defectos           <- rápido
"""
import time

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.comun.caracteristicas import forma_rejilla
from src.comun.configuracion import SEMILLA
from src.comun.datos import imagenes_entrenamiento, muestras_prueba

COMPONENTES_GMM = 8         # cuántos "grupos" de zonas sanas aprende el GMM
MAX_ZONAS_GMM = 100_000     # tope de zonas para ajustar (acota el tiempo)


class ModeloClasico:
    """Aprende cómo son las zonas sanas y puntúa qué tan raras son zonas nuevas.

    GMM (modelo de mezcla gaussiana): supone que las zonas sanas forman
    varios grupos (cáscara clara, cáscara oscura, punta, fondo negro...) y
    modela cada grupo con una campana de Gauss en el espacio de 34 atributos.
    Una zona nueva que cae lejos de todas las campanas es poco probable
    -> posible defecto.
    """

    def __init__(self, componentes: int = COMPONENTES_GMM):
        self.escala = StandardScaler()
        self.gmm = GaussianMixture(n_components=componentes,
                                   covariance_type="full", reg_covar=1e-3,
                                   random_state=SEMILLA, max_iter=200)

    def ajustar(self, zonas_sanas: np.ndarray) -> "ModeloClasico":
        """Aprende a partir de zonas de avellanas SANAS únicamente."""
        if len(zonas_sanas) > MAX_ZONAS_GMM:   # submuestra fija (semilla)
            idx = np.random.default_rng(SEMILLA).choice(
                len(zonas_sanas), MAX_ZONAS_GMM, replace=False)
            zonas_sanas = zonas_sanas[np.sort(idx)]
        # Estandarizar: cada atributo con media 0 y desviación 1, para que
        # ninguno pese más solo por estar en otra escala.
        x = self.escala.fit_transform(zonas_sanas)
        self.gmm.fit(x)
        return self

    def puntuar_zonas(self, zonas: np.ndarray) -> np.ndarray:
        """Anomalía de cada zona = -log(probabilidad bajo el GMM).

        Zona muy parecida a las sanas -> probabilidad alta -> puntaje bajo.
        Zona rara (grieta, mancha)    -> probabilidad baja -> puntaje alto.
        """
        return -self.gmm.score_samples(self.escala.transform(zonas))


def mapa_defectos(puntajes_zonas: np.ndarray, tam: int) -> np.ndarray:
    """Convierte los puntajes de la rejilla de zonas en un mapa tam x tam."""
    rejilla = puntajes_zonas.reshape(forma_rejilla(tam)).astype(np.float32)
    mapa = cv2.resize(rejilla, (tam, tam), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(mapa, (0, 0), sigmaX=tam / 64)   # suaviza bordes


def ejecutar(extraer, tam: int = 256, fraccion: float = 1.0) -> dict:
    """Corre el método clásico de principio a fin y mide cada etapa.

    `extraer(rutas, tam)` debe devolver (lista_de_atributos, info_dict).
    """
    rutas_ent = imagenes_entrenamiento(fraccion=fraccion)
    prueba = muestras_prueba()
    rutas = rutas_ent + [m.ruta for m in prueba]

    t0 = time.perf_counter()
    atributos, info = extraer(rutas, tam)                   # 1. paralelizable
    t1 = time.perf_counter()
    zonas_sanas = np.concatenate(atributos[:len(rutas_ent)])
    modelo = ModeloClasico().ajustar(zonas_sanas)           # 2. secuencial
    t2 = time.perf_counter()
    puntajes, mapas = [], []
    for a in atributos[len(rutas_ent):]:                    # 3. puntuación
        mapa = mapa_defectos(modelo.puntuar_zonas(a), tam)
        mapas.append(mapa)
        puntajes.append(float(mapa.max()))   # puntaje = zona más rara
    t3 = time.perf_counter()

    etiquetas = [m.etiqueta for m in prueba]
    return {
        "tam": tam, "fraccion": fraccion,
        "n_entrenamiento": len(rutas_ent), "n_prueba": len(prueba),
        "t_extraccion": t1 - t0, "t_gmm": t2 - t1,
        "t_puntuacion": t3 - t2, "t_total": t3 - t0,
        **info,
        # AUROC rápido para comprobar calidad; las métricas completas
        # (F1, AUROC por píxel, IoU...) están en src/comun/metricas.py.
        "auroc_imagen": float(roc_auc_score(etiquetas, puntajes)),
        "atributos": atributos, "puntajes": puntajes, "mapas": mapas,
        "muestras": prueba,
    }


def resumen(r: dict) -> str:
    """Una línea legible con lo más importante de una ejecución."""
    return (f"tam={r['tam']} px | entrenamiento={r['n_entrenamiento']} | "
            f"procesos={r.get('procesos', 1)} | "
            f"extracción={r['t_extraccion']:.2f} s | GMM={r['t_gmm']:.2f} s | "
            f"puntuación={r['t_puntuacion']:.2f} s | total={r['t_total']:.2f} s"
            f" | AUROC={r['auroc_imagen']:.3f}")
