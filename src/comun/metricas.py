"""Métricas de calidad: qué tan bien se DETECTAN y se UBICAN los defectos.

Las usan las dos ramas (clásica y profunda) para compararlas en igualdad.

* AUROC por imagen: ¿separa las avellanas sanas de las defectuosas?
  1.0 = perfecto, 0.5 = igual que adivinar. No depende de elegir un umbral:
  mide si, en general, las defectuosas reciben puntajes más altos.

* AUROC por píxel: ¿el mapa de defectos marca los píxeles correctos?
  Compara el mapa con la máscara real (ground_truth), píxel por píxel.

* Precisión, recall y F1: hay que elegir un UMBRAL (puntaje a partir del
  cual decimos "defectuosa"). Elegimos el que maximiza F1.
    - Precisión: de las que marcamos como defectuosas, ¿cuántas lo eran?
    - Recall:    de todas las defectuosas, ¿cuántas encontramos?
    - F1:        equilibrio entre las dos (media armónica).
"""
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

from src.comun.datos import cargar_mascara

TAM_EVAL_PIXEL = 256   # los mapas se comparan a 256 px (igual para todos)


def auroc_imagen(etiquetas, puntajes) -> float:
    """AUROC a nivel de imagen (sana vs. defectuosa)."""
    return float(roc_auc_score(etiquetas, puntajes))


def auroc_pixel(muestras, mapas) -> float:
    """AUROC a nivel de píxel: mapas de defectos vs. máscaras reales."""
    etiquetas, puntajes = [], []
    for m, mapa in zip(muestras, mapas):
        mapa = cv2.resize(np.asarray(mapa, dtype=np.float32),
                          (TAM_EVAL_PIXEL, TAM_EVAL_PIXEL),
                          interpolation=cv2.INTER_LINEAR)
        etiquetas.append(cargar_mascara(m.ruta_mascara, TAM_EVAL_PIXEL).ravel())
        puntajes.append(mapa.ravel())
    return float(roc_auc_score(np.concatenate(etiquetas),
                               np.concatenate(puntajes)))


def mejor_f1(etiquetas, puntajes) -> dict:
    """Umbral que maximiza F1, con su precisión, recall y errores.

    Ojo: el umbral se elige mirando el propio conjunto de prueba (práctica
    común con MVTec AD). Es un valor optimista y así hay que reportarlo.
    """
    p, r, umbrales = precision_recall_curve(etiquetas, puntajes)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.argmax(f1[:-1]))       # el último punto no tiene umbral
    umbral = float(umbrales[i])
    pred = np.asarray(puntajes) >= umbral
    real = np.asarray(etiquetas).astype(bool)
    return {
        "umbral": umbral,
        "precision": float(p[i]),
        "recall": float(r[i]),
        "f1": float(f1[i]),
        "falsos_positivos": int(np.sum(pred & ~real)),   # sana marcada como defecto
        "falsos_negativos": int(np.sum(~pred & real)),   # defecto que se escapó
    }


def resumen_calidad(muestras, puntajes, mapas=None) -> dict:
    """Todas las métricas juntas, incluida el AUROC por tipo de defecto."""
    etiquetas = [m.etiqueta for m in muestras]
    res = {"auroc_imagen": auroc_imagen(etiquetas, puntajes)}
    res.update(mejor_f1(etiquetas, puntajes))
    if mapas is not None:
        res["auroc_pixel"] = auroc_pixel(muestras, mapas)
    # ¿Qué defecto le cuesta más? Sanas vs. cada tipo por separado
    for defecto in ("crack", "cut", "hole", "print"):
        idx = [i for i, m in enumerate(muestras)
               if m.defecto in ("good", defecto)]
        if any(muestras[i].defecto == defecto for i in idx):
            res[f"auroc_{defecto}"] = auroc_imagen(
                [etiquetas[i] for i in idx], [puntajes[i] for i in idx])
    return res


if __name__ == "__main__":
    # Prueba: métricas completas del método clásico a 256 px, usando los
    # mapas que guardó experimentos.rendimiento (sin volver a calcular nada).
    from pathlib import Path
    from src.comun.datos import muestras_prueba

    archivo = Path(__file__).resolve().parents[2] / "resultados" / "mapas_clasico_256.npz"
    datos = np.load(archivo)
    mapas = datos["mapas"]
    puntajes = [float(m.max()) for m in mapas]
    res = resumen_calidad(muestras_prueba(), puntajes, mapas)
    print("Método clásico, 256 px:")
    for k, v in res.items():
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")
