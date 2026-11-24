"""Figuras y tabla de calidad a partir de resultados/calidad.csv.

Genera:
  figuras/fig_pocos_datos.png     AUROC vs. % de datos, ambos métodos (E4)
  figuras/fig_mapas_defectos.png  un ejemplo por defecto: foto, defecto real,
                                  mapa clásico y mapa profundo
  resultados/tabla_calidad.md     todas las métricas por método y condición

Uso:
    python -m experimentos.graficas_calidad
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.comun.datos import cargar_imagen, cargar_mascara, muestras_prueba  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
AZUL, NARANJA = "#2a6fdb", "#e8833a"
NOMBRES = {"clasico": "Clásico (color + GLCM + GMM)",
           "profundo": "Profundo (ResNet18 + vecino)"}
DEFECTOS = {"crack": "grieta", "cut": "corte", "hole": "agujero",
            "print": "impresión"}
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def figura_pocos_datos(df: pd.DataFrame, destino: Path) -> None:
    d = df[df.experimento == "E4"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for metodo, color in (("clasico", AZUL), ("profundo", NARANJA)):
        s = d[d.metodo == metodo].sort_values("fraccion")
        etiquetas = [f"{f:.0%}\n({n})" for f, n in zip(s.fraccion, s.n_entrenamiento)]
        a1.plot(etiquetas, s.auroc_imagen, "o-", color=color, label=NOMBRES[metodo])
        a2.plot(etiquetas, s.auroc_pixel, "o-", color=color, label=NOMBRES[metodo])
    a1.set(title="Detectar: AUROC por imagen",
           xlabel="Datos para aprender (n.º de avellanas sanas)", ylabel="AUROC")
    a2.set(title="Ubicar: AUROC por píxel",
           xlabel="Datos para aprender (n.º de avellanas sanas)")
    a1.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def figura_mapas(destino: Path) -> None:
    clas = np.load(RAIZ / "resultados" / "mapas_clasico_256.npz")["mapas"]
    prof = np.load(RAIZ / "resultados" / "mapas_profundo_256.npz")["mapas"]
    muestras = muestras_prueba()
    elegidas = [next(i for i, m in enumerate(muestras) if m.defecto == d)
                for d in DEFECTOS]
    fig, ejes = plt.subplots(len(elegidas), 4, figsize=(9, 9.5))
    titulos = ["Foto", "Defecto real", "Mapa clásico", "Mapa profundo"]
    for f, i in enumerate(elegidas):
        m = muestras[i]
        img = cargar_imagen(m.ruta, 256)
        ejes[f, 0].imshow(img)
        ejes[f, 1].imshow(cargar_mascara(m.ruta_mascara, 256), cmap="gray")
        for c, mapas in ((2, clas), (3, prof)):
            mapa = mapas[i].astype(np.float32)
            # escala de colores robusta: del percentil 1 al 99 de ese método
            lo, hi = np.percentile(mapas, [1, 99])
            ejes[f, c].imshow(img)
            ejes[f, c].imshow(np.clip((mapa - lo) / (hi - lo), 0, 1),
                              cmap="jet", alpha=.5)
        ejes[f, 0].set_ylabel(f"{m.defecto}\n({DEFECTOS[m.defecto]})")
    for c, t in enumerate(titulos):
        ejes[0, c].set_title(t)
    for e in ejes.ravel():
        e.set_xticks([])
        e.set_yticks([])
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archivo", default=str(RAIZ / "resultados" / "calidad.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.archivo)
    (RAIZ / "figuras").mkdir(exist_ok=True)

    figura_pocos_datos(df, RAIZ / "figuras" / "fig_pocos_datos.png")
    figura_mapas(RAIZ / "figuras" / "fig_mapas_defectos.png")

    columnas = ["experimento", "metodo", "tam", "fraccion", "n_entrenamiento",
                "auroc_imagen", "auroc_pixel", "precision", "recall", "f1",
                "falsos_positivos", "falsos_negativos", "auroc_crack",
                "auroc_cut", "auroc_hole", "auroc_print", "t_total"]
    tabla = (df.groupby(["experimento", "metodo", "tam", "fraccion"], as_index=False)
               [columnas[4:]].agg({c: ("min" if c == "t_total" else "mean")
                                   for c in columnas[4:]})
               .round(3))
    (RAIZ / "resultados" / "tabla_calidad.md").write_text(
        tabla.to_markdown(index=False), encoding="utf-8")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
