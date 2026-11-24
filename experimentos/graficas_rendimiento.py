"""Figuras y tabla de rendimiento a partir de resultados/rendimiento.csv.

Genera:
  figuras/fig_speedup_eficiencia.png  speedup y eficiencia vs. procesos
  figuras/fig_desglose_tiempos.png    en qué se va el tiempo de cada config.
  resultados/tabla_rendimiento.md     media ± desviación, speedup, eficiencia

Speedup    S(p) = T(1) / T(p)    -> cuántas veces más rápido
Eficiencia E(p) = S(p) / p       -> cuánto aprovecha cada proceso (1 = ideal)

Se calculan dos versiones:
  * de la EXTRACCIÓN (la parte que se paraleliza), y
  * del TOTAL (incluye el GMM secuencial -> muestra la ley de Amdahl).

¿Por qué el MÍNIMO de las repeticiones y no el promedio?
  El ruido externo (programas de fondo, calentamiento, núcleos de ahorro)
  solo puede SUMAR tiempo, nunca restarlo. El mínimo es la medición menos
  contaminada y la mejor estimación del costo real (es lo que recomienda la
  documentación del módulo timeit de Python). Para no ocultar la variación,
  la tabla incluye también la mediana, el máximo y la desviación estándar,
  y las gráficas muestran el rango con barras.

Uso:
    python -m experimentos.graficas_rendimiento
    python -m experimentos.graficas_rendimiento --archivo resultados/rendimiento_prueba.csv
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # dibuja sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
AZUL, NARANJA, GRIS, VERDE = "#2a6fdb", "#e8833a", "#9e9e9e", "#4caf50"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def resumir(df: pd.DataFrame) -> pd.DataFrame:
    """Mínimo, mediana, máximo y desviación por (tamaño, procesos),
    + speedup y eficiencia calculados con el mínimo."""
    df = df.fillna({"t_crear_pool": 0, "t_cerrar_pool": 0})
    g = df.groupby(["tam", "procesos"]).agg(
        reps=("t_total", "size"),
        ext_min=("t_extraccion", "min"), ext_mediana=("t_extraccion", "median"),
        ext_max=("t_extraccion", "max"), ext_desv=("t_extraccion", "std"),
        tot_min=("t_total", "min"), tot_mediana=("t_total", "median"),
        tot_max=("t_total", "max"), tot_desv=("t_total", "std"),
        crear_pool=("t_crear_pool", "min"), calculo=("t_calculo", "min"),
        gmm=("t_gmm", "min"), puntuacion=("t_puntuacion", "min"),
        auroc=("auroc_imagen", "mean")).reset_index()
    for parte in ("ext", "tot"):
        t1 = g[g.procesos == 1].set_index("tam")
        base = g.tam.map(t1[f"{parte}_min"])
        g[f"speedup_{parte}"] = base / g[f"{parte}_min"]
        g[f"eficiencia_{parte}"] = g[f"speedup_{parte}"] / g.procesos
        # rango del speedup: peor y mejor caso posibles con las repeticiones
        g[f"speedup_{parte}_bajo"] = g.tam.map(t1[f"{parte}_min"]) / g[f"{parte}_max"]
        g[f"speedup_{parte}_alto"] = g.tam.map(t1[f"{parte}_max"]) / g[f"{parte}_min"]
    return g


def figura_speedup(g: pd.DataFrame, destino: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    p = sorted(g.procesos.unique())
    for tam, color in zip(sorted(g.tam.unique()), (AZUL, NARANJA)):
        s = g[g.tam == tam]
        a1.plot(s.procesos, s.speedup_ext, "o-", color=color,
                label=f"{tam} px · extracción")
        a1.plot(s.procesos, s.speedup_tot, "s--", color=color, alpha=.55,
                label=f"{tam} px · total")
        a2.plot(s.procesos, s.eficiencia_ext, "o-", color=color,
                label=f"{tam} px · extracción")
        a2.plot(s.procesos, s.eficiencia_tot, "s--", color=color, alpha=.55,
                label=f"{tam} px · total")
    a1.plot(p, p, ":", color=GRIS, label="ideal")
    a2.axhline(1, ls=":", color=GRIS)
    a1.set(xlabel="Procesos (p)", ylabel="Speedup  T(1) / T(p)",
           title="Aceleración (mínimo de 3 repeticiones)", xticks=p)
    a2.set(xlabel="Procesos (p)", ylabel="Eficiencia  S(p) / p",
           title="Eficiencia", xticks=p,
           ylim=(0, max(1.15, g[["eficiencia_ext", "eficiencia_tot"]].max().max() + .1)))
    a1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def figura_desglose(g: pd.DataFrame, destino: Path) -> None:
    tams = sorted(g.tam.unique())
    fig, ejes = plt.subplots(1, len(tams), figsize=(5 * len(tams), 4),
                             squeeze=False)
    partes = [("crear_pool", "Crear procesos", GRIS),
              ("calculo", "Extracción (paralelizable)", AZUL),
              ("gmm", "Ajuste GMM (secuencial)", NARANJA),
              ("puntuacion", "Puntuación", VERDE)]
    for eje, tam in zip(ejes[0], tams):
        s = g[g.tam == tam].reset_index(drop=True)
        abajo = pd.Series(0.0, index=s.index)
        for col, nombre, color in partes:
            eje.bar(s.procesos.astype(str), s[col], bottom=abajo,
                    color=color, label=nombre)
            abajo += s[col]
        eje.set(xlabel="Procesos (p)", ylabel="Tiempo mínimo (s)",
                title=f"{tam} px")
    ejes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archivo", default=str(RAIZ / "resultados" / "rendimiento.csv"))
    args = ap.parse_args()

    g = resumir(pd.read_csv(args.archivo))
    (RAIZ / "figuras").mkdir(exist_ok=True)
    figura_speedup(g, RAIZ / "figuras" / "fig_speedup_eficiencia.png")
    figura_desglose(g, RAIZ / "figuras" / "fig_desglose_tiempos.png")

    tabla = g[["tam", "procesos", "reps", "ext_min", "ext_mediana", "ext_max",
               "ext_desv", "tot_min", "tot_mediana", "speedup_ext",
               "eficiencia_ext", "speedup_tot", "eficiencia_tot",
               "gmm", "auroc"]].round(3)
    (RAIZ / "resultados" / "tabla_rendimiento.md").write_text(
        tabla.to_markdown(index=False), encoding="utf-8")
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
