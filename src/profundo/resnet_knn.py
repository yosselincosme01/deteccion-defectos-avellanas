"""Método profundo: ResNet18 preentrenada + vecino más cercano.

Idea (versión simplificada de PatchCore, Roth et al., 2022):
  1. ResNet18, una red neuronal YA ENTRENADA con ImageNet (más de un millón
     de fotos de todo tipo), describe cada zona de la imagen con 384 números
     tomados de sus capas intermedias (layer2 y layer3). NO se reentrena.
  2. Con las avellanas sanas se arma un "banco de memoria" de zonas sanas.
  3. Cada zona de una imagen de prueba se compara con el banco y se busca la
     zona sana MÁS PARECIDA (vecino más cercano, distancia euclidiana).
     Si incluso la más parecida está lejos, la zona es anómala.
  4. Esas distancias forman el mapa de defectos; su máximo es el puntaje.

Diferencias con la rama clásica:
  * Atributos: aprendidos por la red (no diseñados a mano como color/GLCM).
  * Detector: vecino más cercano (no GMM).
  * Paralelismo: PyTorch ya reparte las operaciones entre todos los núcleos
    de la CPU (hilos internos) o usa la GPU. No lo programamos nosotras.

Uso (desde la raíz del proyecto):
    python -m src.profundo.resnet_knn
    python -m src.profundo.resnet_knn --tam 512
    python -m src.profundo.resnet_knn --fraccion 0.25
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.comun.configuracion import SEMILLA
from src.comun.datos import cargar_imagen, imagenes_entrenamiento, muestras_prueba
from src.comun.metricas import resumen_calidad

# Normalización con la que se entrenó la red en ImageNet
MEDIA = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DESV = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TAM_BANCO = 50_000    # zonas sanas guardadas en memoria
LOTE = 16             # imágenes que pasan juntas por la red


def sincronizar(disp: torch.device) -> None:
    """En GPU las operaciones son ASÍNCRONAS: la CPU sigue avanzando mientras
    la GPU trabaja. Sin esperar a que termine, time.perf_counter() mediría
    solo el tiempo de "encargar" el trabajo, no el de hacerlo.
    En CPU no hace nada."""
    if disp.type == "cuda":
        torch.cuda.synchronize(disp)


class ExtractorResNet:
    """Convierte imágenes en un vector de 384 números por cada zona."""

    def __init__(self, dispositivo: str = "cpu"):
        self.disp = torch.device(dispositivo)
        self.red = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.red.eval().to(self.disp)       # modo evaluación: no se entrena
        self.t_hacia_disp = 0.0             # tiempo copiando imágenes a la GPU

    @torch.no_grad()                        # no calcula gradientes (más rápido)
    def __call__(self, imagenes: np.ndarray) -> torch.Tensor:
        """imagenes: (B, tam, tam, 3) uint8 -> (B, 384, tam/8, tam/8)"""
        x = torch.from_numpy(imagenes).permute(0, 3, 1, 2).float() / 255.0
        x = (x - MEDIA) / DESV
        sincronizar(self.disp)
        t = time.perf_counter()
        x = x.to(self.disp)                 # transferencia CPU -> GPU
        sincronizar(self.disp)
        self.t_hacia_disp += time.perf_counter() - t
        r = self.red
        x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        f2 = r.layer2(r.layer1(x))     # 128 canales, 1/8 del tamaño: detalles finos
        f3 = r.layer3(f2)              # 256 canales, 1/16: patrones más amplios
        # Suavizado local: cada zona "ve" también a sus vecinas
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        # Se agranda f3 al tamaño de f2 y se juntan: 128 + 256 = 384 números
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear",
                           align_corners=False)
        return torch.cat([f2, f3], dim=1)


def _lotes(rutas: list[str], tam: int):
    """Entrega las imágenes de LOTE en LOTE."""
    for i in range(0, len(rutas), LOTE):
        yield np.stack([cargar_imagen(r, tam) for r in rutas[i:i + LOTE]])


def _a_zonas(f: torch.Tensor) -> torch.Tensor:
    """(B, 384, h, w) -> (B, h*w, 384): una fila por zona."""
    return f.permute(0, 2, 3, 1).reshape(f.shape[0], -1, f.shape[1])


def ejecutar(tam: int = 256, fraccion: float = 1.0, dispositivo: str = "cpu",
             hilos: int | None = None) -> dict:
    """Corre el método profundo de principio a fin y mide cada etapa."""
    torch.manual_seed(SEMILLA)
    # configuracion.py fija 1 hilo para la rama clásica; aquí le devolvemos
    # a PyTorch todos los núcleos (su paralelismo interno).
    torch.set_num_threads(hilos or os.cpu_count())

    rutas_ent = imagenes_entrenamiento(fraccion=fraccion)
    prueba = muestras_prueba()
    ext = ExtractorResNet(dispositivo)
    disp = ext.disp
    if disp.type == "cuda":
        torch.cuda.reset_peak_memory_stats(disp)
    rng = np.random.default_rng(SEMILLA)
    por_imagen = math.ceil(TAM_BANCO / len(rutas_ent))

    # 1. Banco de memoria: zonas de avellanas sanas (submuestra fija)
    sincronizar(disp)
    t0 = time.perf_counter()
    banco = []
    for imgs in _lotes(rutas_ent, tam):
        for zonas in _a_zonas(ext(imgs)):
            k = min(por_imagen, len(zonas))
            idx = torch.from_numpy(rng.choice(len(zonas), k, replace=False))
            banco.append(zonas[idx.to(zonas.device)])
    banco = torch.cat(banco)
    sincronizar(disp)
    t1 = time.perf_counter()

    # 2. Imágenes de prueba: red + búsqueda del vecino más cercano
    t_red = t_knn = t_desde_disp = 0.0
    puntajes, mapas = [], []
    for imgs in _lotes([m.ruta for m in prueba], tam):
        ta = time.perf_counter()
        f = ext(imgs)
        h, w = f.shape[-2:]
        sincronizar(disp)
        tb = time.perf_counter()
        for zonas in _a_zonas(f):
            # distancia de cada zona a TODO el banco; nos quedamos con la menor
            dist = torch.cat([torch.cdist(z, banco).min(dim=1).values
                              for z in zonas.split(1024)])
            sincronizar(disp)
            tc = time.perf_counter()
            rejilla = dist.reshape(h, w).cpu().numpy().astype(np.float32)  # GPU -> CPU
            t_desde_disp += time.perf_counter() - tc
            mapa = cv2.resize(rejilla, (tam, tam), interpolation=cv2.INTER_LINEAR)
            mapa = cv2.GaussianBlur(mapa, (0, 0), sigmaX=tam / 64)
            mapas.append(mapa)
            puntajes.append(float(mapa.max()))
        t_red += tb - ta
        t_knn += time.perf_counter() - tb
    t2 = time.perf_counter()

    return {
        "metodo": "profundo", "tam": tam, "fraccion": fraccion,
        "dispositivo": dispositivo, "hilos": torch.get_num_threads(),
        "n_entrenamiento": len(rutas_ent), "n_prueba": len(prueba),
        "tam_banco": int(banco.shape[0]),
        "t_banco": t1 - t0,          # red sobre las sanas + armar el banco
        "t_red_prueba": t_red,       # red sobre las imágenes de prueba
        "t_knn": t_knn,              # búsqueda del vecino más cercano
        "t_total": t2 - t0,
        # Transferencias (en CPU son ~0): imágenes hacia el dispositivo y
        # mapas de vuelta. Están INCLUIDAS en t_banco / t_red_prueba / t_knn.
        "t_hacia_disp": ext.t_hacia_disp,
        "t_desde_disp": t_desde_disp,
        "mem_gpu_pico_mb": (torch.cuda.max_memory_allocated(disp) / 2**20
                            if disp.type == "cuda" else 0.0),
        **resumen_calidad(prueba, puntajes, mapas),
        "puntajes": puntajes, "mapas": mapas, "muestras": prueba,
    }


def main():
    ap = argparse.ArgumentParser(description="Método profundo: ResNet18 + vecino más cercano")
    ap.add_argument("--tam", type=int, default=256)
    ap.add_argument("--fraccion", type=float, default=1.0)
    ap.add_argument("--dispositivo", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    r = ejecutar(args.tam, args.fraccion, args.dispositivo)
    print(f"[profundo] tam={r['tam']} px | entrenamiento={r['n_entrenamiento']} | "
          f"hilos={r['hilos']} | banco={r['t_banco']:.1f} s | "
          f"red prueba={r['t_red_prueba']:.1f} s | vecino={r['t_knn']:.1f} s | "
          f"total={r['t_total']:.1f} s")
    for k in ("auroc_imagen", "auroc_pixel", "precision", "recall", "f1",
              "falsos_positivos", "falsos_negativos", "auroc_crack",
              "auroc_cut", "auroc_hole", "auroc_print"):
        v = r[k]
        print(f"  {k:18s} {v:.3f}" if isinstance(v, float) else f"  {k:18s} {v}")


if __name__ == "__main__":
    main()
