# 📚 Datos: MVTec AD, categoría *hazelnut*

Las imágenes **no se incluyen** en este repositorio: pesan unos 600 MB y su licencia no permite redistribuirlas. Cada persona debe descargarlas.

## Fuente

- **Dataset:** MVTec Anomaly Detection (MVTec AD)
- **Autores:** MVTec Software GmbH
- **Página oficial:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Artículo:** Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D. y Steger, C. (2021). The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. *International Journal of Computer Vision*, 129, 1038–1059.
- **Licencia:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Uso no comercial, con atribución y compartiendo bajo la misma licencia.

## Cómo descargarlo

1. Entrar a https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
2. Completar el formulario y aceptar la licencia.
3. Descargar **solo la categoría `hazelnut`** (archivo `hazelnut.tar.xz`, ~600 MB).
4. Descomprimirlo dentro de esta carpeta `datos/`:

```powershell
tar -xf "$HOME\Downloads\hazelnut.tar.xz" -C datos
```

## Estructura esperada

```
datos/hazelnut/
├── train/good/              391 avellanas sanas (para aprender)
├── test/
│   ├── good/                 40 sanas (para evaluar)
│   ├── crack/                18 con grieta
│   ├── cut/                  17 con corte
│   ├── hole/                 18 con agujero
│   └── print/                17 con impresión
└── ground_truth/            máscaras de los 70 defectos (blanco = defecto)
    ├── crack/  cut/  hole/  print/
```

- Imágenes RGB de 1024 × 1024 píxeles, fondo negro y avellanas en distintas orientaciones.
- `ground_truth` no tiene carpeta `good`: una avellana sana no tiene nada que marcar. El código genera para ella una máscara vacía.
- Cada `test/<defecto>/005.png` tiene su máscara en `ground_truth/<defecto>/005_mask.png`.

## Comprobar la descarga

Desde la raíz del proyecto:

```powershell
python -m src.comun.datos
```

```
Entrenamiento: 391 sanas
Prueba: 110 imágenes (40 sanas, 70 con defecto)
```
