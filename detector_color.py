"""Detección del área de piscina por segmentación de color HSV.

Genera una máscara binaria donde la piscina (agua) es blanca
y los separadores de carril (cuerdas/boyas) son negros.

Cada vista tiene su propio método de segmentación:
  IS  → solo canal H + votación vertical 1×15 + mediana 3×3
  ISL, IL, ILP → rango HSV completo (H+S+V) + limpieza morfológica

Uso:
  python detector_color.py                     # procesa vistas IL
  python detector_color.py --vista IS
  python detector_color.py --imagen data/images/IL/IL_00099.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "output" / "color"


# ---------------------------------------------------------------------------
# IS – vista superior: solo canal H, votación vertical y mediana
# ---------------------------------------------------------------------------

H_AGUA_IS = (90, 110)


def _votacion(mascara: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Promedio en ventana y umbral 0.5 (votación de mayoría)."""
    return ((cv2.blur(mascara, (ancho, alto)) > 127) * 255).astype(np.uint8)


def segmentar_is(frame_bgr: np.ndarray) -> np.ndarray:
    """Máscara de agua para vista superior (IS)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    agua = cv2.inRange(hsv[..., 0], *H_AGUA_IS)
    return cv2.medianBlur(_votacion(agua, 1, 15), 3)


# ---------------------------------------------------------------------------
# ISL, IL – vistas laterales: solo canal H + votación direccional + mediana
# ---------------------------------------------------------------------------

# Rango de tono H del agua (mismo criterio que IS).
H_AGUA_LATERAL = (90, 110)

# Kernel de votación por vista:
#   IL  → 15×1 (horizontal): refuerza cuerdas que corren horizontalmente
#   ISL → 5×5  (cuadrado):   sin dirección privilegiada (cuerdas diagonales)
_KERNEL_VOTACION: dict[str, tuple[int, int]] = {
    "IL":  (15, 1),
    "ISL": (5, 5),
}

VISTAS_LATERALES = {"IL", "ISL"}
VISTAS = ("IS", "IL", "ISL", "ILP")


def segmentar_agua(frame_bgr: np.ndarray, vista: str) -> np.ndarray:
    """Máscara de agua para vistas laterales (IL, ISL).

    Usa solo el canal H (90-110) con votación de mayoría en una ventana
    direccional y una mediana 3×3 final — mismo enfoque que IS pero con
    el kernel orientado según la dirección de las cuerdas en cada vista.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    agua = cv2.inRange(hsv[..., 0], *H_AGUA_LATERAL)
    kw, kh = _KERNEL_VOTACION.get(vista, (5, 5))
    votacion = ((cv2.blur(agua, (kw, kh)) > 127) * 255).astype(np.uint8)
    return cv2.medianBlur(votacion, 3)


# ---------------------------------------------------------------------------
# Pipeline por imagen
# ---------------------------------------------------------------------------

def procesar_imagen(
    ruta: Path,
    vista: str,
    carpeta_salida: Path,
) -> None:
    frame = cv2.imread(str(ruta))
    if frame is None:
        print(f"  [!] No se pudo leer {ruta.name}")
        return

    if vista == "IS":
        mascara = segmentar_is(frame)
    elif vista in VISTAS_LATERALES:
        mascara = segmentar_agua(frame, vista)
    else:
        print(f"  [!] Vista '{vista}' no implementada aún.")
        return

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(carpeta_salida / f"{ruta.stem}_mascara.jpg"), mascara)
    print(f"  {ruta.name} → OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera máscara de piscina: blanco=agua, negro=separadores/fondo."
    )
    parser.add_argument("--imagen", type=Path, help="Procesa una sola imagen.")
    parser.add_argument("--vista", default="IL", choices=VISTAS,
                        help="Tipo de vista (por defecto: IL).")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.imagen:
        rutas = [args.imagen]
    else:
        rutas = sorted((IMAGES_DIR / args.vista).glob("*.jpg"))
        if not rutas:
            raise SystemExit(f"No hay imágenes en data/images/{args.vista}/")

    carpeta_salida = OUT_DIR / args.vista

    if args.vista == "IS":
        print(f"Vista: IS | H {H_AGUA_IS[0]}-{H_AGUA_IS[1]}, votación 1×15, mediana 3×3")
    else:
        kw, kh = _KERNEL_VOTACION.get(args.vista, (5, 5))
        print(f"Vista: {args.vista} | H {H_AGUA_LATERAL[0]}-{H_AGUA_LATERAL[1]}, votación {kw}×{kh}, mediana 3×3")
    print(f"Imágenes: {len(rutas)} | Salida: {carpeta_salida}\n")

    for ruta in rutas:
        procesar_imagen(ruta, args.vista, carpeta_salida)

    print(f"\nListo. Revisá los resultados en {carpeta_salida}/")


if __name__ == "__main__":
    main()
