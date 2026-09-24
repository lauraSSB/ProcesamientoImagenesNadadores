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
# ISL, IL, ILP – vistas laterales: rango HSV completo + morfología
# ---------------------------------------------------------------------------

# H: 0-179, S: 0-255, V: 0-255
RANGOS_AGUA: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "ISL": (np.array([85,  40,  70]), np.array([135, 255, 255])),
    "IL":  (np.array([85,  30,  60]), np.array([135, 255, 255])),
    "ILP": (np.array([80,  40,  30]), np.array([140, 255, 200])),
}
RANGO_DEFAULT = (np.array([85, 30, 30]), np.array([140, 255, 255]))

VISTAS_LATERALES = {"IL", "ISL"}

VISTAS = ("IS", *RANGOS_AGUA)


def segmentar_agua(frame_bgr: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Máscara de agua para vistas laterales (ISL, IL, ILP)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara


# ---------------------------------------------------------------------------
# Pipeline por imagen
# ---------------------------------------------------------------------------

def procesar_imagen(
    ruta: Path,
    lower: np.ndarray,
    upper: np.ndarray,
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
        mascara = segmentar_agua(frame, lower, upper)
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
    parser.add_argument(
        "--vista", default="IL", choices=VISTAS,
        help="Tipo de vista (por defecto: IL).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lower, upper = RANGOS_AGUA.get(args.vista, RANGO_DEFAULT)

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
        print(f"Vista: {args.vista} | HSV agua lower={lower.tolist()} upper={upper.tolist()}")
    print(f"Imágenes: {len(rutas)} | Salida: {carpeta_salida}\n")

    for ruta in rutas:
        procesar_imagen(ruta, lower, upper, args.vista, carpeta_salida)

    print(f"\nListo. Revisá los resultados en {carpeta_salida}/")


if __name__ == "__main__":
    main()
