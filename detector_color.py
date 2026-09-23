"""Paso 1 y 2: Detección del nadador por color HSV.

Segmenta el agua de la piscina por su rango de color azul/cian en HSV
e invierte la máscara para obtener la región del nadador.

Uso:
  python detector_color.py                         # procesa las 20 imágenes IS
  python detector_color.py --vista IL              # otra vista
  python detector_color.py --imagen data/images/IS/IS_00123.jpg  # imagen puntual
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "output" / "color"

# Rangos HSV del agua de piscina por tipo de vista.
# H: 0-179, S: 0-255, V: 0-255
RANGOS_AGUA: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "IS":  (np.array([85,  50,  80]), np.array([135, 255, 255])),
    "ISL": (np.array([85,  40,  70]), np.array([135, 255, 255])),
    "IL":  (np.array([85,  30,  60]), np.array([135, 255, 255])),
    "ILP": (np.array([80,  40,  30]), np.array([140, 255, 200])),
}
RANGO_DEFAULT = (np.array([85, 30, 30]), np.array([140, 255, 255]))


def segmentar(frame_bgr: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, np.ndarray]:
    """Devuelve la máscara del agua y la máscara del nadador."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mascara_agua = cv2.inRange(hsv, lower, upper)
    mascara_nadador = cv2.bitwise_not(mascara_agua)

    # Limpieza morfológica básica para quitar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara_nadador = cv2.morphologyEx(mascara_nadador, cv2.MORPH_OPEN, kernel)
    mascara_nadador = cv2.morphologyEx(mascara_nadador, cv2.MORPH_CLOSE, kernel)

    return {
        "mascara_agua": mascara_agua,
        "mascara_nadador": mascara_nadador,
    }


def procesar_imagen(ruta: Path, lower: np.ndarray, upper: np.ndarray, carpeta_salida: Path) -> None:
    frame = cv2.imread(str(ruta))
    if frame is None:
        print(f"  [!] No se pudo leer {ruta.name}")
        return

    masks = segmentar(frame, lower, upper)

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    stem = ruta.stem

    cv2.imwrite(str(carpeta_salida / f"{stem}_mascara_agua.jpg"), masks["mascara_agua"])
    cv2.imwrite(str(carpeta_salida / f"{stem}_mascara_nadador.jpg"), masks["mascara_nadador"])

    print(f"  {ruta.name} → {carpeta_salida.name}/")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detecta al nadador por segmentación de color HSV.")
    parser.add_argument("--imagen", type=Path, help="Procesa una sola imagen.")
    parser.add_argument("--vista", default="IS", choices=list(RANGOS_AGUA.keys()),
                        help="Tipo de vista a procesar (por defecto: IS).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lower, upper = RANGOS_AGUA.get(args.vista, RANGO_DEFAULT)

    if args.imagen:
        rutas = [args.imagen]
        carpeta_salida = OUT_DIR / args.vista
    else:
        rutas = sorted((IMAGES_DIR / args.vista).glob("*.jpg"))
        carpeta_salida = OUT_DIR / args.vista
        if not rutas:
            raise SystemExit(f"No hay imágenes en data/images/{args.vista}/")

    print(f"Vista: {args.vista} | Rango HSV agua: lower={lower.tolist()} upper={upper.tolist()}")
    print(f"Imágenes a procesar: {len(rutas)}")
    print(f"Salida: {carpeta_salida}\n")

    for ruta in rutas:
        procesar_imagen(ruta, lower, upper, carpeta_salida)

    print(f"\nListo. Revisá los resultados en {carpeta_salida}/")
    print("Si la detección es mala ajustá RANGOS_AGUA en el script o corré --calibrar.")


if __name__ == "__main__":
    main()
