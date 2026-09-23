"""Clasifica cada frame de videos de natación con Claude y lo guarda en data/images.

Cada frame se guarda. La llamada a Claude no se repite mientras el plano no cambia:
se envía un frame al inicio de cada corte y, como máximo, cada --max-gap frames.
El resto hereda esa etiqueta. Con --max-gap 1 se consulta cada frame.

Carpetas (la numeración es propia de cada clase, sigue entre videos y no pisa archivos):
  data/images/IS/IS_00001.jpg    vista superior (cenital)
  data/images/ISL/ISL_00001.jpg  superior lateral (elevada, de costado)
  data/images/IL/IL_00001.jpg    lateral fuera de la piscina, a nivel del agua
  data/images/ILP/ILP_00001.jpg  lateral dentro del agua
  data/images/D/D_00001.jpg      desconocido

Uso (PowerShell):
  copy .env.example .env
  # pega ANTHROPIC_API_KEY y, si la clave no está ligada a un workspace, ANTHROPIC_WORKSPACE_ID
  python classify_frames.py --dry-run
  python classify_frames.py
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "data" / "videos"
IMAGES_DIR = ROOT / "data" / "images"
MANIFEST_PATH = IMAGES_DIR / "_manifest.jsonl"

LABELS = ("IS", "ISL", "IL", "ILP", "D")
# 448x252 = 16x9 bloques de 28 px. Claude cobra ceil(w/28)*ceil(h/28) tokens visuales.
API_SIZE = (448, 252)
VISUAL_TOKENS_PER_IMAGE = ((API_SIZE[0] + 27) // 28) * ((API_SIZE[1] + 27) // 28)
PAD = 5
CODE_RE = re.compile(r"\b(ISL|ILP|IS|IL|D)\b")

# USD por millón de tokens (input, output). Solo para el estimado final.
PRICE_PER_MTOK = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
}

EXAMPLES_DIR = ROOT / "data" / "ejemplos"
EXAMPLE_ORDER = ("IS", "ISL", "IL", "ILP")

SYSTEM_PROMPT = """\
Clasifica el plano de cámara de una transmisión de natación. Ignora cronómetro, marcador, banderas y logos. Mira solo hacia dónde apunta el eje de la cámara respecto al agua.

Responde una sola línea con exactamente N códigos, en el mismo orden, separados por un espacio. Sin explicación. No incluyas los ejemplos.

IS = superior. El eje es perpendicular a la superficie: vista en planta, coronal a la piscina. Se ven coronilla y espalda, los carriles cruzan el cuadro como un mapa y el agua llena la imagen. No se ve el muro vertical del fondo ni gente de pie como telón.
ISL = superior lateral. El eje NO es perpendicular: se mira de costado y hacia abajo. Hay perspectiva. Se ve la superficie de varios carriles y también el borde, el muro o la grada. Incluye el plano alto (carriles horizontales, deck a un lado) y el oblicuo más bajo, mientras se mire hacia abajo y no a ras de agua.
IL = lateral fuera de la piscina. El eje es casi horizontal, a la altura del agua. El nadador está de perfil, grande, con salpicadura en primer plano. El muro queda detrás, a la altura de la cabeza. No se mira hacia abajo sobre las calles.
ILP = lateral dentro del agua. Cámara sumergida: azul, burbujas, nadador bajo la superficie, cuerdas cruzando, se ve el fondo o la superficie desde abajo.
D = primer plano, ceremonia, público, pantalla dividida, transición, animación o solo gráfico.

Orden: bajo el agua → ILP. Eje perpendicular, en planta → IS. Eje horizontal a ras de agua → IL. Eje inclinado, de costado y hacia abajo → ISL. Ver el muro no convierte un plano en IL. Ver la piscina desde arriba no convierte un plano en IS si el eje está inclinado.
"""


@dataclass
class Segment:
    video: str
    api_b64: str
    frames: list[tuple[int, bytes]] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


def parse_codes(text: str, n: int) -> list[str] | None:
    """Extrae exactamente n códigos. ISL/ILP van antes que IS/IL en el regex."""
    for line in text.upper().splitlines():
        codes = CODE_RE.findall(line)
        if len(codes) == n:
            return codes
    codes = CODE_RE.findall(text.upper())
    if len(codes) == n:
        return codes
    return None


def make_thumb(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)


def frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(a, b)))


def jpeg_bytes(frame: np.ndarray, quality: int, size: tuple[int, int] | None = None) -> bytes:
    image = frame if size is None else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("OpenCV no pudo codificar el frame a JPEG")
    return buf.tobytes()


def scan_counters(images_dir: Path) -> dict[str, int]:
    counters = {}
    number = re.compile(r"_(\d+)$")
    for label in LABELS:
        folder = images_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        highest = 0
        for path in folder.glob(f"{label}_*.jpg"):
            match = number.search(path.stem)
            if match:
                highest = max(highest, int(match.group(1)))
        counters[label] = highest
    return counters


def load_done(manifest_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not manifest_path.exists():
        return done
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = record.get("file")
            video = record.get("video")
            frame = record.get("frame")
            if not rel or video is None or frame is None:
                continue
            if (IMAGES_DIR / rel).exists():
                done.add((str(video), int(frame)))
    return done


def usage_from_response(usage: object) -> Usage:
    incoming = int(getattr(usage, "input_tokens", 0) or 0)
    incoming += int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    incoming += int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    outgoing = int(getattr(usage, "output_tokens", 0) or 0)
    return Usage(incoming, outgoing)


def response_text(response: object) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)


def load_examples() -> list[tuple[str, str]]:
    """Cuatro planos de referencia, reducidos al mismo tamaño que se envía a Claude."""
    loaded: list[tuple[str, str]] = []
    missing = [label for label in EXAMPLE_ORDER if not (EXAMPLES_DIR / f"{label}.jpg").exists()]
    if missing:
        raise SystemExit(
            "Faltan ejemplos en data/ejemplos: "
            + ", ".join(f"{label}.jpg" for label in missing)
        )
    for label in EXAMPLE_ORDER:
        frame = cv2.imread(str(EXAMPLES_DIR / f"{label}.jpg"))
        if frame is None:
            raise SystemExit(f"No se pudo leer data/ejemplos/{label}.jpg")
        payload = base64.b64encode(jpeg_bytes(frame, 60, API_SIZE)).decode("ascii")
        loaded.append((label, payload))
    return loaded


def classify_batch(
    client: object,
    model: str,
    images_b64: list[str],
    examples: list[tuple[str, str]],
) -> tuple[list[str], Usage]:
    """Pide una etiqueta por imagen. Si la respuesta no cierra, parte el lote."""
    total = Usage()

    def once(images: list[str], extra: str) -> list[str] | None:
        content: list[dict] = [{"type": "text", "text": "EJEMPLOS ya etiquetados. No los cuentes."}]
        for label, payload in examples:
            content.append({"type": "text", "text": f"Ejemplo {label}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": payload,
                    },
                }
            )
        content.append({"type": "text", "text": "Clasifica solo estas imágenes numeradas."})
        for index, payload in enumerate(images, start=1):
            content.append({"type": "text", "text": f"[{index}]"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": payload,
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": (
                    f"N={len(images)}. Responde exactamente {len(images)} códigos, "
                    f"uno por cada imagen numerada, en orden, una línea, separados por espacio. "
                    f"No repitas los ejemplos.{extra}"
                ),
            }
        )
        response = client.messages.create(
            model=model,
            max_tokens=max(32, 6 * len(images)),
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        total.add(usage_from_response(response.usage))
        return parse_codes(response_text(response), len(images))

    labels = once(images_b64, "")
    if labels is None:
        labels = once(images_b64, " La respuesta anterior no servía. Solo la línea de códigos.")
    if labels is not None:
        return labels, total
    if len(images_b64) == 1:
        raise RuntimeError("Claude no devolvió un código válido para un frame")
    mid = len(images_b64) // 2
    left, left_usage = classify_batch(client, model, images_b64[:mid], examples)
    right, right_usage = classify_batch(client, model, images_b64[mid:], examples)
    total.add(left_usage)
    total.add(right_usage)
    return left + right, total


class BatchWriter:
    """Clasifica lotes en paralelo y guarda en orden de frame para que la secuencia no se cruce."""

    def __init__(self, workers: int, classify, counters: dict[str, int], manifest, usage: Usage):
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))
        self._workers = max(1, workers)
        self._classify = classify
        self._counters = counters
        self._manifest = manifest
        self._usage = usage
        self._inflight: list[tuple[Future, list[Segment]]] = []
        self._closed = False
        self.saved = {label: 0 for label in LABELS}
        self.api_images = 0

    def submit(self, segments: list[Segment]) -> None:
        if not segments:
            return
        future = self._executor.submit(self._classify, [segment.api_b64 for segment in segments])
        self._inflight.append((future, segments))
        self._drain(wait=len(self._inflight) >= self._workers)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._drain(wait=True)
        finally:
            self._executor.shutdown(wait=True)

    def _drain(self, wait: bool) -> None:
        while self._inflight:
            future, segments = self._inflight[0]
            if not future.done() and not wait:
                return
            self._inflight.pop(0)
            labels, batch_usage = future.result()
            if len(labels) != len(segments):
                raise RuntimeError(
                    f"Claude devolvió {len(labels)} códigos para {len(segments)} imágenes"
                )
            self._usage.add(batch_usage)
            self.api_images += len(segments)
            for segment, label in zip(segments, labels):
                self._save_segment(segment, label)
            wait = False

    def _save_segment(self, segment: Segment, label: str) -> None:
        if label not in self._counters:
            label = "D"
        folder = IMAGES_DIR / label
        for frame_index, payload in segment.frames:
            self._counters[label] += 1
            number = self._counters[label]
            name = f"{label}_{number:0{PAD}d}.jpg"
            rel = f"{label}/{name}"
            (folder / name).write_bytes(payload)
            record = {
                "video": segment.video,
                "frame": frame_index,
                "label": label,
                "file": rel,
            }
            self._manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.saved[label] += 1
        self._manifest.flush()
        last = segment.frames[-1][0]
        counts = " ".join(f"{key}:{self.saved[key]}" for key in LABELS)
        print(
            f"  {segment.video} frame {last} -> {label} x{len(segment.frames)} | {counts}",
            flush=True,
        )


def video_paths(requested: list[Path] | None) -> list[Path]:
    if requested:
        paths = [path if path.is_absolute() else ROOT / path for path in requested]
    else:
        paths = sorted(VIDEOS_DIR.glob("*.mp4")) + sorted(VIDEOS_DIR.glob("*.avi"))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit("No existe: " + ", ".join(missing))
    if not paths:
        raise SystemExit(f"No hay videos en {VIDEOS_DIR}")
    return paths


def process_video(
    path: Path,
    args: argparse.Namespace,
    done: set[tuple[str, int]],
    writer: BatchWriter | None,
    pending: list[Segment],
) -> dict[str, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_name = path.name
    limit = args.limit_frames if args.limit_frames > 0 else total
    print(
        f"{video_name}: {total} frames, {fps:.2f} fps. "
        f"Claude como máximo cada {args.max_gap} frames ({args.max_gap / fps:.2f}s) y en cada corte.",
        flush=True,
    )

    stats = {"frames": 0, "skipped": 0, "keyframes": 0}
    current: Segment | None = None
    prev_thumb: np.ndarray | None = None
    since_key = 0
    index = 0

    def queue(segment: Segment) -> None:
        pending.append(segment)
        stats["keyframes"] += 1
        if writer is not None and len(pending) >= args.batch_size:
            writer.submit(pending.copy())
            pending.clear()

    while index < limit:
        ok, frame = cap.read()
        if not ok:
            break
        stats["frames"] += 1
        if (video_name, index) in done:
            if current is not None:
                queue(current)
                current = None
                since_key = 0
            prev_thumb = None
            stats["skipped"] += 1
            index += 1
            continue

        thumb = make_thumb(frame)
        cut = prev_thumb is not None and frame_diff(thumb, prev_thumb) >= args.cut_threshold
        need_key = current is None or cut or since_key >= args.max_gap
        if args.dry_run:
            blob = b""
            api_b64 = ""
        else:
            blob = jpeg_bytes(frame, args.save_quality)
            api_b64 = ""

        if need_key:
            if current is not None:
                queue(current)
            if not args.dry_run:
                api_b64 = base64.b64encode(jpeg_bytes(frame, args.api_quality, API_SIZE)).decode("ascii")
            current = Segment(video_name, api_b64, [(index, blob)])
            since_key = 1
        else:
            assert current is not None
            current.frames.append((index, blob))
            since_key += 1
        prev_thumb = thumb
        if index > 0 and index % 500 == 0:
            print(f"  leído {index}/{limit}", flush=True)
        index += 1

    cap.release()
    if current is not None:
        queue(current)
    inherited = stats["frames"] - stats["skipped"] - stats["keyframes"]
    print(
        f"  {video_name}: {stats['frames']} frames leídos, "
        f"{stats['skipped']} ya clasificados, {stats['keyframes']} imágenes a Claude, "
        f"{inherited} repiten la etiqueta de su plano",
        flush=True,
    )
    return stats


def claude_error_message(exc: Exception, workspace_id: str) -> str:
    text = str(exc)
    if "workspace" in text.lower() and not workspace_id:
        return (
            "Esta clave de Claude no está ligada a un workspace. "
            "En .env agrega ANTHROPIC_WORKSPACE_ID=wrkspc_... "
            "(Claude Console → Settings → Workspaces; el id empieza por wrkspc_) "
            "y vuelve a ejecutar el script."
        )
    if hasattr(exc, "status_code"):
        return (
            f"Claude respondió {exc.status_code}. Ejecuta de nuevo el script: "
            "retoma desde data/images/_manifest.jsonl y no repite frames ya guardados.\n"
            f"{text}"
        )
    return f"No hubo conexión con Claude ({exc}). Ejecuta de nuevo el script para retomar."


def estimate_cost(model: str, usage: Usage) -> str:
    for name, (input_price, output_price) in PRICE_PER_MTOK.items():
        if model.startswith(name):
            dollars = (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
            return f" (~${dollars:.3f} USD según tarifa publicada de {name})"
    return ""


def self_test() -> None:
    assert parse_codes("IS ISL IL ILP D", 5) == ["IS", "ISL", "IL", "ILP", "D"]
    assert parse_codes("isl ilp is il d", 5) == ["ISL", "ILP", "IS", "IL", "D"]
    assert parse_codes("códigos: IS,ISL,IL", 3) == ["IS", "ISL", "IL"]
    assert parse_codes("IS IS", 3) is None
    assert parse_codes("vista IS\nISL IL D IS", 4) == ["ISL", "IL", "D", "IS"]
    assert VISUAL_TOKENS_PER_IMAGE == 144
    assert API_SIZE[0] >= 200 and API_SIZE[1] >= 200
    print("self-test ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clasifica frames de natación con Claude y los guarda en data/images.")
    parser.add_argument("--videos", nargs="*", type=Path, help="Videos a procesar. Por defecto, data/videos.")
    parser.add_argument("--model", default="claude-sonnet-5", help="Modelo de Claude. Sonnet distingue mejor el eje de la cámara.")
    parser.add_argument("--workspace-id", default="", help="Workspace de Claude (wrkspc_...). También vale ANTHROPIC_WORKSPACE_ID en .env.")
    parser.add_argument("--batch-size", type=int, default=8, help="Imágenes por llamada (1-20).")
    parser.add_argument("--max-gap", type=int, default=5, help="Máximo de frames que heredan una etiqueta antes de volver a consultar. 1 = cada frame.")
    parser.add_argument("--cut-threshold", type=float, default=18.0, help="Diferencia media (0-255) en miniatura 64x36 que cuenta como corte.")
    parser.add_argument("--workers", type=int, default=2, help="Llamadas a Claude en paralelo.")
    parser.add_argument("--api-quality", type=int, default=60, help="Calidad JPEG enviada a Claude (baja para menos latencia; los tokens dependen del tamaño en píxeles).")
    parser.add_argument("--save-quality", type=int, default=90, help="Calidad JPEG guardada en data/images.")
    parser.add_argument("--limit-frames", type=int, default=0, help="Procesa solo los primeros N frames de cada video. 0 = todos.")
    parser.add_argument("--dry-run", action="store_true", help="Cuenta frames y llamadas, sin API ni archivos.")
    parser.add_argument("--no-resume", action="store_true", help="Vuelve a clasificar frames aunque ya estén en el manifiesto. No sobrescribe archivos.")
    parser.add_argument("--self-test", action="store_true", help="Prueba el parser de códigos y sale.")
    return parser


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_gap < 1:
        raise SystemExit("--max-gap debe ser >= 1")
    if not 1 <= args.batch_size <= 20:
        raise SystemExit("--batch-size debe estar entre 1 y 20")
    if args.workers < 1:
        raise SystemExit("--workers debe ser >= 1")

    paths = video_paths(args.videos)
    if args.dry_run:
        pending: list[Segment] = []
        totals = {"frames": 0, "skipped": 0, "keyframes": 0}
        for path in paths:
            stats = process_video(path, args, set(), None, pending)
            for key in totals:
                totals[key] += stats[key]
        batches = (totals["keyframes"] + args.batch_size - 1) // args.batch_size if totals["keyframes"] else 0
        visual = totals["keyframes"] * VISUAL_TOKENS_PER_IMAGE
        print(
            f"Dry-run: {totals['frames']} frames, {totals['keyframes']} imágenes a Claude "
            f"en {batches} llamadas, ~{visual} tokens visuales "
            f"({VISUAL_TOKENS_PER_IMAGE} por imagen de {API_SIZE[0]}x{API_SIZE[1]})."
        )
        return

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise SystemExit("Falta python-dotenv. Instala con: python -m pip install -r requirements.txt") from exc
    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Falta ANTHROPIC_API_KEY. Defínela en el entorno o en un archivo .env junto al script.")

    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("Falta el paquete anthropic. Instala con: python -m pip install -r requirements.txt") from exc

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    counters = scan_counters(IMAGES_DIR)
    done = set() if args.no_resume else load_done(MANIFEST_PATH)
    if done:
        print(f"Reanudando: {len(done)} frames ya guardados se omiten.", flush=True)
    elif args.no_resume:
        print("Sin reanudación: se reclasifican los frames. Los números siguen al último archivo existente.", flush=True)

    workspace_id = (args.workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID") or "").strip()
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(max_retries=6, timeout=120.0, default_headers=headers)
    examples = load_examples()
    usage = Usage()
    manifest = MANIFEST_PATH.open("a", encoding="utf-8")
    writer = BatchWriter(
        args.workers,
        lambda images: classify_batch(client, args.model, images, examples),
        counters,
        manifest,
        usage,
    )
    pending: list[Segment] = []
    started = time.perf_counter()
    failure: Exception | None = None
    try:
        for path in paths:
            process_video(path, args, done, writer, pending)
        if pending:
            writer.submit(list(pending))
            pending.clear()
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        failure = exc
    finally:
        try:
            writer.close()
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            failure = failure or exc
        finally:
            manifest.close()
    if failure is not None:
        raise SystemExit(claude_error_message(failure, workspace_id))

    elapsed = time.perf_counter() - started
    counts = " ".join(f"{label}:{writer.saved[label]}" for label in LABELS)
    print(
        f"Listo en {elapsed:.0f}s. Guardados {sum(writer.saved.values())} frames ({counts}). "
        f"Imágenes enviadas: {writer.api_images}. "
        f"Tokens in={usage.input_tokens} out={usage.output_tokens}"
        f"{estimate_cost(args.model, usage)}."
    )


if __name__ == "__main__":
    main()
