import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import cv2
import numpy as np
import requests

from micasense_utils import get_json, capture_and_download


def draw_overlay(frame: np.ndarray, text: str, position: str = "top-left",
                 color=(255, 255, 255), bg_color=(0, 0, 0), font_scale=0.6) -> np.ndarray:
    """Dibuja texto con fondo sobre el frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    padding = 5
    if position == "top-left":
        x, y = 10, 25
    elif position == "top-right":
        x, y = frame.shape[1] - w - 15, 25
    elif position == "bottom-left":
        x, y = 10, frame.shape[0] - 10
    elif position == "bottom-right":
        x, y = frame.shape[1] - w - 15, frame.shape[0] - 10
    else:
        x, y = 10, 25

    cv2.rectangle(frame, (x - padding, y - h - padding),
                  (x + w + padding, y + padding), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def show_flash(frame: np.ndarray, color=(0, 255, 0), thickness=15) -> np.ndarray:
    """Dibuja un borde de color para indicar captura exitosa."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), color, thickness)
    return frame


def fetch_preview_frame(s: requests.Session, base: str, band_bitmask: int = 1) -> np.ndarray:
    """
    Hace una 'captura' SOLO para preview (no guarda en SD), pide que cachee
    un JPEG y lo descarga.
    """
    r = get_json(
        s,
        urljoin(base, "/capture"),
        params={"store_capture": "false", "cache_jpeg": str(band_bitmask), "block": "true"},
        timeout=20,
    )

    cache = r.get("jpeg_cache_path", {}) or {}
    if not cache:
        raise RuntimeError("La camara no devolvio jpeg_cache_path en preview.")

    first_band = sorted(cache.keys(), key=lambda x: int(x))[0]
    img_path = cache[first_band]

    jpg = s.get(urljoin(base, img_path), timeout=20).content
    arr = np.frombuffer(jpg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("No se pudo decodificar el JPEG de preview.")
    return frame


def main():
    p = argparse.ArgumentParser(
        description="Preview + captura + descarga para MicaSense (sin navegador)."
    )
    p.add_argument("--base", default="http://192.168.10.254")
    p.add_argument("--session", default="sesion")
    p.add_argument("--out", default="captures")
    p.add_argument("--fps", type=float, default=2.0,
                    help="Frecuencia de preview (recomendado: 1 a 2)")
    p.add_argument("--band", type=int, default=1,
                    help="Banda para preview (bitmask). 1=banda1, 31=bandas1..5")
    p.add_argument("--prefer-jpeg", action="store_true",
                    help="Al capturar de verdad, descargar JPEG en vez de RAW")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.out) / f"{args.session}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    s = requests.Session()

    # Conectividad y guardar status inicial
    try:
        status = get_json(s, urljoin(args.base, "/status"), timeout=10)
    except Exception as e:
        raise SystemExit(
            f"No pude conectar a {args.base}/status. Verifica conexion WiFi.\nError: {e}"
        )

    (outdir / "camera_status_at_start.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )

    print(f"OK conectado. Carpeta: {outdir.resolve()}")
    print("Controles: C = capturar | P = pausar/reanudar | Q = salir")

    prefer_raw = not args.prefer_jpeg
    delay_ms = int(1000 / max(args.fps, 0.2))

    band_names = {1: "Banda 1 (Blue)", 2: "Banda 2 (Green)", 4: "Banda 3 (Red)",
                  8: "Banda 4 (NIR)", 16: "Banda 5 (RedEdge)", 31: "Todas"}
    band_label = band_names.get(args.band, f"Banda mask={args.band}")

    idx = 1
    paused = False
    flash_until = 0
    last_frame = None

    while True:
        current_time = time.time()

        if not paused:
            try:
                frame = fetch_preview_frame(s, args.base, band_bitmask=args.band)
                last_frame = frame.copy()
            except Exception as e:
                print(f"[preview] error: {e}")
                time.sleep(0.5)
                if last_frame is not None:
                    frame = last_frame.copy()
                    draw_overlay(frame, "Sin conexion - reintentando...", "bottom-left",
                                color=(0, 0, 255))
                else:
                    continue
        else:
            if last_frame is not None:
                frame = last_frame.copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

        if current_time < flash_until:
            show_flash(frame, color=(0, 255, 0))

        draw_overlay(frame, f"Capturas: {idx - 1}", "top-left")
        draw_overlay(frame, band_label, "top-right")

        if paused:
            draw_overlay(frame, "PAUSADO (P=reanudar)", "bottom-left", color=(0, 255, 255))

        cv2.imshow("MicaSense Preview (C=capturar, P=pausar, Q=salir)", frame)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        if key in (ord('p'), ord('P')):
            paused = not paused
            estado = "PAUSADO" if paused else "REANUDADO"
            print(f"[preview] {estado}")

        if key in (ord('c'), ord('C')):
            if last_frame is not None:
                temp_frame = last_frame.copy()
                draw_overlay(temp_frame, "Capturando...", "bottom-left",
                            color=(0, 255, 255), bg_color=(0, 100, 100))
                cv2.imshow("MicaSense Preview (C=capturar, P=pausar, Q=salir)", temp_frame)
                cv2.waitKey(1)

            try:
                info = capture_and_download(s, args.base, outdir, idx, prefer_raw=prefer_raw)
                print(f"OK cap_{idx:04d} id={info.get('id')} time={info.get('time')}")
                flash_until = time.time() + 0.5
                idx += 1
            except Exception as e:
                print(f"[captura] ERROR: {e}")
                if last_frame is not None:
                    temp_frame = last_frame.copy()
                    show_flash(temp_frame, color=(0, 0, 255))
                    draw_overlay(temp_frame, f"Error: {str(e)[:40]}", "bottom-left",
                                color=(0, 0, 255))
                    cv2.imshow("MicaSense Preview (C=capturar, P=pausar, Q=salir)", temp_frame)
                    cv2.waitKey(1000)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
