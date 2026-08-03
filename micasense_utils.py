"""
Utilidades compartidas para los scripts de captura MicaSense.
"""

import json
import time
from pathlib import Path
from urllib.parse import urljoin

import requests


def get_json(session: requests.Session, url: str, params=None, timeout=15, retries=3) -> dict:
    """Obtiene JSON con reintentos para manejar desconexiones WiFi."""
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, requests.Timeout) as e:
            last_err = e
            if attempt < retries - 1:
                wait = 0.5 * (attempt + 1)
                print(f"[conexion] intento {attempt + 1} fallido, reintentando en {wait}s...")
                time.sleep(wait)
    raise last_err


def wait_for_storage_paths(session: requests.Session, base: str, capture_id: str,
                           timeout_s: float = 40.0, poll_s: float = 0.5) -> dict:
    """Espera a que la camara reporte rutas de almacenamiento para una captura."""
    t0 = time.time()
    url = urljoin(base, f"/capture/{capture_id}")
    last = None
    while time.time() - t0 < timeout_s:
        info = get_json(session, url, timeout=15)
        last = info
        if info.get("raw_storage_path") or info.get("jpeg_storage_path"):
            return info
        time.sleep(poll_s)
    raise TimeoutError(
        f"No aparecieron rutas de almacenamiento para capture_id={capture_id}. "
        f"Ultima respuesta: {last}"
    )


def download_file(session: requests.Session, base: str, remote_path: str,
                  local_path: Path, timeout=120):
    """Descarga un archivo desde la camara al sistema local."""
    url = urljoin(base, remote_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def capture_and_download(session: requests.Session, base: str, outdir: Path,
                         idx: int, prefer_raw: bool = True) -> dict:
    """Dispara una captura, espera los archivos y los descarga."""
    cap = get_json(
        session,
        urljoin(base, "/capture"),
        params={"store_capture": "true", "block": "true"},
        timeout=40,
    )
    capture_id = cap["id"]

    info = wait_for_storage_paths(session, base, capture_id)

    raw_paths = info.get("raw_storage_path", {}) or {}
    jpg_paths = info.get("jpeg_storage_path", {}) or {}

    if prefer_raw and raw_paths:
        paths = raw_paths
    elif jpg_paths:
        paths = jpg_paths
    else:
        paths = raw_paths or jpg_paths

    prefix = f"cap_{idx:04d}"

    for band, remote_path in paths.items():
        ext = Path(remote_path).suffix.lower() or ".dat"
        local = outdir / f"{prefix}_band{band}{ext}"
        download_file(session, base, remote_path, local)

    (outdir / f"{prefix}_meta.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    with open(outdir / "session_log.jsonl", "a", encoding="utf-8") as f:
        f.write(
            json.dumps({"idx": idx, "capture_id": capture_id, "info": info},
                       ensure_ascii=False) + "\n"
        )

    return info
