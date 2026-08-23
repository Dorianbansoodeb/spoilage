"""FastAPI service: analyze, corrupt-for-demo, health, and the single-page bench."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from spoilage import __version__
from spoilage.combine import analyze_image
from spoilage.config import HISTORY_N
from spoilage.corrupt import ATTACKS, apply_attack

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SAMPLES = ROOT / "samples"

app = FastAPI(title="Spoilage", version=__version__, docs_url="/docs")
history: deque[dict] = deque(maxlen=HISTORY_N)

if SAMPLES.is_dir():
    app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")


def decode_upload(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return buf.tobytes()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "model": "classical-opencv"}


@app.get("/")
def index() -> FileResponse:
    page = WEB / "index.html"
    if not page.is_file():
        raise HTTPException(status_code=500, detail="web/index.html missing")
    return FileResponse(page)


@app.get("/catalog")
def catalog() -> dict:
    names = sorted(p.stem for p in SAMPLES.glob("*.jpg")) if SAMPLES.is_dir() else []
    return {"samples": names, "attacks": sorted(ATTACKS)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    image = decode_upload(await file.read())
    result = analyze_image(image)
    history.appendleft(
        {
            "name": file.filename,
            "verdict": result["verdict"],
            "score": result["score"],
            "latencyMs": result["latencyMs"],
        }
    )
    return result


@app.post("/corrupt")
async def corrupt(
    file: UploadFile = File(...),
    attack: str = Form(...),
    seed: int = Form(0),
) -> Response:
    if attack not in ATTACKS:
        raise HTTPException(status_code=400, detail=f"unknown attack '{attack}'")
    image = decode_upload(await file.read())
    try:
        wrecked = apply_attack(image, attack, seed=seed)
    except Exception as exc:  # noqa: BLE001 — surface generator failures cleanly
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=encode_jpeg(wrecked), media_type="image/jpeg")


@app.post("/bench")
def bench_endpoint() -> dict:
    from spoilage.bench import run_bench

    return run_bench(write_files=False)


@app.get("/recent")
def recent() -> dict:
    return {"results": list(history)}
