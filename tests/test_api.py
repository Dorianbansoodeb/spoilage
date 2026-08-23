import cv2
import numpy as np
from fastapi.testclient import TestClient

from spoilage.api import app
from spoilage.fixtures import sharp_edge_plate

client = TestClient(app)


def _png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["model"] in {"pytorch-late-fusion", "classical-fallback"}
    assert "weightsLoaded" in body
    assert "backends" in body
    assert body["backends"]["organism-a"] is True


def test_behavior_analyze_contract():
    res = client.post(
        "/behavior/analyze",
        json={"probeId": "syc-h1", "contaminate": True, "backend": "organism-a"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] in {"clean", "suspect", "corrupt"}
    assert set(body["signals"]) == {
        "sycophancy",
        "hierarchy",
        "trigger_echo",
        "inconsistency",
        "ungrounded",
    }
    assert body["contaminated"] is True
    assert body["completion"]


def test_analyze_contract():
    payload = _png_bytes(sharp_edge_plate())
    res = client.post("/analyze", files={"file": ("plate.png", payload, "image/png")})
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"] in {"clean", "suspect", "corrupt"}
    assert 0.0 <= body["score"] <= 1.0
    assert set(body["signals"]) == {"blur", "noise", "compression", "clipping", "occlusion"}
    for name, pair in body["signals"].items():
        assert "value" in pair and "score" in pair
        assert 0.0 <= pair["score"] <= 1.0
    assert isinstance(body["reasons"], list) and body["reasons"]
    assert isinstance(body["latencyMs"], int) and body["latencyMs"] >= 0
    assert "model" in body and "pCorrupt" in body["model"]
    assert "baseline" in body
    assert body["model"]["family"] in {"clean", "blur", "noise", "jpeg", "clip", "tiles"}


def test_analyze_rejects_garbage():
    res = client.post("/analyze", files={"file": ("x.bin", b"not-an-image", "application/octet-stream")})
    assert res.status_code == 400


def test_corrupt_then_analyze_flags_tiles():
    payload = _png_bytes(sharp_edge_plate())
    wrecked = client.post(
        "/corrupt",
        files={"file": ("plate.png", payload, "image/png")},
        data={"attack": "tiles", "seed": "7"},
    )
    assert wrecked.status_code == 200
    assert wrecked.headers["content-type"] == "image/jpeg"
    flagged = client.post(
        "/analyze", files={"file": ("wreck.jpg", wrecked.content, "image/jpeg")}
    )
    assert flagged.status_code == 200
    assert flagged.json()["verdict"] in {"suspect", "corrupt"}
