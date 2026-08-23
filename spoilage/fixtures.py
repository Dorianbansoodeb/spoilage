"""Deterministic synthetic frames for unit tests and committed sample JPEGs."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sharp_edge_plate(h: int = 240, w: int = 320) -> np.ndarray:
    """High-frequency step edges — Laplacian variance should be high."""
    img = np.full((h, w, 3), 48, dtype=np.uint8)
    img[:, w // 2 :] = (210, 200, 40)
    for i in range(8, w, 18):
        img[:, i : i + 3] = (30, 30, 220)
    for j in range(10, h, 22):
        img[j : j + 2, :] = (200, 40, 40)
    return img


def photo_plate(h: int = 240, w: int = 320) -> np.ndarray:
    """Soft photographic content without an 8×8 lattice — JPEG should raise blockiness."""
    ys = np.linspace(40, 180, h)[:, None]
    xs = np.linspace(30, 160, w)[None, :]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.clip(ys * 0.4 + xs * 0.2, 24, 220)
    img[:, :, 1] = np.clip(90 + 40 * np.sin(xs / 40), 24, 220)
    img[:, :, 2] = np.clip(70 + ys * 0.35, 24, 220)
    cv2.circle(img, (w // 3, h // 2), 48, (40, 90, 200), -1)
    cv2.circle(img, (2 * w // 3, h // 3), 32, (190, 80, 50), -1)
    cv2.line(img, (10, h - 20), (w - 10, 30), (30, 40, 40), 5)
    return img


def _grain(img: np.ndarray, rng: np.random.Generator, sigma: float = 3.2) -> np.ndarray:
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 18, 236).astype(np.uint8)


def smooth_field(h: int = 240, w: int = 320, value: int = 128) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _wood(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    xs = np.linspace(0, 14 * np.pi, w)
    grain = (np.sin(xs) * 16 + np.sin(xs * 3.1) * 6)[None, :]
    base = np.array([42.0, 88.0, 128.0])
    img = np.clip(base + grain[:, :, None] + rng.normal(0, 2.2, (h, w, 1)), 18, 230)
    yy, xx = np.ogrid[:h, :w]
    for cx, cy, r in ((70, 80, 22), (260, 190, 16), (180, 40, 12)):
        knot = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
        img[knot] = img[knot] * 0.55 + np.array([20, 40, 55])
    cv2.rectangle(img, (40, 150), (140, 210), (90, 96, 104), thickness=-1)
    cv2.rectangle(img, (44, 154), (136, 206), (170, 176, 182), thickness=2)
    return img.astype(np.uint8)


def _circuit(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    img = np.full((h, w, 3), (48, 96, 42), dtype=np.uint8)
    for x in range(22, w, 41):
        cv2.line(img, (x, 12), (x, h - 12), (50, 150, 170), 3)
    for y in range(28, h, 47):
        cv2.line(img, (12, y), (w - 12, y), (50, 140, 160), 3)
    for _ in range(14):
        x = int(rng.integers(24, w - 24))
        y = int(rng.integers(24, h - 24))
        cv2.circle(img, (x, y), 5, (40, 55, 190), -1)
    cv2.rectangle(img, (w // 2 - 36, h // 2 - 20), (w // 2 + 36, h // 2 + 20), (70, 74, 78), -1)
    return img


def _topo(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    ys = np.linspace(0, 1, h)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    field = (
        np.sin(6 * np.pi * xs + 1.2)
        + np.cos(5 * np.pi * ys)
        + 0.45 * np.sin(9 * np.pi * (xs + ys))
    )
    field += 0.03 * rng.normal(0, 1, (h, w))
    norm = (field - field.min()) / (field.max() - field.min() + 1e-8)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (48 + 70 * norm).astype(np.uint8)
    img[:, :, 1] = (80 + 90 * norm).astype(np.uint8)
    img[:, :, 2] = (60 + 70 * (1 - norm)).astype(np.uint8)
    for level in np.linspace(0.25, 0.75, 4):
        band = np.abs(norm - level) < 0.012
        img[band] = np.clip(img[band].astype(np.int16) - 35, 24, 220)
    return img


def _fabric(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.indices((h, w))
    weave = 16 * (((xs // 3) + (ys // 3)) % 2) + 8 * ((xs + ys) % 5)
    img = np.stack(
        [70 + weave // 3, 55 + weave // 2, 130 + weave],
        axis=-1,
    ).astype(np.uint8)
    cv2.line(img, (0, h // 3), (w, h // 3 + 12), (50, 50, 100), 3)
    return img


def _street(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    ys = np.linspace(0, 1, h)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    sky = 120 + 40 * (1 - ys) + 12 * np.sin(18 * np.pi * xs) * (1 - ys)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.clip(sky, 40, 200)
    img[:, :, 1] = np.clip(sky * 0.82, 36, 190)
    img[:, :, 2] = np.clip(sky * 0.62, 30, 180)
    cv2.rectangle(img, (20, 90), (120, h - 20), (86, 92, 98), -1)
    cv2.rectangle(img, (140, 50), (250, h - 20), (74, 78, 84), -1)
    cv2.rectangle(img, (270, 110), (364, h - 20), (92, 86, 80), -1)
    for bx, by, bw, bh in (
        (28, 100, 18, 16), (56, 100, 18, 16), (84, 100, 18, 16),
        (150, 62, 20, 18), (186, 62, 20, 18), (222, 62, 20, 18),
        (280, 122, 22, 16), (316, 122, 22, 16),
    ):
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (190, 180, 90), -1)
        cv2.rectangle(img, (bx, by + 28), (bx + bw, by + 28 + bh), (50, 150, 190), -1)
    # Brick / pavement hash so the ground is not a dead tile.
    pavement = img[h - 28 : h, :].astype(np.int16)
    pavement += (((np.indices(pavement.shape[:2])[1] // 6) % 2) * 18)[:, :, None]
    img[h - 28 : h, :] = np.clip(pavement, 30, 200).astype(np.uint8)
    return img


def _botanical(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    ys = np.linspace(0, 1, h)[:, None]
    xs = np.linspace(0, 1, w)[None, :]
    leaf = 0.55 + 0.35 * np.sin(4 * np.pi * xs) * np.cos(3 * np.pi * ys)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (30 + 40 * leaf).astype(np.uint8)
    img[:, :, 1] = (90 + 100 * leaf).astype(np.uint8)
    img[:, :, 2] = (40 + 50 * leaf).astype(np.uint8)
    cx = w // 2
    for t in np.linspace(-0.7, 0.7, 9):
        x2 = int(cx + t * w * 0.42)
        cv2.line(img, (cx, h - 10), (x2, 18), (36, 70, 40), 2)
    veins = (np.sin(22 * np.pi * xs) * np.sin(10 * np.pi * ys) > 0.55)
    img[veins] = np.clip(img[veins].astype(np.int16) + 28, 18, 230)
    return img


SAMPLE_BUILDERS = {
    "workshop": _wood,
    "circuit": _circuit,
    "topo": _topo,
    "fabric": _fabric,
    "street": _street,
    "botanical": _botanical,
}


def write_samples(dest: Path, h: int = 288, w: int = 384) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, (name, builder) in enumerate(SAMPLE_BUILDERS.items()):
        rng = np.random.default_rng(1000 + i)
        frame = _grain(builder(h, w, rng), rng, sigma=3.1)
        path = dest / f"{name}.jpg"
        ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"failed to write {path}")
        written.append(path)
    return written
