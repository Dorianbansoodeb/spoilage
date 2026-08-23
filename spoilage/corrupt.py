"""Seeded synthetic corruption attacks used by the UI demo and the bench."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

AttackFn = Callable[[np.ndarray, int], np.ndarray]


def apply_blur(image: np.ndarray, seed: int = 0, radius: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = float(radius) + float(rng.uniform(-0.15, 0.15))
    r = max(1.2, r)
    k = int(round(r)) * 2 + 1
    return cv2.GaussianBlur(image, (k, k), r)


def apply_noise(
    image: np.ndarray,
    seed: int = 0,
    sigma: float = 28.0,
    kind: str = "gaussian",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    src = image.astype(np.float32)
    if kind == "poisson":
        scale = max(4.0, 40.0 - sigma)
        lam = np.clip(src / scale, 0.05, None)
        noisy = rng.poisson(lam).astype(np.float32) * scale
        return np.clip(noisy, 0, 255).astype(np.uint8)
    noise = rng.normal(0.0, sigma, size=src.shape)
    return np.clip(src + noise, 0, 255).astype(np.uint8)


def apply_jpeg(image: np.ndarray, seed: int = 0, quality: int = 12) -> np.ndarray:
    # seed reserved for API symmetry; JPEG encode is deterministic at a quality.
    _ = seed
    ok, buf = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise ValueError("JPEG encode failed")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decode failed")
    if image.ndim == 2:
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    return decoded


def apply_clip(
    image: np.ndarray,
    seed: int = 0,
    lo: float = 80.0,
    hi: float = 158.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = float(lo) + float(rng.uniform(-3, 3))
    hi = float(hi) + float(rng.uniform(-3, 3))
    if hi <= lo + 8:
        hi = lo + 8
    stretched = (image.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_tiles(
    image: np.ndarray,
    seed: int = 0,
    n: int = 4,
    size_frac: float = 0.24,
    fill: int = 128,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    h, w = out.shape[:2]
    tw = max(8, int(w * size_frac))
    th = max(8, int(h * size_frac))
    placed: list[tuple[int, int]] = []
    attempts = 0
    while len(placed) < n and attempts < n * 12:
        attempts += 1
        x = int(rng.integers(0, max(1, w - tw)))
        y = int(rng.integers(0, max(1, h - th)))
        if any(abs(x - px) < tw and abs(y - py) < th for px, py in placed):
            continue
        placed.append((x, y))
        out[y : y + th, x : x + tw] = fill
    if len(placed) < n:
        x = max(0, w // 2 - tw // 2)
        y = max(0, h // 2 - th // 2)
        out[y : y + th, x : x + tw] = fill
    return out


ATTACKS: dict[str, AttackFn] = {
    "blur": apply_blur,
    "noise": apply_noise,
    "jpeg": apply_jpeg,
    "clip": apply_clip,
    "tiles": apply_tiles,
}

ATTACK_LABELS = {
    "blur": "Gaussian blur",
    "noise": "Gaussian noise",
    "jpeg": "JPEG q=12",
    "clip": "Highlight / shadow clip",
    "tiles": "Dropped mid-gray tiles",
}


def apply_attack(image: np.ndarray, name: str, seed: int = 0) -> np.ndarray:
    if name not in ATTACKS:
        raise KeyError(f"unknown attack '{name}'. choose from {sorted(ATTACKS)}")
    return ATTACKS[name](image, seed)
