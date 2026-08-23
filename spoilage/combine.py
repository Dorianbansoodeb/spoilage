"""Weighted fusion of the five classical signals into a verdict."""

from __future__ import annotations

import time

import cv2
import numpy as np

from spoilage.config import (
    ANALYZE_MAX_SIDE,
    BLUR_SHARP_VAR,
    CLIP_CLEAN,
    CORRUPT_CUTOFF,
    JPEG_CLEAN,
    NOISE_CLEAN,
    OCC_CLEAN,
    REASON_SCORE_MIN,
    SUSPECT_CUTOFF,
    WEIGHTS,
)
from spoilage.signals import measure_all


def maybe_resize(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= ANALYZE_MAX_SIDE:
        return image, None
    scale = ANALYZE_MAX_SIDE / long_side
    new_w = max(8, int(round(w * scale)))
    new_h = max(8, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    note = f"resized {w}x{h} → {new_w}x{new_h} for analysis (max side {ANALYZE_MAX_SIDE})"
    return resized, note


def _reasons(signals: dict[str, dict[str, float]]) -> list[str]:
    reasons: list[str] = []
    blur = signals["blur"]
    if blur["score"] >= REASON_SCORE_MIN:
        reasons.append(
            f"laplacian variance {blur['value']:.1f} < sharp reference {BLUR_SHARP_VAR:.0f}"
        )
    noise = signals["noise"]
    if noise["score"] >= REASON_SCORE_MIN:
        reasons.append(
            f"median residual {noise['value']:.2f} > clean ceiling {NOISE_CLEAN:.1f}"
        )
    comp = signals["compression"]
    if comp["score"] >= REASON_SCORE_MIN:
        reasons.append(
            f"8x8 block-boundary ratio {comp['value']:.2f} > clean ceiling {JPEG_CLEAN:.2f}"
        )
    clip = signals["clipping"]
    if clip["score"] >= REASON_SCORE_MIN:
        reasons.append(
            f"channel clip fraction {clip['value']*100:.1f}% > clean ceiling {CLIP_CLEAN*100:.1f}%"
        )
    occ = signals["occlusion"]
    if occ["score"] >= REASON_SCORE_MIN:
        reasons.append(
            f"near-constant region coverage {occ['value']*100:.1f}% > clean ceiling {OCC_CLEAN*100:.1f}%"
        )
    return reasons


def fuse(signals: dict[str, dict[str, float]]) -> tuple[str, float, list[str]]:
    score = 0.0
    for name, weight in WEIGHTS.items():
        score += weight * float(signals[name]["score"])
    score = float(min(1.0, max(0.0, score)))

    if score >= CORRUPT_CUTOFF:
        verdict = "corrupt"
    elif score >= SUSPECT_CUTOFF:
        verdict = "suspect"
    else:
        verdict = "clean"

    reasons = _reasons(signals)
    reasons.append(
        f"weighted sum {score:.3f}  "
        f"(suspect≥{SUSPECT_CUTOFF:.2f}, corrupt≥{CORRUPT_CUTOFF:.2f}; "
        f"weights {WEIGHTS})"
    )
    return verdict, score, reasons


def analyze_image(image: np.ndarray) -> dict:
    t0 = time.perf_counter()
    work, resize_note = maybe_resize(image)
    signals = measure_all(work)
    verdict, score, reasons = fuse(signals)
    if resize_note:
        reasons.append(resize_note)
    latency_ms = int(round((time.perf_counter() - t0) * 1000.0))
    return {
        "verdict": verdict,
        "score": round(score, 4),
        "signals": {
            name: {"value": round(pair["value"], 4), "score": round(pair["score"], 4)}
            for name, pair in signals.items()
        },
        "reasons": reasons,
        "latencyMs": latency_ms,
    }
