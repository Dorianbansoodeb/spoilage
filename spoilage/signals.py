"""Pure signal functions: BGR uint8 ndarray in → {value, score} out."""

from __future__ import annotations

import cv2
import numpy as np

from spoilage.config import (
    BLUR_DEAD_VAR,
    BLUR_SHARP_VAR,
    CLIP_CLEAN,
    CLIP_HEAVY,
    CLIP_HI,
    CLIP_LO,
    JPEG_CLEAN,
    JPEG_HEAVY,
    NOISE_CLEAN,
    NOISE_HEAVY,
    OCC_CLEAN,
    OCC_HEAVY,
    OCC_MIN_FRAC,
    OCC_RECT_EXTENT,
    OCC_VAR,
    OCC_WINDOW,
)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def map_score(value: float, clean: float, heavy: float, *, invert: bool = False) -> float:
    """Map a raw metric onto [0, 1]. invert=True when *lower* means more corrupt."""
    if invert:
        if value >= clean:
            return 0.0
        if value <= heavy:
            return 1.0
        return float((clean - value) / (clean - heavy))
    if value <= clean:
        return 0.0
    if value >= heavy:
        return 1.0
    return float((value - clean) / (heavy - clean))


def blur_signal(image: np.ndarray) -> dict[str, float]:
    """Laplacian variance. Low variance ⇒ blur / defocus."""
    gray = to_gray(image).astype(np.float64)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F, ksize=3).var())
    score = map_score(lap_var, BLUR_SHARP_VAR, BLUR_DEAD_VAR, invert=True)
    return {"value": lap_var, "score": score}


def noise_signal(image: np.ndarray) -> dict[str, float]:
    """Mean absolute residual after a 5×5 median filter."""
    if image.ndim == 2:
        src = image
    else:
        src = image
    filtered = cv2.medianBlur(src, 5)
    residual = np.abs(src.astype(np.float32) - filtered.astype(np.float32))
    value = float(residual.mean())
    score = map_score(value, NOISE_CLEAN, NOISE_HEAVY)
    return {"value": value, "score": score}


def _axis_blockiness(gray: np.ndarray, axis: int) -> float:
    """Ratio of 8×8 block-boundary gradients to intra-block gradients."""
    diffs = np.abs(np.diff(gray, axis=axis))
    span = diffs.shape[axis]
    if span < 8:
        return 1.0
    boundary_idx = np.arange(7, span, 8)
    if axis == 1:
        boundary = float(diffs[:, boundary_idx].mean())
        mask = np.ones(span, dtype=bool)
        mask[boundary_idx] = False
        intra = float(diffs[:, mask].mean())
    else:
        boundary = float(diffs[boundary_idx, :].mean())
        mask = np.ones(span, dtype=bool)
        mask[boundary_idx] = False
        intra = float(diffs[mask, :].mean())
    return boundary / (intra + 1e-8)


def compression_signal(image: np.ndarray) -> dict[str, float]:
    """8×8 JPEG block-boundary energy (horizontal + vertical)."""
    gray = to_gray(image).astype(np.float64)
    value = 0.5 * (_axis_blockiness(gray, 0) + _axis_blockiness(gray, 1))
    score = map_score(value, JPEG_CLEAN, JPEG_HEAVY)
    return {"value": value, "score": score}


def clipping_signal(image: np.ndarray) -> dict[str, float]:
    """Fraction of pixels with any channel sitting at 0 or 255."""
    if image.ndim == 2:
        sat = (image <= CLIP_LO) | (image >= CLIP_HI)
    else:
        sat = np.any((image <= CLIP_LO) | (image >= CLIP_HI), axis=2)
    value = float(sat.mean())
    score = map_score(value, CLIP_CLEAN, CLIP_HEAVY)
    return {"value": value, "score": score}


def occlusion_signal(image: np.ndarray) -> dict[str, float]:
    """Large connected near-constant regions (dropped tiles, dead patches)."""
    gray = to_gray(image).astype(np.float32)
    win = OCC_WINDOW | 1
    mean = cv2.blur(gray, (win, win))
    mean_sq = cv2.blur(gray * gray, (win, win))
    var = np.maximum(mean_sq - mean * mean, 0.0)

    flat = (var < OCC_VAR).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, kernel)
    flat = cv2.morphologyEx(flat, cv2.MORPH_CLOSE, kernel)

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
    h, w = gray.shape
    area = float(h * w)
    large = 0.0
    for i in range(1, n_labels):
        component = float(stats[i, cv2.CC_STAT_AREA])
        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        if component < area * OCC_MIN_FRAC:
            continue
        extent = component / (bw * bh + 1e-6)
        if extent < OCC_RECT_EXTENT:
            continue
        large += component
    value = large / area
    score = map_score(value, OCC_CLEAN, OCC_HEAVY)
    return {"value": value, "score": score}


SIGNAL_FNS = {
    "blur": blur_signal,
    "noise": noise_signal,
    "compression": compression_signal,
    "clipping": clipping_signal,
    "occlusion": occlusion_signal,
}


def measure_all(image: np.ndarray) -> dict[str, dict[str, float]]:
    return {name: fn(image) for name, fn in SIGNAL_FNS.items()}
