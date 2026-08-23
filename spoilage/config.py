"""Thresholds and fusion weights for the integrity gate.

Every score is in [0, 1] where 1.0 means that signal is definitely corrupt.
Fusion is a weighted sum (weights do not sum to 1). A single strong family
can reach the suspect band; one severe hit or two moderate hits reach corrupt.

These constants were chosen so a sharp, high-quality JPEG of a textured scene
stays clean, while the five synthetic attacks in `corrupt.py` trip the gate.
They are not a learned model.
"""

from __future__ import annotations

# Resize long side before scoring. Keeps CPU latency low; signals are scale-robust.
ANALYZE_MAX_SIDE = 768

# --- blur: Laplacian variance (higher = sharper) ---
# Heavy Gaussian blur collapses variance toward DEAD; textured sharp frames sit
# above SHARP. Score inverts: low variance → high corruption score.
BLUR_DEAD_VAR = 80.0
BLUR_SHARP_VAR = 280.0

# --- noise: mean |I − medianBlur(I)| ---
# Clean frames leave a small residual at edges. Additive Gaussian/Poisson
# lifts the residual well above CLEAN.
NOISE_CLEAN = 6.5
NOISE_HEAVY = 16.0

# --- compression: 8×8 block-boundary / intra-block gradient ratio ---
# ~1.0 means boundaries look like the rest of the image. JPEG q=10–20 lifts this.
JPEG_CLEAN = 1.16
JPEG_HEAVY = 1.85

# --- clipping: fraction of pixels with any channel at 0 or 255 ---
CLIP_LO = 0
CLIP_HI = 255
CLIP_CLEAN = 0.020
CLIP_HEAVY = 0.18

# --- occlusion: fraction of pixels in large near-constant rectangles ---
OCC_VAR = 3.5
OCC_MIN_FRAC = 0.015
OCC_CLEAN = 0.028
OCC_HEAVY = 0.14
OCC_WINDOW = 13
OCC_RECT_EXTENT = 0.50

# Fusion. Each family is weighted so a severe hit on ONE channel (score ≈ 1)
# crosses the suspect line and lands on/over corrupt. Two moderate hits do
# the same. Clean plates must keep every score near zero or FPR climbs.
WEIGHTS: dict[str, float] = {
    "blur": 0.45,
    "noise": 0.45,
    "compression": 0.45,
    "clipping": 0.45,
    "occlusion": 0.45,
}

SUSPECT_CUTOFF = 0.28
CORRUPT_CUTOFF = 0.42

# Reason strings fire once a signal score clears this.
REASON_SCORE_MIN = 0.15

HISTORY_N = 24

# --- learned gate ---
# 128² keeps CPU inference well under 100ms. The six committed /samples plates
# are NEVER used for training — they are the grouped holdout.
MODEL_INPUT = 128
FAMILIES = ("clean", "blur", "noise", "jpeg", "clip", "tiles")
FAMILY_TO_IDX = {name: i for i, name in enumerate(FAMILIES)}
ML_SUSPECT = 0.35
ML_CORRUPT = 0.55
WEIGHTS_NAME = "gate.pt"
