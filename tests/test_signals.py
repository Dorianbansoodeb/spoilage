import numpy as np

from spoilage.corrupt import apply_blur, apply_clip, apply_jpeg, apply_noise, apply_tiles
from spoilage.fixtures import photo_plate, sharp_edge_plate, smooth_field
from spoilage.signals import (
    blur_signal,
    clipping_signal,
    compression_signal,
    noise_signal,
    occlusion_signal,
)


def test_blur_score_rises_on_gaussian():
    sharp = sharp_edge_plate()
    blurry = apply_blur(sharp, seed=1, radius=8.0)
    s0 = blur_signal(sharp)
    s1 = blur_signal(blurry)
    assert s0["value"] > s1["value"]
    assert s1["score"] > s0["score"] + 0.35
    assert s1["score"] > 0.6


def test_noise_score_rises_on_gaussian():
    clean = sharp_edge_plate()
    noisy = apply_noise(clean, seed=2, sigma=28.0)
    s0 = noise_signal(clean)
    s1 = noise_signal(noisy)
    assert s1["value"] > s0["value"]
    assert s1["score"] > s0["score"] + 0.3


def test_compression_score_rises_on_jpeg():
    clean = photo_plate()
    jpg = apply_jpeg(clean, seed=3, quality=8)
    s0 = compression_signal(clean)
    s1 = compression_signal(jpg)
    assert s1["value"] > s0["value"]
    assert s1["score"] > s0["score"]


def test_clipping_score_rises_on_crush():
    mid = np.full((180, 240, 3), 128, dtype=np.uint8)
    mid[20:80, 20:80] = (40, 90, 160)
    crushed = apply_clip(mid, seed=4, lo=70, hi=170)
    s0 = clipping_signal(mid)
    s1 = clipping_signal(crushed)
    assert s1["value"] > s0["value"]
    assert s1["score"] > 0.25


def test_occlusion_score_rises_on_dropped_tiles():
    rng = np.random.default_rng(5)
    textured = np.clip(
        photo_plate().astype(np.int16) + rng.integers(-5, 6, (240, 320, 3)),
        20,
        230,
    ).astype(np.uint8)
    tiled = apply_tiles(textured, seed=5, n=4, size_frac=0.24)
    s0 = occlusion_signal(textured)
    s1 = occlusion_signal(tiled)
    assert s1["value"] > s0["value"]
    assert s1["score"] > s0["score"] + 0.35


def test_smooth_field_is_blurry_and_occluded():
    flat = smooth_field()
    assert blur_signal(flat)["score"] > 0.8
    assert occlusion_signal(flat)["score"] > 0.8
