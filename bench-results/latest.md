# Spoilage bench

Detected 100.0% of synthetically corrupted images at 0.0% false-positive rate across 36 images, mean analysis latency 5ms on CPU.

| Metric | Value |
| --- | --- |
| Detection rate (recall on corrupted) | 100.0% |
| False-positive rate (clean) | 0.0% |
| N (clean + corrupted) | 36 |
| Mean CPU latency | 5 ms |
| Blur recall | 100.0% |
| Noise recall | 100.0% |
| JPEG recall | 100.0% |
| Clip recall | 100.0% |
| Missing-tile recall | 100.0% |

Classical OpenCV/NumPy signals only. No learned model.
