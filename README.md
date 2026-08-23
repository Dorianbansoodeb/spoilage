# Spoilage — Image Integrity Gate

A quality gate in front of deployed AI. A **27k-parameter PyTorch model** scores a frame for corruption; five **classical OpenCV/NumPy signals** explain the call. Upload an image, get `clean` / `suspect` / `corrupt`, p(corrupt), a predicted family, per-signal bars, and human-readable reasons.

**Public demo:** https://spoilage-integrity-gate.fly.dev/  
**Repository:** https://github.com/Dorianbansoodeb/spoilage

## Why AMD should care

AMD’s 2027 undergrad ML/AI posting assigns **Image Corruption Detection** as project work and asks for research, development, and deployment of ML and computer vision on products. Spoilage is that gate: a learned detector in front of a downstream model, so garbage does not wreck **user experience**, plus the classical reasons so a broken deployment can be **fixed**. Python, PyTorch, OpenCV, a grouped holdout, a public cloud deploy — without pretending this is ImageNet.

## What is actually learned

| Head | Input | Job |
| --- | --- | --- |
| Signal MLP | 10-d vector (5 OpenCV measurements + 5 scores) | Binary p(corrupt) |
| TinyCNN | 128×128 RGB | Spatial features |
| Family head | Late fusion of both | 6-way: clean / blur / noise / jpeg / clip / tiles |

The binary head is **features-only**. A pixel CNN overfits synthetic texture when the source plate is held out; the Laplacian / blockiness / clip / tile measurements transfer. That is the research result, not a footnote.

Training data is synthetic (procedural plates + seeded attacks). The six committed `/samples` plates are a **grouped holdout** — they never appear in train or val. Thresholds are calibrated from val score quantiles, then given a domain-shift floor so a clean holdout plate does not land in `suspect`.

No ImageNet weights. No overnight training. 27,528 parameters, `spoilage/weights/gate.pt` (117 KB).

## Holdout bench (CPU)

> Holdout (unseen plates): AUROC 1.000 vs classical 1.000; detected 100.0% of corrupted frames at 0.0% FPR across 36 images, mean CPU latency 8ms, family accuracy 43.3%.

| Metric | Learned gate | Classical baseline |
| --- | --- | --- |
| AUROC | 1.000 | 1.000 |
| Recall on corrupted | 100.0% | 100.0% |
| False-positive rate | 0.0% | 0.0% |
| Family accuracy (5-way on corrupted) | 43.3% | — |
| N (6 unseen plates × clean + 5 attacks) | 36 | 36 |
| Mean CPU latency | 8 ms | — |
| Val AUROC (training distribution, N=120) | 0.975 | — |

Family ID is auxiliary and only modestly above chance (20%). The product is the binary gate. See `bench-results/latest.md`.

## Signals (the explainable baseline)

| Signal | Raw metric |
| --- | --- |
| blur | Laplacian variance (low ⇒ blurry) |
| noise | mean \|I − medianBlur(I)\| |
| compression | 8×8 block-boundary / intra-block gradient ratio |
| clipping | fraction of pixels with a channel at 0 or 255 |
| occlusion | fraction of pixels in large near-constant rectangles |

Hand-tuned weighted sum is still computed and returned as `baseline` so the learned score can be compared on every request.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spoilage.train          # optional: retrain on CPU (~2 min)
python -m spoilage.bench
uvicorn spoilage.api:app --reload --port 7860
pytest
```

- UI: http://127.0.0.1:7860/
- `GET /health` — reports whether `gate.pt` loaded
- `POST /analyze` — multipart `file` → verdict, `model`, `baseline`, signals, reasons
- `POST /corrupt` — form `attack` ∈ `blur|noise|jpeg|clip|tiles`

## Resume bullet

> Built a PyTorch image-corruption detector (learned fusion of OpenCV signals + TinyCNN) with a grouped holdout: 1.00 AUROC, 100% recall at 0% FPR across 36 unseen-plate frames, 8ms mean CPU latency, plus human-readable classical reasons; trained on synthetic attacks and deployed as a public demo.
