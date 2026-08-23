# Spoilage — Image Integrity Gate

A quality gate in front of deployed AI. Upload a frame; five **classical** OpenCV/NumPy signals score it for blur, noise, JPEG blockiness, clipping, and missing tiles; the API returns a verdict (`clean` / `suspect` / `corrupt`), per-signal scores, and human-readable reasons.

No learned model is shipped. Numbers below are from `python -m spoilage.bench` on the six committed plates — not a claim about real camera data.

**Public demo:** https://spoilage-integrity-gate.fly.dev/

## Why AMD should care

AMD’s 2027 undergrad ML/AI posting assigns **Image Corruption Detection** as project work and asks for research, development, and deployment of ML and computer vision on products. Spoilage is that gate in miniature: it sits in front of a downstream model, flags garbage before it wrecks **user experience**, and exposes the reasons so a broken deployment can be **fixed**. The stack matches the posting’s tools (Python, classical/CV math you can defend, a public deploy) without pretending a two-week research program ran overnight.

## Bench (CPU, classical signals only)

> Detected 100.0% of synthetically corrupted images at 0.0% false-positive rate across 36 images, mean analysis latency 5ms on CPU.

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

Hold-out: 6 clean plates × 5 synthetic families (blur, noise, JPEG q=12, highlight/shadow clip, dropped tiles). Flagged = verdict `suspect` or `corrupt`. See `bench-results/latest.md`.

## How it maps to the posting

| Posting phrase | What this repo actually does |
| --- | --- |
| Image Corruption Detection | Five explicit corruption families, scored and fused |
| Deployed AI / user experience | Gate the input; explain the reject so UX can be repaired |
| Python · PyTorch | Python + OpenCV/NumPy. **No PyTorch** — a pretrained download was cut so the demo ships today |
| Cloud | Public Docker service on Fly.io (Hugging Face / Render were not authenticated on this machine) |

## Signals and weights

Each signal is a pure function: `ndarray → {value, score}` with `score ∈ [0, 1]` (1 = corrupt).

| Signal | Raw metric | Weight |
| --- | --- | --- |
| blur | Laplacian variance (low ⇒ blurry) | 0.45 |
| noise | mean \|I − medianBlur(I)\| | 0.45 |
| compression | 8×8 block-boundary / intra-block gradient ratio | 0.45 |
| clipping | fraction of pixels with a channel at 0 or 255 | 0.45 |
| occlusion | fraction of pixels in large near-constant rectangles | 0.45 |

Fusion is a **weighted sum**. Each family is weighted so a severe hit on one channel trips the gate. Verdict cutoffs: suspect ≥ 0.28, corrupt ≥ 0.42. Thresholds live in `spoilage/config.py`.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn spoilage.api:app --reload --port 7860
```

- UI: http://127.0.0.1:7860/
- `GET /health`
- `POST /analyze` multipart field `file`
- `POST /corrupt` + form `attack` ∈ `blur|noise|jpeg|clip|tiles`
- `python -m spoilage.bench`
- `pytest`

## Synthetic attacks

`spoilage/corrupt.py` takes a clean frame and a seed:

- Gaussian blur
- Gaussian / Poisson noise
- JPEG recompress at quality 10/12/20
- highlight / shadow clip
- random rectangular tiles filled with mid-gray

Six small public-domain synthetic plates are committed under `/samples` so the bench is reproducible.

## Resume bullet

> Built a Python/OpenCV image-integrity gate that detected 100% of synthetically corrupted images at 0% false positives across 36 samples (blur, JPEG, noise, clipping, missing tiles), exposing per-signal scores and human-readable reasons over a FastAPI endpoint with 5ms mean CPU latency; deployed as a public demo.
