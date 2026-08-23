"""Hold-out bench: clean frames vs each synthetic corruption family."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from spoilage.combine import analyze_image
from spoilage.corrupt import ATTACKS, apply_attack
from spoilage.fixtures import write_samples

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
RESULTS = ROOT / "bench-results"

FAMILIES = {
    "blur": "blur",
    "noise": "noise",
    "jpeg": "jpeg",
    "clip": "clip",
    "tiles": "tiles",
}


def load_samples() -> list[tuple[str, object]]:
    if not SAMPLES.is_dir() or not list(SAMPLES.glob("*.jpg")):
        write_samples(SAMPLES)
    frames: list[tuple[str, object]] = []
    for path in sorted(SAMPLES.glob("*.jpg")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {path}")
        frames.append((path.stem, image))
    if not frames:
        raise RuntimeError("no sample images")
    return frames


def _flagged(verdict: str) -> bool:
    return verdict in {"suspect", "corrupt"}


def run_bench(*, write_files: bool = True) -> dict:
    frames = load_samples()
    rows: list[dict] = []
    latencies: list[int] = []

    clean_flagged = 0
    for name, image in frames:
        result = analyze_image(image)
        latencies.append(result["latencyMs"])
        flagged = _flagged(result["verdict"])
        clean_flagged += int(flagged)
        rows.append(
            {
                "image": name,
                "family": "clean",
                "verdict": result["verdict"],
                "score": result["score"],
                "latencyMs": result["latencyMs"],
                "detected": flagged,
            }
        )

    family_hits: dict[str, list[bool]] = {k: [] for k in FAMILIES}
    for name, image in frames:
        for family, attack in FAMILIES.items():
            wrecked = apply_attack(image, attack, seed=abs(hash(name + family)) % 10_000)
            result = analyze_image(wrecked)
            latencies.append(result["latencyMs"])
            flagged = _flagged(result["verdict"])
            family_hits[family].append(flagged)
            rows.append(
                {
                    "image": name,
                    "family": family,
                    "verdict": result["verdict"],
                    "score": result["score"],
                    "latencyMs": result["latencyMs"],
                    "detected": flagged,
                }
            )

    n_clean = len(frames)
    n_corrupt = sum(len(v) for v in family_hits.values())
    n_total = n_clean + n_corrupt
    corrupt_hits = sum(sum(v) for v in family_hits.values())
    recall = corrupt_hits / n_corrupt if n_corrupt else 0.0
    fpr = clean_flagged / n_clean if n_clean else 0.0
    per_family = {
        family: (sum(hits) / len(hits) if hits else 0.0)
        for family, hits in family_hits.items()
    }
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    headline = (
        f"Detected {recall*100:.1f}% of synthetically corrupted images at "
        f"{fpr*100:.1f}% false-positive rate across {n_total} images, "
        f"mean analysis latency {mean_latency:.0f}ms on CPU."
    )

    payload = {
        "headline": headline,
        "nImages": n_total,
        "nClean": n_clean,
        "nCorrupted": n_corrupt,
        "recall": round(recall, 4),
        "falsePositiveRate": round(fpr, 4),
        "perFamilyRecall": {k: round(v, 4) for k, v in per_family.items()},
        "meanLatencyMs": round(mean_latency, 2),
        "rows": rows,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": "classical-opencv",
    }

    if write_files:
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        json_path = RESULTS / f"{stamp}.json"
        md_path = RESULTS / f"{stamp}.md"
        latest_json = RESULTS / "latest.json"
        latest_md = RESULTS / "latest.md"
        text = render_markdown(payload)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(text, encoding="utf-8")
        latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        latest_md.write_text(text, encoding="utf-8")
        payload["paths"] = {"json": str(json_path), "markdown": str(md_path)}

    return payload


def render_markdown(payload: dict) -> str:
    fam = payload["perFamilyRecall"]
    lines = [
        "# Spoilage bench",
        "",
        payload["headline"],
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Detection rate (recall on corrupted) | {payload['recall']*100:.1f}% |",
        f"| False-positive rate (clean) | {payload['falsePositiveRate']*100:.1f}% |",
        f"| N (clean + corrupted) | {payload['nImages']} |",
        f"| Mean CPU latency | {payload['meanLatencyMs']:.0f} ms |",
        f"| Blur recall | {fam['blur']*100:.1f}% |",
        f"| Noise recall | {fam['noise']*100:.1f}% |",
        f"| JPEG recall | {fam['jpeg']*100:.1f}% |",
        f"| Clip recall | {fam['clip']*100:.1f}% |",
        f"| Missing-tile recall | {fam['tiles']*100:.1f}% |",
        "",
        "Classical OpenCV/NumPy signals only. No learned model.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = run_bench(write_files=True)
    print(render_markdown(payload))
    print(payload.get("paths"))


if __name__ == "__main__":
    main()
