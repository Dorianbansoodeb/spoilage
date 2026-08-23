"""Holdout behavioral bench + Organism-B transfer. Train probes never evaluated here."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spoilage.behavior.analyze import analyze_transcript
from spoilage.behavior.detect import BehaviorGate, fit_logistic, get_gate
from spoilage.behavior.probes import probes
from spoilage.behavior.signals import features_from_signals
from spoilage.model import roc_auc

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "bench-results"


def _rows_for(backend: str, split: str) -> list[dict]:
    rows = []
    for probe in probes(split=split):
        for contaminated in (False, True):
            result = analyze_transcript(
                probe=probe,
                contaminate=contaminated,
                backend=backend,
            )
            rows.append(
                {
                    "probeId": probe.id,
                    "family": probe.family,
                    "backend": backend,
                    "contaminated": contaminated,
                    "verdict": result["verdict"],
                    "score": result["score"],
                    "baseline": result["baseline"],
                    "signals": result["signals"],
                    "detected": result["verdict"] in {"suspect", "corrupt"},
                    "y": int(contaminated),
                }
            )
    return rows


def fit_detector() -> dict:
    """Fit logistic weights on Organism A *train* probes only."""
    X: list[list[float]] = []
    y: list[int] = []
    for probe in probes(split="train"):
        for contaminated in (False, True):
            result = analyze_transcript(
                probe=probe,
                contaminate=contaminated,
                backend="organism-a",
            )
            X.append(features_from_signals(result["signals"]))
            y.append(int(contaminated))
    blob = fit_logistic(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
    gate = BehaviorGate()
    gate.save(blob)
    global_gate = get_gate()
    global_gate.blob = blob
    return {"n": len(y), "weights": blob["weights"].tolist(), "bias": blob["bias"]}


def _summarize(rows: list[dict], label: str) -> dict:
    y_true = np.asarray([r["y"] for r in rows], dtype=float)
    y_score = np.asarray([r["score"] for r in rows], dtype=float)
    y_base = np.asarray([r["baseline"]["score"] for r in rows], dtype=float)
    clean = [r for r in rows if not r["contaminated"]]
    spoiled = [r for r in rows if r["contaminated"]]
    recall = sum(r["detected"] for r in spoiled) / len(spoiled) if spoiled else 0.0
    fpr = sum(r["detected"] for r in clean) / len(clean) if clean else 0.0
    base_recall = (
        sum(r["baseline"]["verdict"] in {"suspect", "corrupt"} for r in spoiled) / len(spoiled)
        if spoiled
        else 0.0
    )
    base_fpr = (
        sum(r["baseline"]["verdict"] in {"suspect", "corrupt"} for r in clean) / len(clean)
        if clean
        else 0.0
    )
    families = sorted({r["family"] for r in rows})
    per_family = {}
    for family in families:
        fam = [r for r in spoiled if r["family"] == family]
        per_family[family] = sum(r["detected"] for r in fam) / len(fam) if fam else 0.0
    return {
        "label": label,
        "n": len(rows),
        "recall": round(recall, 4),
        "falsePositiveRate": round(fpr, 4),
        "auroc": round(roc_auc(y_true, y_score), 4),
        "baseline": {
            "recall": round(base_recall, 4),
            "falsePositiveRate": round(base_fpr, 4),
            "auroc": round(roc_auc(y_true, y_base), 4),
        },
        "perFamilyRecall": {k: round(v, 4) for k, v in per_family.items()},
    }


def run_bench(*, write_files: bool = True, fit: bool = True) -> dict:
    fit_info = fit_detector() if fit else {"n": 0}
    holdout = _rows_for("organism-a", "holdout")
    transfer = _rows_for("organism-b", "holdout")
    a_stats = _summarize(holdout, "organism-a holdout")
    b_stats = _summarize(transfer, "organism-b transfer")
    headline = (
        f"Behavior holdout: Organism A AUROC {a_stats['auroc']:.3f} "
        f"(classical {a_stats['baseline']['auroc']:.3f}); "
        f"transfer to Organism B AUROC {b_stats['auroc']:.3f}. "
        f"Detected {a_stats['recall']*100:.1f}% of contaminated A transcripts "
        f"at {a_stats['falsePositiveRate']*100:.1f}% FPR."
    )
    payload = {
        "headline": headline,
        "split": "grouped holdout — train probes never scored; Organism B is a trigger-transfer",
        "fit": fit_info,
        "organismA": a_stats,
        "organismB": b_stats,
        "rows": holdout + transfer,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "model": "logistic-on-classical-signals",
    }
    if write_files:
        RESULTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        text = render_markdown(payload)
        (RESULTS / f"{stamp}-behavior.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (RESULTS / f"{stamp}-behavior.md").write_text(text, encoding="utf-8")
        (RESULTS / "behavior-latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (RESULTS / "behavior-latest.md").write_text(text, encoding="utf-8")
        payload["paths"] = {
            "json": str(RESULTS / f"{stamp}-behavior.json"),
            "markdown": str(RESULTS / f"{stamp}-behavior.md"),
        }
    return payload


def render_markdown(payload: dict) -> str:
    a, b = payload["organismA"], payload["organismB"]
    fam = a["perFamilyRecall"]
    lines = [
        "# Spoilage behavioral bench",
        "",
        payload["headline"],
        "",
        payload["split"],
        "",
        "| Metric | Organism A holdout | Organism B transfer | Classical (A) |",
        "| --- | --- | --- | --- |",
        f"| AUROC | {a['auroc']:.3f} | {b['auroc']:.3f} | {a['baseline']['auroc']:.3f} |",
        f"| Recall on contaminated | {a['recall']*100:.1f}% | {b['recall']*100:.1f}% | {a['baseline']['recall']*100:.1f}% |",
        f"| False-positive rate | {a['falsePositiveRate']*100:.1f}% | {b['falsePositiveRate']*100:.1f}% | {a['baseline']['falsePositiveRate']*100:.1f}% |",
        f"| N (clean + contaminated) | {a['n']} | {b['n']} | {a['n']} |",
    ]
    if fam:
        lines.append("")
        lines.append("Per-family recall on contaminated Organism A holdout:")
        for name, value in fam.items():
            lines.append(f"- {name}: {value*100:.1f}%")
    lines += [
        "",
        "Detector is logistic regression on five classical signals, fit only on Organism A train probes.",
        "Organism B uses a different trigger wrapper so transfer is not a string match.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = run_bench(write_files=True, fit=True)
    print(render_markdown(payload))
    print(payload.get("paths"))


if __name__ == "__main__":
    main()
