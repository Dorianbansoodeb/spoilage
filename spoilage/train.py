"""Train the late-fusion gate on synthetic plates that are not the public holdout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spoilage.config import FAMILIES, FAMILY_TO_IDX, ML_CORRUPT, ML_SUSPECT
from spoilage.corrupt import TRAIN_SEVERITIES, apply_attack
from spoilage.fixtures import make_train_plate
from spoilage.model import (
    WEIGHTS_DIR,
    WEIGHTS_PATH,
    IntegrityGate,
    features_from_signals,
    image_to_tensor,
    roc_auc,
)
from spoilage.signals import measure_all

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Example:
    image: np.ndarray
    features: np.ndarray
    y_bin: int
    y_fam: int


def _build_example(plate: np.ndarray, family: str, seed: int, **attack_kwargs) -> Example:
    frame = plate if family == "clean" else apply_attack(plate, family, seed=seed, **attack_kwargs)
    signals = measure_all(frame)
    return Example(
        image=frame,
        features=features_from_signals(signals),
        y_bin=0 if family == "clean" else 1,
        y_fam=FAMILY_TO_IDX[family],
    )


def synthesize(n_plates: int, seed0: int) -> list[Example]:
    examples: list[Example] = []
    for i in range(n_plates):
        plate = make_train_plate(seed0 + i)
        # Match attack count so the binary head is not majority-corrupt.
        for j in range(5):
            examples.append(
                _build_example(make_train_plate(seed0 + i + 10_000 * (j + 1)), "clean", seed0 + i + j)
            )
        for family, grid in TRAIN_SEVERITIES.items():
            params = grid[i % len(grid)]
            examples.append(
                _build_example(plate, family, seed0 + i * 31 + FAMILY_TO_IDX[family], **params)
            )
    return examples


class GateDataset(Dataset):
    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        return (
            image_to_tensor(ex.image),
            torch.from_numpy(ex.features),
            torch.tensor(ex.y_bin, dtype=torch.float32),
            torch.tensor(ex.y_fam, dtype=torch.long),
        )


def _epoch(model: IntegrityGate, loader: DataLoader, opt: torch.optim.Optimizer | None) -> dict:
    train = opt is not None
    model.train(train)
    total = 0.0
    n = 0
    ys: list[float] = []
    ps: list[float] = []
    fam_correct = 0
    for image, feats, y_bin, y_fam in loader:
        logit, fam_logit = model(image, feats)
        loss = 0.65 * torch.nn.functional.binary_cross_entropy_with_logits(
            logit, y_bin
        ) + 0.35 * torch.nn.functional.cross_entropy(fam_logit, y_fam)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total += float(loss.item()) * len(y_bin)
        n += len(y_bin)
        ys.extend(y_bin.tolist())
        ps.extend(torch.sigmoid(logit).detach().tolist())
        fam_correct += int((fam_logit.argmax(dim=-1) == y_fam).sum().item())
    return {
        "loss": total / max(n, 1),
        "auroc": roc_auc(np.asarray(ys), np.asarray(ps)),
        "familyAcc": fam_correct / max(n, 1),
        "n": n,
    }


def _choose_thresholds(model: IntegrityGate, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    ys: list[float] = []
    ps: list[float] = []
    with torch.inference_mode():
        for image, feats, y_bin, _y_fam in loader:
            logit, _ = model(image, feats)
            ys.extend(y_bin.tolist())
            ps.extend(torch.sigmoid(logit).tolist())
    y = np.asarray(ys)
    p = np.asarray(ps)
    clean_scores = p[y == 0]
    corrupt_scores = p[y == 1]
    if len(clean_scores) and len(corrupt_scores):
        hi_clean = float(np.quantile(clean_scores, 0.99))
        lo_cor = float(np.quantile(corrupt_scores, 0.10))
        mid = 0.5 * (hi_clean + lo_cor) if lo_cor > hi_clean else hi_clean + 0.15
        best_t = max(0.48, min(0.80, mid))
        suspect = max(0.40, hi_clean + 0.06, best_t - 0.12)
    else:
        best_t = ML_CORRUPT
        suspect = ML_SUSPECT
    return min(suspect, best_t - 0.04), best_t


def main() -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    train_ex = synthesize(40, seed0=2026)
    val_ex = synthesize(12, seed0=9000)
    train_loader = DataLoader(GateDataset(train_ex), batch_size=32, shuffle=True)
    val_loader = DataLoader(GateDataset(val_ex), batch_size=32, shuffle=False)

    model = IntegrityGate()
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    history = []
    best_state = None
    best_auroc = -1.0
    for epoch in range(1, 18):
        tr = _epoch(model, train_loader, opt)
        va = _epoch(model, val_loader, None)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"epoch {epoch:02d}  train loss {tr['loss']:.3f} auroc {tr['auroc']:.3f}  "
            f"val loss {va['loss']:.3f} auroc {va['auroc']:.3f} fam {va['familyAcc']:.3f}"
        )
        if va["auroc"] >= best_auroc:
            best_auroc = va["auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    suspect, corrupt = _choose_thresholds(model, val_loader)
    mix = float(torch.sigmoid(model.mix).item())
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "threshold_suspect": suspect,
        "threshold_corrupt": corrupt,
        "mix": mix,
        "families": list(FAMILIES),
        "val_auroc": best_auroc,
    }
    torch.save(payload, WEIGHTS_PATH)
    meta = {
        "weights": str(WEIGHTS_PATH.relative_to(ROOT)),
        "trainN": len(train_ex),
        "valN": len(val_ex),
        "valAuroc": round(best_auroc, 4),
        "thresholdSuspect": round(suspect, 4),
        "thresholdCorrupt": round(corrupt, 4),
        "learnedMixCnn": round(mix, 4),
        "holdout": "committed /samples plates — never used in training",
        "history": history,
    }
    (WEIGHTS_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {WEIGHTS_PATH}  val AUROC {best_auroc:.3f}  t_sus={suspect:.2f} t_cor={corrupt:.2f}")


if __name__ == "__main__":
    main()
