"""Late-fusion integrity gate: TinyCNN on pixels + MLP on classical signals."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spoilage.config import FAMILIES, MODEL_INPUT, WEIGHTS_NAME

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / WEIGHTS_NAME

SIGNAL_ORDER = ("blur", "noise", "compression", "clipping", "occlusion")


def features_from_signals(signals: dict[str, dict[str, float]]) -> np.ndarray:
    """10-d vector: raw values (scaled) + per-signal scores."""
    raw = [
        np.log1p(signals["blur"]["value"]) / 8.0,
        signals["noise"]["value"] / 20.0,
        signals["compression"]["value"] / 3.0,
        signals["clipping"]["value"],
        signals["occlusion"]["value"],
    ]
    scores = [signals[name]["score"] for name in SIGNAL_ORDER]
    vec = np.asarray(raw + scores, dtype=np.float32)
    return np.clip(vec, 0.0, 4.0)


def image_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(bgr, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return F.relu(self.proj(h))


class SignalMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntegrityGate(nn.Module):
    """Signal MLP detects corruption; TinyCNN + MLP jointly name the family.

    Pixel CNNs overfit synthetic texture when the source plate is held out.
    The five classical measurements transfer, so the binary head is features-only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cnn = TinyCNN()
        self.mlp = SignalMLP()
        self.mix = nn.Parameter(torch.tensor(0.0))
        self.binary = nn.Linear(32, 1)
        self.family = nn.Linear(32, len(FAMILIES))

    def forward(
        self, image: torch.Tensor, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_mlp = self.mlp(feats)
        h_cnn = self.cnn(image)
        a = torch.sigmoid(self.mix)
        h_fam = a * h_cnn + (1.0 - a) * h_mlp
        return self.binary(h_mlp).squeeze(-1), self.family(h_fam)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.min() == y_true.max():
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    tpr = np.concatenate([[0.0], tps / tps[-1]])
    fpr = np.concatenate([[0.0], fps / fps[-1]])
    return float(np.trapezoid(tpr, fpr))


class GateSession:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or WEIGHTS_PATH
        self.device = torch.device("cpu")
        self.model = IntegrityGate().to(self.device)
        self.loaded = False
        self.threshold_suspect = 0.35
        self.threshold_corrupt = 0.55
        self.mix = 0.5
        if self.path.is_file():
            blob = torch.load(self.path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(blob["state_dict"])
            self.threshold_suspect = float(blob.get("threshold_suspect", 0.35))
            self.threshold_corrupt = float(blob.get("threshold_corrupt", 0.55))
            self.mix = float(blob.get("mix", 0.5))
            self.loaded = True
        self.model.eval()

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray, signals: dict[str, dict[str, float]]) -> dict:
        image = image_to_tensor(bgr).unsqueeze(0)
        feats = torch.from_numpy(features_from_signals(signals)).unsqueeze(0)
        logit, fam_logit = self.model(image, feats)
        p_corrupt = float(torch.sigmoid(logit)[0])
        fam_prob = torch.softmax(fam_logit, dim=-1)[0].tolist()
        family = FAMILIES[int(np.argmax(fam_prob))]
        if p_corrupt >= self.threshold_corrupt:
            verdict = "corrupt"
        elif p_corrupt >= self.threshold_suspect:
            verdict = "suspect"
        else:
            verdict = "clean"
        return {
            "loaded": self.loaded,
            "verdict": verdict,
            "pCorrupt": round(p_corrupt, 4),
            "family": family,
            "familyProbs": {name: round(p, 4) for name, p in zip(FAMILIES, fam_prob)},
            "thresholds": {
                "suspect": self.threshold_suspect,
                "corrupt": self.threshold_corrupt,
            },
        }


_SESSION: GateSession | None = None


def get_session() -> GateSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = GateSession()
    return _SESSION
