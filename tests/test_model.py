import numpy as np

from spoilage.fixtures import photo_plate, sharp_edge_plate
from spoilage.model import IntegrityGate, features_from_signals, roc_auc
from spoilage.signals import measure_all


def test_feature_vector_is_10d():
    vec = features_from_signals(measure_all(photo_plate()))
    assert vec.shape == (10,)
    assert np.isfinite(vec).all()


def test_gate_forward_shapes():
    import torch

    from spoilage.model import image_to_tensor

    model = IntegrityGate()
    model.eval()
    image = image_to_tensor(sharp_edge_plate()).unsqueeze(0)
    feats = torch.from_numpy(features_from_signals(measure_all(sharp_edge_plate()))).unsqueeze(0)
    logit, fam = model(image, feats)
    assert logit.shape == (1,)
    assert fam.shape == (1, 6)


def test_roc_auc_perfect_and_chance():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) > 0.99
    assert 0.4 < roc_auc(y, np.array([0.4, 0.6, 0.5, 0.5])) < 0.7
