import numpy as np

from landcover import config
from landcover.predict import area_stats, classify_features


class DummyModel:
    """Predicts class 3 for every pixel (unet) or patch (cnn) - no TensorFlow needed."""
    def __init__(self, kind):
        self.kind = kind

    def predict(self, x, verbose=0):
        n, s = len(x), x.shape[1]
        shape = (n, s, s, config.N_CLASSES) if self.kind == "unet" else (n, config.N_CLASSES)
        p = np.zeros(shape, np.float32)
        p[..., 3] = 1.0
        return p


def test_classify_features_handles_non_multiple_sizes():
    feats = np.random.rand(100, 70, config.N_FEATURES).astype("float32")
    for kind in ("unet", "cnn"):
        out = classify_features(feats, DummyModel(kind), kind)
        assert out.shape == (100, 70) and out.dtype == np.uint8 and (out == 3).all()


def test_area_stats():
    cm = np.array([[0, 0, 1, 255], [1, 1, 2, 255]], np.uint8)
    s = area_stats(cm, pixel_area_m2=100.0)
    assert abs(s["percent"].sum() - 100.0) < 1e-9
    assert s.loc[s.class_id == 1, "pixels"].iloc[0] == 3
    assert abs(s["area_km2"].sum() - 6 * 100 / 1e6) < 1e-12
