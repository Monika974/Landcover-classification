import numpy as np

from landcover import config
from landcover.preprocess import extract_patches


def _scene(h=128, w=128):
    feats = np.random.rand(h, w, config.N_FEATURES).astype("float32")
    mask = np.zeros((h, w), np.uint8)
    mask[:, 64:] = 2
    return feats, mask, np.ones((h, w), bool)


def test_patch_count_and_shapes():
    feats, mask, valid = _scene()
    X, ym, yp = extract_patches(feats, mask, valid, size=64, stride=64)
    assert X.shape == (4, 64, 64, config.N_FEATURES) and X.dtype == np.float16
    assert ym.shape == (4, 64, 64) and sorted(yp.tolist()) == [0, 0, 2, 2]


def test_invalid_patches_dropped_and_pixels_ignored():
    feats, mask, valid = _scene()
    valid[:64, :64] = False                          # one fully invalid patch
    valid[64:70, 64:70] = False                      # a few invalid pixels elsewhere
    X, ym, yp = extract_patches(feats, mask, valid, size=64, stride=64)
    assert len(X) == 3
    assert (ym == config.IGNORE_INDEX).any()


def test_empty_scene():
    feats, mask, valid = _scene(32, 32)
    X, ym, yp = extract_patches(feats, mask, valid, size=64, stride=64)
    assert len(X) == len(ym) == len(yp) == 0
