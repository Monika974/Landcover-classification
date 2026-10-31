import numpy as np

from landcover import config
from landcover.indices import build_features, ndvi, ndwi
from landcover.labels import remap_worldcover


def test_worldcover_remap():
    codes = np.array([[10, 40, 50], [60, 80, 0]], dtype=np.uint8)
    out = remap_worldcover(codes)
    assert out[0].tolist() == [1, 2, 3]
    assert out[1, 0] == 4 and out[1, 1] == 0
    assert out[1, 2] == config.IGNORE_INDEX          # code 0 = no data


def test_indices_range_and_sign():
    assert ndvi(np.array([0.5]), np.array([0.1]))[0] > 0.5
    assert ndwi(np.array([0.3]), np.array([0.05]))[0] > 0.5
    assert ndvi(np.array([0.0]), np.array([0.0]))[0] == 0.0


def test_build_features_shape():
    dn = np.random.randint(1, 5000, size=(5, 8, 9)).astype("float32")
    f = build_features(dn)
    assert f.shape == (8, 9, config.N_FEATURES)
    assert f.dtype == np.float32 and f[..., :5].max() <= 1.0
