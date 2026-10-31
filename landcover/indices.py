"""Spectral indices and feature stack construction."""
import numpy as np
from . import config

EPS = 1e-6


def ndvi(nir, red):
    return (nir - red) / (nir + red + EPS)


def ndwi(green, nir):
    return (green - nir) / (green + nir + EPS)


def build_features(dn):
    """dn: (5, H, W) raw Sentinel-2 SR digital numbers in BANDS order.

    Returns a (H, W, 7) float32 array: B2, B3, B4, B8, B11 (0..1), NDVI, NDWI.
    """
    refl = np.clip(dn.astype("float32") / config.REFLECTANCE_SCALE, 0.0, 1.0)
    blue, green, red, nir, swir = refl
    feats = np.stack([blue, green, red, nir, swir, ndvi(nir, red), ndwi(green, nir)], axis=-1)
    return feats.astype("float32")
