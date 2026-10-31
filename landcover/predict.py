"""Classify a full Sentinel-2 GeoTIFF and write a georeferenced class map.

    python -m landcover.predict --input data/raw/nagpur_s2.tif --model-type unet \
        --out outputs/nagpur_classified.tif --vectorize
TensorFlow is imported lazily so the helpers here can be used (and tested) without it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from . import config
from .preprocess import load_s2_features


def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def classify_features(features, model, model_type, size=config.PATCH_SIZE, chunk=256):
    """Tile (H,W,F) features into size x size patches, predict, and stitch back to (H,W) uint8."""
    H, W, C = features.shape
    ph, pw = (-H) % size, (-W) % size
    padded = np.pad(features, ((0, ph), (0, pw), (0, 0)), mode="edge")
    nr, nc = padded.shape[0] // size, padded.shape[1] // size
    tiles = padded.reshape(nr, size, nc, size, C).transpose(0, 2, 1, 3, 4).reshape(-1, size, size, C)
    out = []
    for i in range(0, len(tiles), chunk):
        prob = np.asarray(model.predict(tiles[i:i + chunk], verbose=0))
        cls = prob.argmax(-1).astype(np.uint8)
        if model_type == "cnn":                                 # one label per patch
            cls = np.broadcast_to(cls[:, None, None], (len(cls), size, size))
        out.append(cls)
    cls = np.concatenate(out).reshape(nr, nc, size, size).transpose(0, 2, 1, 3).reshape(nr * size, nc * size)
    return np.ascontiguousarray(cls[:H, :W])


def classify_raster(tif_path, model, model_type):
    """Returns (class_map with 255 at nodata, rasterio profile, feature stack)."""
    feats, valid, profile = load_s2_features(tif_path)
    class_map = classify_features(feats, model, model_type)
    class_map[~valid] = config.IGNORE_INDEX
    return class_map, profile, feats


def _hex_to_rgba(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


def write_geotiff(class_map, profile, out_path):
    """Save the class map with the source CRS/transform, so QGIS overlays it correctly."""
    prof = profile.copy()
    for k in ("blockxsize", "blockysize", "tiled", "interleave", "photometric"):
        prof.pop(k, None)
    prof.update(count=1, dtype="uint8", nodata=config.IGNORE_INDEX, compress="lzw")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(class_map.astype("uint8"), 1)
        dst.write_colormap(1, {i: _hex_to_rgba(c) for i, c in enumerate(config.CLASS_COLORS)})
        dst.update_tags(1, **{f"class_{i}": n for i, n in enumerate(config.CLASS_NAMES)})


def pixel_area_from_profile(profile):
    """Pixel area in m^2. Falls back to 10 m x 10 m if the CRS is geographic."""
    crs, t = profile.get("crs"), profile["transform"]
    if crs is not None and crs.is_projected:
        return abs(t.a * t.e)
    return float(config.SCALE_M ** 2)


def area_stats(class_map, pixel_area_m2=100.0):
    total = int((class_map != config.IGNORE_INDEX).sum())
    rows = []
    for i, name in enumerate(config.CLASS_NAMES):
        n = int((class_map == i).sum())
        rows.append({"class_id": i, "class": name, "pixels": n,
                     "area_km2": n * pixel_area_m2 / 1e6,
                     "percent": 100.0 * n / max(total, 1)})
    return pd.DataFrame(rows)


def vectorize_classes(class_map, transform, crs):
    """Polygonise the class map (GeoPandas/Shapely) for use in QGIS. Slow on very large scenes."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape

    mask = class_map != config.IGNORE_INDEX
    geoms, ids = [], []
    for g, v in shapes(class_map.astype("uint8"), mask=mask, transform=transform):
        geoms.append(shape(g))
        ids.append(int(v))
    gdf = gpd.GeoDataFrame({"class_id": ids}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve(by="class_id", as_index=False)
    gdf["class"] = [config.CLASS_NAMES[i] for i in gdf["class_id"]]
    gdf["area_km2"] = gdf.geometry.area / 1e6 if crs is not None and crs.is_projected else np.nan
    return gdf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--model-type", choices=["cnn", "unet"], default="unet")
    p.add_argument("--model-path", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--vectorize", action="store_true", help="also write a GeoPackage of class polygons")
    a = p.parse_args()

    model_path = a.model_path or config.MODEL_DIR / f"{a.model_type}.keras"
    out = a.out or config.OUT_DIR / (Path(a.input).stem.replace("_s2", "") + "_classified.tif")
    config.OUT_DIR.mkdir(exist_ok=True)

    class_map, profile, _ = classify_raster(a.input, load_model(model_path), a.model_type)
    write_geotiff(class_map, profile, out)
    stats = area_stats(class_map, pixel_area_from_profile(profile))
    stats.to_csv(str(out).replace(".tif", "_area_stats.csv"), index=False)
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"wrote {out}")
    if a.vectorize:
        gdf = vectorize_classes(class_map, profile["transform"], profile.get("crs"))
        gpkg = str(out).replace(".tif", ".gpkg")
        gdf.to_file(gpkg, driver="GPKG")
        print(f"wrote {gpkg}")


if __name__ == "__main__":
    main()
