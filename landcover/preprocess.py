"""Turn exported GeoTIFFs into patch datasets (train/val/test .npz files)."""
import argparse

import numpy as np
import rasterio

from . import config
from .indices import build_features
from .labels import remap_worldcover


def load_s2_features(path):
    """Read an exported Sentinel-2 GeoTIFF -> (features HxWx7, valid mask HxW, profile)."""
    with rasterio.open(path) as src:
        if src.count < len(config.BANDS):
            raise ValueError(f"{path} has {src.count} bands, expected {len(config.BANDS)}")
        dn = src.read(list(range(1, len(config.BANDS) + 1))).astype("float32")
        profile = src.profile.copy()
    valid = (dn > 0).any(axis=0)           # exported with unmask(0): all-zero = no data
    feats = build_features(dn)
    feats[~valid] = 0.0
    return feats, valid, profile


def extract_patches(features, mask, valid, size=config.PATCH_SIZE, stride=config.STRIDE,
                    min_valid=config.MIN_VALID_FRACTION, n_classes=config.N_CLASSES):
    """Slide a window over the scene.

    Returns X (N,size,size,F) float16, y_mask (N,size,size) uint8 with 255 = ignore,
    y_patch (N,) uint8 = majority class of valid pixels (for the patch CNN).
    """
    mask = mask.copy()
    mask[~valid] = config.IGNORE_INDEX
    H, W = mask.shape
    xs, ys, yp = [], [], []
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            m = mask[r:r + size, c:c + size]
            ok = m != config.IGNORE_INDEX
            if ok.mean() < min_valid:
                continue
            xs.append(features[r:r + size, c:c + size])
            ys.append(m)
            yp.append(np.bincount(m[ok].astype(np.int64), minlength=n_classes).argmax())
    if not xs:
        f = features.shape[-1]
        return (np.zeros((0, size, size, f), np.float16), np.zeros((0, size, size), np.uint8),
                np.zeros((0,), np.uint8))
    return (np.stack(xs).astype(np.float16), np.stack(ys).astype(np.uint8),
            np.array(yp, dtype=np.uint8))


def build_dataset(size=config.PATCH_SIZE, stride=config.STRIDE):
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    buckets = {"train": [], "val": [], "test": []}
    for name, info in config.REGIONS.items():
        s2_path = config.RAW_DIR / f"{name}_s2.tif"
        lab_path = config.RAW_DIR / f"{name}_label.tif"
        if not (s2_path.exists() and lab_path.exists()):
            print(f"[skip] {name}: missing {s2_path.name} or {lab_path.name}")
            continue
        feats, valid, _ = load_s2_features(s2_path)
        with rasterio.open(lab_path) as src:
            labels = remap_worldcover(src.read(1))
        h, w = min(feats.shape[0], labels.shape[0]), min(feats.shape[1], labels.shape[1])
        if (h, w) != feats.shape[:2] or (h, w) != labels.shape:
            print(f"[warn] {name}: image/label shapes differ, cropping to {h}x{w}")
        X, y_mask, y_patch = extract_patches(feats[:h, :w], labels[:h, :w], valid[:h, :w], size, stride)
        print(f"[{info['split']:5s}] {name}: {len(X)} patches")
        buckets[info["split"]].append((X, y_mask, y_patch, np.full(len(X), name)))

    for split, items in buckets.items():
        if not items:
            print(f"[warn] no data for split '{split}'")
            continue
        X = np.concatenate([i[0] for i in items])
        y_mask = np.concatenate([i[1] for i in items])
        y_patch = np.concatenate([i[2] for i in items])
        regions = np.concatenate([i[3] for i in items])
        np.savez_compressed(config.PROC_DIR / f"{split}.npz", X=X, y_mask=y_mask,
                            y_patch=y_patch, region=regions)
        counts = np.bincount(y_patch, minlength=config.N_CLASSES)
        print(f"saved {split}.npz  X={X.shape}  patch-class counts={dict(zip(config.CLASS_NAMES, counts.tolist()))}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=config.PATCH_SIZE)
    p.add_argument("--stride", type=int, default=config.STRIDE)
    a = p.parse_args()
    build_dataset(a.size, a.stride)


if __name__ == "__main__":
    main()
