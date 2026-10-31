"""Per-pixel Random Forest baseline (no spatial context) for the deep models to beat.

    python -m landcover.baseline_rf
"""
import argparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from . import config
from .metrics import compute_metrics, save_report


def _pixels(split, per_class=None, max_total=None, rng=None):
    d = np.load(config.PROC_DIR / f"{split}.npz")
    X = d["X"].reshape(-1, config.N_FEATURES).astype("float32")
    y = d["y_mask"].reshape(-1)
    keep = y != config.IGNORE_INDEX
    X, y = X[keep], y[keep]
    if per_class:
        idx = []
        for c in range(config.N_CLASSES):
            ci = np.flatnonzero(y == c)
            idx.append(rng.choice(ci, size=min(per_class, len(ci)), replace=False))
        idx = np.concatenate(idx)
        return X[idx], y[idx]
    if max_total and len(y) > max_total:
        idx = rng.choice(len(y), max_total, replace=False)
        return X[idx], y[idx]
    return X, y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-class", type=int, default=30000, help="balanced training samples per class")
    p.add_argument("--trees", type=int, default=200)
    a = p.parse_args()

    rng = np.random.default_rng(config.SEED)
    Xtr, ytr = _pixels("train", per_class=a.per_class, rng=rng)
    rf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=config.SEED)
    rf.fit(Xtr, ytr)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(rf, config.MODEL_DIR / "rf.joblib", compress=3)

    Xte, yte = _pixels("test", max_total=1_000_000, rng=rng)
    save_report("rf_test", compute_metrics(yte, rf.predict(Xte)))


if __name__ == "__main__":
    main()
