"""Evaluate a trained model on the spatially independent test regions.

    python -m landcover.evaluate --model unet
    python -m landcover.evaluate --model cnn
Note: the CNN is scored per patch, the U-Net and RF per pixel.
"""
import argparse

import numpy as np

from . import config
from .metrics import compute_metrics, save_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cnn", "unet"], required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    a = p.parse_args()

    import tensorflow as tf
    model = tf.keras.models.load_model(config.MODEL_DIR / f"{a.model}.keras", compile=False)
    d = np.load(config.PROC_DIR / f"{a.split}.npz")
    X = d["X"].astype("float32")
    prob = model.predict(X, batch_size=32, verbose=0)
    if a.model == "unet":
        y, pred = d["y_mask"], prob.argmax(-1)
        keep = y != config.IGNORE_INDEX
        y_true, y_pred = y[keep], pred[keep]
    else:
        y_true, y_pred = d["y_patch"], prob.argmax(-1)
    save_report(f"{a.model}_{a.split}", compute_metrics(y_true, y_pred))


if __name__ == "__main__":
    main()
