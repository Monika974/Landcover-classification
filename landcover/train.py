"""Train the patch CNN or the U-Net.

    python -m landcover.train --model cnn  --epochs 30
    python -m landcover.train --model unet --epochs 40
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from . import config
from .models import build_model, masked_accuracy, masked_sparse_ce


def load_split(name, kind):
    d = np.load(config.PROC_DIR / f"{name}.npz")
    X = d["X"].astype("float32")
    y = d["y_mask"][..., None] if kind == "unet" else d["y_patch"]
    return X, y


def class_weights(y):
    """Square-root-dampened inverse-frequency weights (ignores 255)."""
    y = y[y != config.IGNORE_INDEX].astype(np.int64)
    counts = np.bincount(y, minlength=config.N_CLASSES).astype("float64")
    w = np.where(counts > 0, (counts.sum() / (config.N_CLASSES * np.maximum(counts, 1))) ** 0.5, 0.0)
    return (w / w[w > 0].mean()).astype("float32")


def make_dataset(X, y, batch, training, kind):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=config.SEED)

        def augment(x, t):
            if kind == "unet":
                z = tf.concat([x, tf.cast(t, tf.float32)], axis=-1)
                z = tf.image.random_flip_up_down(tf.image.random_flip_left_right(z))
                return z[..., :-1], tf.cast(z[..., -1:], t.dtype)
            x = tf.image.random_flip_up_down(tf.image.random_flip_left_right(x))
            return x, t

        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def plot_history(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ("loss", "accuracy" if "accuracy" in hist else "masked_accuracy")):
        ax.plot(hist[key], label="train")
        ax.plot(hist[f"val_{key}"], label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["cnn", "unet"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUT_DIR.mkdir(exist_ok=True)

    Xtr, ytr = load_split("train", a.model)
    Xva, yva = load_split("val", a.model)
    cw = class_weights(ytr)
    print("class weights:", dict(zip(config.CLASS_NAMES, cw.round(2).tolist())))

    model = build_model(a.model)
    opt = tf.keras.optimizers.Adam(a.lr)
    if a.model == "unet":
        model.compile(opt, loss=masked_sparse_ce(cw), metrics=[masked_accuracy])
        fit_kw = {}
    else:
        model.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        fit_kw = {"class_weight": {i: float(w) for i, w in enumerate(cw)}}
    model.summary()

    ckpt = config.MODEL_DIR / f"{a.model}.keras"
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    h = model.fit(make_dataset(Xtr, ytr, a.batch, True, a.model),
                  validation_data=make_dataset(Xva, yva, a.batch, False, a.model),
                  epochs=a.epochs, callbacks=cbs, **fit_kw)
    hist = {k: [float(v) for v in vs] for k, vs in h.history.items()}
    (config.OUT_DIR / f"{a.model}_history.json").write_text(json.dumps(hist, indent=2))
    plot_history(hist, config.OUT_DIR / f"{a.model}_training_curves.png")
    print(f"best model saved to {ckpt}")


if __name__ == "__main__":
    main()
