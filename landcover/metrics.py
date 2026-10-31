"""Evaluation helpers shared by the RF baseline, CNN and U-Net."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config


def compute_metrics(y_true, y_pred):
    labels = list(range(config.N_CLASSES))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=config.CLASS_NAMES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "per_class": {n: {k: float(rep[n][k]) for k in ("precision", "recall", "f1-score", "support")}
                      for n in config.CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, path, title="Confusion matrix (row-normalised)"):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(config.N_CLASSES), config.CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(config.N_CLASSES), config.CLASS_NAMES)
    for i in range(config.N_CLASSES):
        for j in range(config.N_CLASSES):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_report(name, metrics):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(metrics["confusion_matrix"], config.OUT_DIR / f"{name}_confusion_matrix.png",
                          title=f"{name}: confusion matrix (row-normalised)")
    print(f"\n{name}: overall accuracy = {metrics['overall_accuracy']:.4f}, macro F1 = {metrics['macro_f1']:.4f}")
    for n, v in metrics["per_class"].items():
        print(f"  {n:18s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1-score']:.3f} n={int(v['support'])}")
