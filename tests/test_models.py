import numpy as np
import pytest

pytest.importorskip("tensorflow")

from landcover import config  # noqa: E402
from landcover.models import build_patch_cnn, build_unet, masked_accuracy, masked_sparse_ce  # noqa: E402


def test_output_shapes():
    x = np.random.rand(2, 64, 64, config.N_FEATURES).astype("float32")
    assert build_patch_cnn().predict(x, verbose=0).shape == (2, config.N_CLASSES)
    assert build_unet().predict(x, verbose=0).shape == (2, 64, 64, config.N_CLASSES)


def test_masked_loss_ignores_255():
    import tensorflow as tf
    y_pred = tf.constant(np.full((1, 2, 2, config.N_CLASSES), 1 / config.N_CLASSES, "float32"))
    y_true = tf.constant(np.array([[[[255], [255]], [[255], [255]]]], "uint8"))
    loss = masked_sparse_ce(np.ones(config.N_CLASSES, "float32"))(y_true, y_pred)
    assert float(loss) == 0.0
    assert float(masked_accuracy(y_true, y_pred)) == 0.0
