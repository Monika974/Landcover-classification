"""Keras models: patch-level CNN (baseline) and a small U-Net (pixel-level)."""
import tensorflow as tf
from tensorflow.keras import Model, layers

from . import config


def _conv_block(x, filters):
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return x


def build_patch_cnn(input_shape=(config.PATCH_SIZE, config.PATCH_SIZE, config.N_FEATURES),
                    num_classes=config.N_CLASSES):
    """Whole-patch classifier: one label per 64x64 patch."""
    inp = layers.Input(input_shape)
    x = layers.BatchNormalization()(inp)
    for f in (32, 64, 128):
        x = _conv_block(x, f)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="patch_cnn")


def build_unet(input_shape=(config.PATCH_SIZE, config.PATCH_SIZE, config.N_FEATURES),
               num_classes=config.N_CLASSES, base=16):
    """Small U-Net: per-pixel classification, so area statistics are pixel-accurate."""
    inp = layers.Input(input_shape)
    x = layers.BatchNormalization()(inp)
    skips = []
    for mult in (1, 2, 4):
        x = _conv_block(x, base * mult)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
    x = _conv_block(x, base * 8)
    x = layers.Dropout(0.3)(x)
    for mult, skip in zip((4, 2, 1), reversed(skips)):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = _conv_block(x, base * mult)
    out = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return Model(inp, out, name="unet")


def build_model(kind, **kw):
    return {"cnn": build_patch_cnn, "unet": build_unet}[kind](**kw)


def masked_sparse_ce(class_weights):
    """Class-weighted cross-entropy that ignores pixels labelled IGNORE_INDEX."""
    w = tf.constant(class_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        y = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        valid = tf.cast(tf.not_equal(y, config.IGNORE_INDEX), tf.float32)
        y_safe = tf.where(tf.equal(y, config.IGNORE_INDEX), 0, y)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_safe, y_pred)
        wt = tf.gather(w, y_safe) * valid
        return tf.reduce_sum(ce * wt) / (tf.reduce_sum(wt) + 1e-7)

    return loss


def masked_accuracy(y_true, y_pred):
    y = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    valid = tf.cast(tf.not_equal(y, config.IGNORE_INDEX), tf.float32)
    correct = tf.cast(tf.equal(y, tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)), tf.float32)
    return tf.reduce_sum(correct * valid) / (tf.reduce_sum(valid) + 1e-7)
