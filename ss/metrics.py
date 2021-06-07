import tensorflow as tf
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
  """Sparse version of MeanIoU metric. See https://github.com/tensorflow/tensorflow/issues/32875."""

  def __init__(self, num_classes, name=None, dtype=None):
    super().__init__(num_classes=num_classes, name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.math.argmax(y_pred, axis=-1)
    return super().update_state(y_true, y_pred, sample_weight)