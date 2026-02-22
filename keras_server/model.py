import numpy as np
import tensorflow as tf
from tensorflow import keras


class DominoModel:
    """Keras model for domino AI with action masking and GPU acceleration."""

    def __init__(self, input_dim=126, hidden_dims=None, output_dim=56):
        if hidden_dims is None:
            hidden_dims = [150]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.model = self._build_model()
        self._optimizer = None
        self._optimizer_lr = None

    def _get_optimizer(self, learning_rate):
        if self._optimizer is None or self._optimizer_lr != learning_rate:
            self._optimizer = keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)
            self._optimizer_lr = learning_rate
        return self._optimizer

    def _build_model(self):
        inputs = keras.Input(shape=(self.input_dim,), name="features")
        x = inputs
        for i, dim in enumerate(self.hidden_dims):
            x = keras.layers.Dense(
                dim,
                activation="relu",
                kernel_initializer="glorot_uniform",
                bias_initializer="glorot_uniform",
                name=f"hidden_{i}",
            )(x)
        outputs = keras.layers.Dense(
            self.output_dim,
            activation="linear",
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            name="output",
        )(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    def predict(self, features, valid_mask):
        """Predict the best action given features and a valid action mask.

        Args:
            features: numpy array of shape (126,)
            valid_mask: numpy array of shape (56,) with 1.0 for valid actions

        Returns:
            card index (0-27), side string ("left" or "right"), confidence float
        """
        features_batch = np.expand_dims(features, axis=0)
        raw_output = self.model(features_batch, training=False).numpy()[0]

        # Mask invalid actions with large negative value
        masked_output = np.where(valid_mask > 0, raw_output, -1e9)
        best_idx = int(np.argmax(masked_output))
        confidence = float(masked_output[best_idx])

        if best_idx >= 28:
            card = best_idx - 28
            side = "right"
        else:
            card = best_idx
            side = "left"

        return card, side, confidence

    def train_batch(self, features_batch, target_batch, mask_batch, learning_rate=0.001):
        """Train on a batch with action masking.

        Only updates loss for the action that was actually taken (mask=1.0).

        Args:
            features_batch: numpy array of shape (N, 126)
            target_batch: numpy array of shape (N, 56) - reward at action index
            mask_batch: numpy array of shape (N, 56) - 1.0 at taken action
            learning_rate: float

        Returns:
            average loss (float)
        """
        optimizer = self._get_optimizer(learning_rate)
        features_tensor = tf.constant(features_batch, dtype=tf.float32)
        target_tensor = tf.constant(target_batch, dtype=tf.float32)
        mask_tensor = tf.constant(mask_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            predictions = self.model(features_tensor, training=True)
            # MSE only on the masked (taken) actions
            diff = predictions - target_tensor
            masked_diff = diff * mask_tensor
            loss = tf.reduce_mean(tf.square(masked_diff))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return float(loss.numpy())

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)
