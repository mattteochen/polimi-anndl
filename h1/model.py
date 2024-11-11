# file: model.py

import numpy as np

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

class Model:
    def __init__(self):
        """Initializes the model's internal state."""
        self.neural_network = tfk.models.load_model('KaggleEfficientNetV2L80.07.keras')
    
    def predict(self, X):
        """Returns a numpy array of labels for the given input X."""
        preds = self.neural_network.predict(X)
        if len(preds.shape) == 2:
            preds = np.argmax(preds, axis=1)
        return preds

