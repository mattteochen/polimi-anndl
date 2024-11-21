# file: model.py
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import keras_cv

class Model:
    # Applying TTA
    def custom_aug(self, X):
        augmix = keras_cv.layers.AugMix([0, 255])
        rand_aug = keras_cv.layers.RandAugment(value_range=(0, 255), augmentations_per_image=3, magnitude=0.5)
        
        randaug_applied = rand_aug(X).numpy() 
        augmix_applied = augmix(X).numpy() 
    
        # Stack the original image and its augmented versions (RandAugment, AugMix)
        return np.stack([X, randaug_applied, augmix_applied], axis=1)
    
    def __init__(self):
        self.neural_network = tfk.models.load_model('efficientnetv2-l-finetuned5blocks-98.66-241121_1144.keras')
    
    def predict(self, X):
        # Augment the input images using custom_aug()
        augmented_images = self.custom_aug(X)

        # Initialize a list to collect the predictions for each image's augmented versions
        predictions = []

        # Loop through each image (in the batch) and get predictions for all 3 augmented versions
        for i in range(X.shape[0]):
            # Extract the 3 augmented versions of the i-th image
            triplet = augmented_images[i]  # Shape: (3, 96, 96, 3)

            # Get predictions for all 3 versions
            triplet_preds = self.neural_network.predict(triplet)  # Shape: (3, n_classes)

            # Average the predictions for the 3 augmented versions
            averaged_preds = np.mean(triplet_preds, axis=0)

            predictions.append(averaged_preds)

        # Convert the list of averaged predictions to a numpy array
        predictions = np.array(predictions)

        # If the model has a softmax output (probabilities), we may want to return the class index
        if len(predictions.shape) == 2:
            predictions = np.argmax(predictions, axis=1)

        return predictions