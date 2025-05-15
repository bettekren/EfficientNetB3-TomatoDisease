#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:36:24 2025
predict_single.py
@author: betulekren
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from data_loader import get_data_generators
from config import MODEL_KERAS_PATH

# Load the data
train_gen, _, test_gen = get_data_generators()
model = load_model(MODEL_KERAS_PATH)

# Random test image
random_index = random.randint(0, test_gen.samples - 1)
test_gen.reset()
for i in range(random_index + 1):
    test_image_batch, test_label_batch = next(test_gen)

test_image = test_image_batch[0]
test_label = test_label_batch[0]

# Make a prediction
predictions = model.predict(np.expand_dims(test_image, axis=0))
predicted_index = np.argmax(predictions)
true_index = np.argmax(test_label)
class_labels = list(test_gen.class_indices.keys())

plt.imshow(test_image)
plt.axis("off")
plt.title(f"Real: {class_labels[true_index]}\nPrediction: {class_labels[predicted_index]} ({predictions[0][predicted_index]*100:.2f}%)")
plt.show()

print(f"Real Class: {class_labels[true_index]}")
print(f"Predicted Class {class_labels[predicted_index]}")
print(f"Probability Distribution: {predictions[0]}")
