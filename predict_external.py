#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:37:48 2025
predict_external.py
@author: betulekren
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from tensorflow.keras.models import load_model
from config import MODEL_KERAS_PATH, IMG_SIZE, EXTERNAL_IMG_PATH

#Segmentation Function

def segmentation_LAB_features(img):
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a = img_lab[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(7, 7))
    clahe_img = clahe.apply(a)
    _, binary = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = np.zeros_like(imgrgb)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(imgrgb, mask)
    return result

# Read the image
img = cv2.imread(EXTERNAL_IMG_PATH)
segmented_img = segmentation_LAB_features(img)
resized_img = cv2.resize(segmented_img, IMG_SIZE)
normalized_img = resized_img / 255.0
normalized_img = np.expand_dims(normalized_img, axis=0)

# Load the model
model = load_model(MODEL_KERAS_PATH)

# Make a prediction
prediction = model.predict(normalized_img)

# Class labels
class_labels = [
   'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
   'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot',
   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
   'Tomato___healthy', 'Two-spotted_spider_mite'
]

# Result
pred_index = np.argmax(prediction)
confidence = prediction[0][pred_index]
predicted_class = class_labels[pred_index]

# Visualization
plt.figure(figsize=(10, 10))
plt.imshow(np.squeeze(normalized_img, axis=0))
plt.title(f"🔍 Prediction: {predicted_class}\n🎯 Güven: {confidence:.2%}")
plt.axis("off")
plt.show()

print(f" Predicted Label: {predicted_class}")
print(f" Confidence Score {confidence:.2%}")
