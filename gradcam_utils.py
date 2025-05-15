#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:34:38 2025

@author: betulekren
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
from config import EXTERNAL_IMG_PATH, MODEL_KERAS_PATH



# Input dimensions of the model
IMG_H, IMG_W = 224, 224 


# 📌 **Segmentation Function (Lab Color Space and Otsu Thresholding)**
def segmentation_LAB_features(img):
 
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    a = img_lab[:, :, 1]  # 'a' kanalını al

    clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(7, 7))
    clahe_img = clahe.apply(a)



    _, binary = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
 
    mask = np.zeros_like(imgrgb)

  
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(imgrgb, mask)
    
    return result


img = cv2.imread(EXTERNAL_IMG_PATH) 
segmented_img = segmentation_LAB_features(img)  


resized_img = cv2.resize(segmented_img, (IMG_H, IMG_W)) 
normalized_img = resized_img / 255.0  #Normalization
normalized_img = np.expand_dims(normalized_img, axis=0) 


model = load_model(MODEL_KERAS_PATH) 
prediction = model.predict(normalized_img)


class_labels = [
   'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
   'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Target_Spot',
   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
   'Tomato___healthy', 'Two-spotted_spider_mite'
]


pred_index = np.argmax(prediction)  
confidence = prediction[0][pred_index] 
predicted_class = class_labels[pred_index] 

# Display the normalized image after segmentation
plt.figure(figsize=(10, 10))

normalized_display_img = np.squeeze(normalized_img, axis=0)


plt.imshow(normalized_display_img)
plt.title(f"🔍 Prediction {predicted_class}\n🎯Confidence: {confidence:.2%}")
plt.axis("off")
plt.show()

print(f"Predicted Label: {predicted_class}")
print(f"Güven Skoru: {confidence:.2%}")



#To view the model architecture
model.summary()

#Grad-CAM

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_gradcam(model, img_array, class_index, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]


    grads = tape.gradient(loss, conv_outputs)

 
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed

#  The target class in the model
class_idx = np.argmax(prediction)

# Generate Grad-CAM heatmap
heatmap = get_gradcam(model, normalized_img, class_idx, last_conv_layer_name="top_conv")

# Overlay the heatmap on the original image
original_img = cv2.imread(EXTERNAL_IMG_PATH)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
original_img_resized = cv2.resize(original_img, (IMG_W, IMG_H))

overlay_img = overlay_heatmap(heatmap, original_img_resized)


plt.figure(figsize=(10, 10))
plt.imshow(overlay_img)
plt.title(f"Grad-CAM: {predicted_class} ({confidence:.2%})")
plt.axis("off")
plt.show()
