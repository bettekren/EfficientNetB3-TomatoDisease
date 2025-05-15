#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:32:52 2025

@author: betulekren
"""

from data_utils import loading_the_data, custom_autopct
from visualization import model_performance, plot_confusion_matrix
from model_blocks import conv_block, dense_block
from data_loader import get_data_generators
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_KERAS_PATH, HISTORY_JSON
train_gen, valid_gen, test_gen = get_data_generators()


from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.applications import EfficientNetB3
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from collections import Counter
import seaborn as sns
import pandas as pd
import random
import os

# Model architecture
img_shape = (224, 224, 3)
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=img_shape, pooling=None)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = dense_block(128, 0.5)(x)
x = dense_block(32, 0.2)(x)

class_counts = len(train_gen.class_indices)
predictions = Dense(class_counts, activation="softmax")(x)

EfficientNetB3_model = Model(inputs=base_model.input, outputs=predictions)
EfficientNetB3_model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
EfficientNetB3_model.summary()

#Train
epochs = EPOCHS
EfficientNetB3_history = EfficientNetB3_model.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False)
model_performance(EfficientNetB3_history, epochs)

# Print the results
final_train_acc = EfficientNetB3_history.history['accuracy'][-1]
final_val_acc = EfficientNetB3_history.history['val_accuracy'][-1]
final_train_loss = EfficientNetB3_history.history['loss'][-1]
final_val_loss = EfficientNetB3_history.history['val_loss'][-1]

print("\nFinal Model Performance")
print("=" * 40)
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print("=" * 40)

# Save the training history as a JSON file
history_dict = EfficientNetB3_history.history
with open("EfficientNetB3_historyLastUseIt.json", 'w') as f:
    json.dump(history_dict, f, indent=4)

EfficientNetB3_model.save(MODEL_KERAS_PATH)




