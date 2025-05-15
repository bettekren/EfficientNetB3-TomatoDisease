#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:36:27 2025
config.py

@author: betulekren
"""
#Training configuration constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

#Data directories
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "/dataset/test"
EXTERNAL_IMG_PATH = "external_images/bacterial.jpeg"

#Model and output filenames
MODEL_NAME = "EfficientNetB3_model"
MODEL_KERAS_PATH = f"EfficientNetB3_model.keras"
HISTORY_JSON = f"EfficientNetB3_historyLastUseIt.json"
TEST_RESULTS_JSON = f"EfficientB3LastTest_results.json"
GRADCAM_LAYER = "top_conv"  # Name of the last layer to be used for Grad-CAM
