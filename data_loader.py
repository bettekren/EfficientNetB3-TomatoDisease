#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:50:38 2025

@author: betulekren
"""

# data_loader.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import BATCH_SIZE, IMG_SIZE, TRAIN_DIR, VAL_DIR, TEST_DIR

def get_data_generators():
    tr_gen = ImageDataGenerator(rescale=1. / 255)
    val_gen = ImageDataGenerator(rescale=1. / 255)
    test_gen = ImageDataGenerator(rescale=1. / 255)

    train_gen = tr_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    valid_gen = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True
    )

    test_gen = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )

    return train_gen, valid_gen, test_gen
