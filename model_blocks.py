#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:57:39 2025
model_blocks.py
@author: betulekren
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D


def conv_block(filters, act='relu'):
    
    block = Sequential()
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(BatchNormalization())
    block.add(MaxPooling2D())
    
    return block

def dense_block(units, dropout_rate, act='relu'):
    def block(x):
        x = Dense(units, activation=act)(x)  # Add Dense Layer
        x = BatchNormalization()(x)  # Add BatchNormalization
        x = Dropout(dropout_rate)(x)  # Add Dropout 
        return x
    return block