#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:56:43 2025

@author: betulekren
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

sns.set_style('darkgrid')

def model_performance(history):
    """Displays the model's training history as a graph"""
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(tr_acc) + 1)

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, tr_loss, 'r', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'g', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, tr_acc, 'r', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'g', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(test_gen, y_pred, normalize=False):
    """Plots the confusion matrix for the model's predictions"""
    classes = list(test_gen.class_indices.keys())
    cm = confusion_matrix(test_gen.classes, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix' if normalize else 'Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10, ha='right')
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
        plt.text(j, i, value,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    

def plot_saved_history(json_file):
    with open(json_file, 'r') as f:
     history = json.load(f)
    
    epochs = range(1, len(history['accuracy']) + 1)

    # 🔹 Loss Grafiği
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'r', label="Training Loss")
    plt.plot(epochs, history['val_loss'], 'g', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # 🔹 Accuracy Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'r', label="Training Accuracy")
    plt.plot(epochs, history['val_accuracy'], 'g', label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.show()


