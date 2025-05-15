#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 13:35:51 2025
evaluate_model.py
@author: betulekren
"""



import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from config import MODEL_KERAS_PATH, TEST_RESULTS_JSON, HISTORY_JSON
from data_loader import get_data_generators
from visualization import plot_saved_history
from config import MODEL_KERAS_PATH, TEST_RESULTS_JSON, HISTORY_JSON



# Load the model and the data
model = load_model(MODEL_KERAS_PATH)
train_gen, valid_gen, test_gen = get_data_generators()

# Plot the training history
plot_saved_history(HISTORY_JSON)

with open(HISTORY_JSON, 'r') as f:
   history_dict = json.load(f)
   
# Check the data
print(HISTORY_JSON)

# Extract accuracy and validation accuracy values from the training history

train_acc = history_dict['accuracy']
val_acc =  history_dict['val_accuracy']
train_loss =  history_dict['loss']
val_loss =  history_dict['val_loss']

# Print accuracy and val_accuracy values for all epochs
for i in range(len(train_acc)):
    print(f"Epoch {i+1}:")
    print(f" - Training Accuracy: {train_acc[i]:.4f}")
    print(f" - Validation Accuracy: {val_acc[i]:.4f}")
    print(f" - Training Loss: {train_loss[i]:.4f}")
    print(f" - Validation Loss: {val_loss[i]:.4f}")
    print("-" * 40)


final_train_acc = history_dict['accuracy'][-1]
final_val_acc = history_dict['val_accuracy'][-1]

    
# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_gen, steps=len(test_gen), verbose=1)
print(f"\n Test Loss: {test_loss:.4f}")
print(f" Test Accuracy: {test_acc:.4f}")

# Save test results as a JSON file
results = {
    "Test Loss": round(test_loss, 4),
    "Test Accuracy": round(test_acc, 4),
    "Total Test Samples": test_gen.samples,
    "Batch Size": test_gen.batch_size,
    "Total Batches": len(test_gen),
    "Training Accuracy": round(final_train_acc, 4),
    "Validation Accuracy": round(final_val_acc, 4)
}

with open(TEST_RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=4)
print(f"Test results have been saved as '{TEST_RESULTS_JSON}'")


# Read the JSON file
with open("EfficientB3LastTest_results.json", "r") as f:
    loaded_test_results = json.load(f)

# Print the results
print("\nLoaded Test Results:")
for key, value in loaded_test_results.items():
    print(f" {key}: {value}")
    

# Make a prediction on a random test sample
random_index = random.randint(0, test_gen.samples - 1)
test_gen.reset()
for _ in range(random_index + 1):
    test_image_batch, test_label_batch = next(test_gen)

test_image = test_image_batch[0]
test_label = test_label_batch[0]

predictions = model.predict(np.expand_dims(test_image, axis=0))
predicted_class_index = np.argmax(predictions)
predicted_confidence = predictions[0][predicted_class_index] * 100

predicted_class_name = list(test_gen.class_indices.keys())[predicted_class_index]
true_class_index = np.argmax(test_label)
true_class_name = list(test_gen.class_indices.keys())[true_class_index]

# Visualize the prediction result

plt.imshow(test_image)
plt.axis("off")
plt.title(f"Real: {true_class_name}\nPrediction: {predicted_class_name} ({predicted_confidence:.2f}%)")
plt.show()


