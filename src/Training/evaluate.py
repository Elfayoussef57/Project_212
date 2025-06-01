import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from data_analysis import data_generator
from sklearn.metrics import classification_report, confusion_matrix

# === 1. Charger le modèle ===
model = load_model('model_vgg.h5')

test_generator =  data_generator("../../data/Chest X_ray/test", batch_size=16, img_size=224)

# === 3. Évaluer le modèle ===
results = model.evaluate(test_generator)
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}")
print(f"Recall: {results[3]:.4f}")

# Prédictions
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)  # pour softmax à 2 neurones

# Vrais labels
y_true = test_generator.classes

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

# Matrice de confusion
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
