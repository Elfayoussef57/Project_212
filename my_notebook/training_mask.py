import pandas as pd
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Configuration ====
image_dir = "../data/Mask/images"        # Folder with images
csv_path = "../data/Mask/jsrt_metadata.csv"          # Path to your CSV
image_size = 224                         # Resize images to this size

# ==== Load and Prepare Labels ====
labels_df = pd.read_csv(csv_path)

# Generate the actual filenames (assumes .jpg extension)
labels_df['filename'] = labels_df['study_id'].astype(str)

# Binary label: 1 if diagnosis is 'Nodule', else 0
labels_df['label'] = labels_df['diagnosis'].apply(lambda x: 1 if str(x).lower() == 'nodule' else 0)

# ==== Load Images ====
def load_images_and_labels(image_dir, labels_df):
    X, y = [], []
    for idx, row in labels_df.iterrows():
        image_path = os.path.join(image_dir, row['filename'])
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = cv2.resize(img, (image_size, image_size))
            img = img / 255.0  # normalize
            X.append(img)
            y.append(row['label'])
        else:
            print(f"Warning: {image_path} not found.")
    return np.array(X), np.array(y)

X, y = load_images_and_labels(image_dir, labels_df)

# ==== Split Data ====
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Build Model ====
model = models.Sequential([
    layers.Input(shape=(image_size, image_size, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==== Train Model ====
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

model.save('../models/mask_model.h5')  # Save the model
# ==== Evaluate Model ====
print("\nEvaluating model on validation set:")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")

# ==== Predict & Report ====
y_pred = (model.predict(X_val) > 0.5).astype("int32").flatten()

print("\nClassification Report:")
print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred,
    labels=[0, 1],
    target_names=['No Nodule', 'Nodule']
))

# ==== Confusion Matrix ====
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Nodule', 'Nodule'], yticklabels=['No Nodule', 'Nodule'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# ==== Plot Training History ====
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train")
plt.plot(history.history['val_accuracy'], label="Val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train")
plt.plot(history.history['val_loss'], label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
