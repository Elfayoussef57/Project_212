import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras import layers, models, regularizers, mixed_precision # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-5

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Alloue la mémoire GPU de manière progressive
    except RuntimeError as e:
        print(e)

DICOM_DATA_DIR = "../data/RSNA/stage_2_train_images"
LABELS_CSV = "../data/RSNA/stage_2_train_labels.csv"
PNG_OUTPUT_DIR = "../data/RSNA/rsna_pneumonia_png_images"

os.makedirs(PNG_OUTPUT_DIR, exist_ok=True)

# --- DICOM to PNG conversion ---
df_labels = pd.read_csv(LABELS_CSV)
patient_ids = df_labels['patientId'].unique()

print(f"{len(patient_ids)} DICOM files are being converted to PNG...")

for i, patient_id in enumerate(patient_ids):
    dicom_path = os.path.join(DICOM_DATA_DIR, patient_id + ".dcm")
    output_path = os.path.join(PNG_OUTPUT_DIR, patient_id + ".png")

    if os.path.exists(output_path):
        continue

    try:
        dicom = pydicom.dcmread(dicom_path)
        img = dicom.pixel_array.astype(np.float32)

        if 'WindowCenter' in dicom and 'WindowWidth' in dicom:
            window_center = dicom.WindowCenter
            window_width = dicom.WindowWidth
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = window_center[0]
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = window_width[0]

            min_val = window_center - window_width / 2
            max_val = window_center + window_width / 2

            img = np.clip(img, min_val, max_val)
            img = ((img - min_val) / (max_val - min_val + 1e-5)) * 255
        else:
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * 255

        img = img.astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(output_path, img)

    except Exception as e:
        print(f"Error: An error occurred while converting {patient_id}.dcm: {e}")
        continue

    if (i + 1) % 1000 == 0:
        print(f"{i + 1} images converted.")

print("All DICOM files have been converted to PNG.")

# --- Prepare the dataset ---
df_labels['filename'] = df_labels['patientId'] + ".png"
df_labels['Target'] = df_labels['Target'].astype(str)

train_val_df, test_df = train_test_split(df_labels, test_size=0.2, stratify=df_labels["Target"], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1, stratify=train_val_df["Target"], random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_df["Target"]), y=train_df["Target"])
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# --- TF Dataset function ---
def process_path(filename, label):
    img_path = tf.strings.join([PNG_OUTPUT_DIR, "/", filename])
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)  # Grayscale
    img = tf.image.grayscale_to_rgb(img)     # Convert to 3 channels
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]

    # Data augmentation (only for training)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_zoom(img, (0.85, 1.15)) if hasattr(tf.image, "random_zoom") else img
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    return img, label

def process_path_no_aug(filename, label):
    img_path = tf.strings.join([PNG_OUTPUT_DIR, "/", filename])
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, label

# Convert labels to integers
train_labels = train_df['Target'].astype(int).values
val_labels = val_df['Target'].astype(int).values
test_labels = test_df['Target'].astype(int).values

train_ds = tf.data.Dataset.from_tensor_slices((train_df['filename'].values, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_df['filename'].values, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_df['filename'].values, test_labels))

train_ds = train_ds.shuffle(1000).map(process_path, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(process_path_no_aug, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(process_path_no_aug, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_vgg16_from_scratch(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    mixed_precision.set_global_policy('mixed_float16')
    l2_reg = regularizers.l2(0.0005)

    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg, input_shape=input_shape))
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 2
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 3
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 4
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    # Block 5
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2_reg))
    model.add(layers.MaxPooling2D((2,2), strides=(2,2)))

    model.add(layers.Flatten())

    # Déplacer les couches denses sur le CPU
    with tf.device('/CPU:0'):
        model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu', kernel_regularizer=l2_reg))
        model.add(layers.Dropout(0.5))

    #  La dernière couche sur GPU
    model.add(layers.Dense(1, activation='sigmoid', dtype='float32'))

    return model

model = build_vgg16_from_scratch()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

# Model
model.summary()

# --- Modeli fit ---
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

model.save("my_model_vggv2.h5")

def plot_history(hist):
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(14,5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.show()

plot_history(history)

y_true = []
y_pred_probs = []

# Dataset'ten verileri çek
for batch in val_ds:
    X_batch, y_batch = batch
    y_true.extend(y_batch.numpy())  # Gerçek etiketleri topla
    preds = model.predict(X_batch, verbose=0)  # Tahmin olasılıkları
    y_pred_probs.extend(preds)

# Listeyi numpy dizisine çevir
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

# Metrikleri hesapla
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_probs)

print(f"🔹 Accuracy     : {acc:.4f}")
print(f"🔹 Precision    : {prec:.4f}")
print(f"🔹 Recall       : {rec:.4f}")
print(f"🔹 F1-Score     : {f1:.4f}")
print(f"🔹 ROC AUC      : {roc_auc:.4f}")

# Récupérer les valeurs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Créer la figure
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r--', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Sauvegarder la figure
plt.tight_layout()
plt.savefig("training_metrics.png", dpi=300)  # Enregistrement au format PNG
plt.show()

# Calcule la matrice de confusion
cm = confusion_matrix(y_true, y_pred)

# Affichage avec seaborn pour un graphique
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonie'], yticklabels=['Normal', 'Pneumonie'])
plt.xlabel('Prédiction')
plt.ylabel('Vérité terrain')
plt.title('Matrice de Confusion')
plt.show()

