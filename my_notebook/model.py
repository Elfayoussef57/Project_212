import numpy as np
import pydicom
import pandas as pd
import os
import csv
import random
import shutil
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Rectangle
from skimage import transform
from tensorflow.keras import layers, models, callbacks, losses, optimizers #type: ignore

# Configuration des chemins
OUTPUT_DIR = "../output/"
MODEL_DIR = "../models/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Fonctions utilitaires
def load_pneumonia_locations():
    pneumonia_locations = {}
    with open(os.path.join('../data/RSNA/stage_2_train_labels.csv'), mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None)
        for rows in reader:
            filename = rows[0]
            location = rows[1:5]
            pneumonia = rows[5]
            if pneumonia == '1':
                location = [int(float(i)) for i in location]
                if filename in pneumonia_locations:
                    pneumonia_locations[filename].append(location)
                else:
                    pneumonia_locations[filename] = [location]
    return pneumonia_locations

# Classes personnalisées
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=8, image_size=128, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def __load__(self, filename):
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        msk = np.zeros(img.shape, dtype=np.float32)
        filename_key = filename.split('.')[0]
        
        if self.pneumonia_locations and filename_key in self.pneumonia_locations:
            for location in self.pneumonia_locations[filename_key]:
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        
        img = transform.resize(img, (self.image_size, self.image_size), mode='reflect', anti_aliasing=True)
        msk = transform.resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        msk = msk.astype(np.float32)
        
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        
        return np.expand_dims(img, -1), np.expand_dims(msk, -1)

    def __loadpredict__(self, filename):
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        img = transform.resize(img, (self.image_size, self.image_size), mode='reflect', anti_aliasing=True)
        return np.expand_dims(img, -1)

    def __getitem__(self, index):
        batch_files = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        
        if self.predict:
            imgs = [self.__loadpredict__(f) for f in batch_files]
            return np.array(imgs), batch_files
        else:
            items = [self.__load__(f) for f in batch_files]
            imgs, msks = zip(*items)
            return np.array(imgs), np.array(msks)

    def __len__(self):
        return int(np.ceil(len(self.filenames) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

# Architecture du modèle optimisée pour MX330
def create_downsample(channels, inputs):
    x = layers.BatchNormalization(momentum=0.9)(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    return layers.MaxPool2D(2)(x)

def create_resblock(channels, inputs):
    x = layers.BatchNormalization(momentum=0.9)(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=3):
    inputs = layers.Input(shape=(input_size, input_size, 1))
    x = layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    
    for d in range(depth):
        channels *= 2
        x = create_downsample(channels, x)
        for _ in range(n_blocks):
            x = create_resblock(channels, x)
    
    x = layers.BatchNormalization(momentum=0.9)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = layers.UpSampling2D(2**depth)(x)
    return models.Model(inputs=inputs, outputs=outputs)

# Fonctions de perte
def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

def iou_bce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    return 0.5 * losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones_like(intersect)
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

# Fonction principale
def main():
    # Configuration pour GPU limité
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1536)])
        except RuntimeError as e:
            print(e)
    
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Précision mixte activée avec succès")
    except ValueError as e:
        print("Précision mixte non supportée:", e)
        tf.keras.mixed_precision.set_global_policy('float32')
    
    # Chargement des données
    pneumonia_locations = load_pneumonia_locations()
    dicom_dir = '../data/RSNA/stage_2_train_images/'
    filenames = os.listdir(dicom_dir)
    random.shuffle(filenames)
    
    # Split des données
    n_valid_samples = 1024
    train_filenames = filenames[n_valid_samples:]
    valid_filenames = filenames[:n_valid_samples]
    print(f'Train samples: {len(train_filenames)}, Valid samples: {len(valid_filenames)}')

    # Création du générateur de données
    train_gen = DataGenerator(
        dicom_dir, 
        train_filenames, 
        pneumonia_locations, 
        batch_size=8,
        image_size=128,
        shuffle=True, 
        augment=True
    )
    
    valid_gen = DataGenerator(
        dicom_dir, 
        valid_filenames, 
        pneumonia_locations, 
        batch_size=8,
        image_size=128,
        shuffle=False
    )

    # Initialisation du modèle
    model = create_network(
        input_size=128,
        channels=16,
        n_blocks=2, 
        depth=3
    )
    
    model.summary()
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=iou_bce_loss,
        metrics=['accuracy', mean_iou]
    )

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'pneumonia_model_mx330.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Entraînement
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=10,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    # Sauvegarde de l'historique
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(OUTPUT_DIR, 'training_history.csv'), index=False)
    
    # ===================================================================
    # ÉVALUATION ET VISUALISATION (NOUVEAU)
    # ===================================================================
    
    # 1. Chargement du meilleur modèle
    model.load_weights(os.path.join(MODEL_DIR, 'pneumonia_model_mx330.h5'))
    
    # 2. Évaluation finale
    print("\nÉvaluation finale sur l'ensemble de validation:")
    evaluation = model.evaluate(valid_gen)
    print(f"Loss: {evaluation[0]:.4f}, Accuracy: {evaluation[1]:.4f}, IoU: {evaluation[2]:.4f}")
    
    # 3. Visualisation des courbes d'apprentissage
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Évolution de la Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Évolution de l\'Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # IoU
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mean_iou'], label='Train')
    plt.plot(history.history['val_mean_iou'], label='Validation')
    plt.title('Évolution du IoU')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
    plt.close()
    
    # 4. Visualisation des prédictions
    sample_gen = DataGenerator(
        dicom_dir, 
        valid_filenames[:16], 
        pneumonia_locations, 
        batch_size=16,
        image_size=128,
        shuffle=False
    )
    imgs, true_masks = sample_gen[0]
    pred_masks = model.predict(imgs)
    
    plt.figure(figsize=(20, 20))
    for i in range(16):
        # Image originale
        plt.subplot(4, 8, 2*i + 1)
        plt.imshow(imgs[i].squeeze(), cmap='bone')
        plt.imshow(true_masks[i].squeeze(), alpha=0.3, cmap='Reds')
        plt.title('Vérité terrain')
        plt.axis('off')
        
        # Prédiction
        plt.subplot(4, 8, 2*i + 2)
        plt.imshow(imgs[i].squeeze(), cmap='bone')
        plt.imshow(pred_masks[i].squeeze(), alpha=0.3, cmap='Reds')
        plt.title('Prédiction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_comparison.png'))
    plt.close()
    
    # 5. Matrice IoU par classe
    thresholds = np.linspace(0.1, 0.9, 9)
    iou_scores = []
    
    for thresh in thresholds:
        preds = (pred_masks > thresh).astype(np.float32)
        iou = np.mean([mean_iou(true_masks[i:i+1], preds[i:i+1]) for i in range(len(preds))])
        iou_scores.append(iou)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, iou_scores, 'o-')
    plt.title('IoU en fonction du seuil de décision')
    plt.xlabel('Seuil de décision')
    plt.ylabel('IoU moyen')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'iou_thresholds.png'))
    plt.close()
    
    print("Évaluation et visualisation terminées. Résultats sauvegardés dans", OUTPUT_DIR)

if __name__ == "__main__":
    main()