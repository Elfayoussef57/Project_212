import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D,Dropout, GlobalAveragePooling2D, BatchNormalization # type: ignore
from tensorflow.keras.applications.vgg19 import VGG19 # type: ignore
from tensorflow.keras.optimizers import SGD, RMSprop, Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from data_pipeline import get_data
from data_analysis import data_generator
from tensorflow.keras.regularizers import l2 # type: ignore
import os
from data_analysis import data_generator
import numpy as np
import matplotlib.pyplot as plt

def build_model(input_shape: tuple) -> Model:
    # Charge VGG19 pré-entraîné
    base_model = VGG19(input_shape = (128,128,3),
                     include_top = False,
                     weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    flat = Flatten()(x)


    class_1 = Dense(4608, activation = 'relu')(flat)
    dropout = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation = 'relu')(dropout)
    output = Dense(2, activation = 'softmax')(class_2)

    model = Model(base_model.input, output)

    # Compilation avec learning rate adaptatif
    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )

    return model

# Callbacks recommandés
def get_callbacks():
    filepath = "model.h5"
    es = EarlyStopping(monitor="val_loss", verbose=1, mode="min", patience=4)
    cp=ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, save_weights_only=False,mode="auto", save_freq="epoch")
    lrr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)
    return [es, cp, lrr]

def plot_training(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.close()

def main():

    # Utilisation
    model = build_model(input_shape=(128, 128, 3))
    model.summary()

    # Chargement des données
    train_generator = data_generator("../data/train")
    val_generator = data_generator("../data/val")


    # Entraînement avec data augmentation (exemple)
    history = model.fit(
        train_generator,
        steps_per_epoch=50,
        validation_data=val_generator,
        epochs=50,
        callbacks=get_callbacks(),
    )

    plot_training(history)


if __name__ == "__main__":
    main()