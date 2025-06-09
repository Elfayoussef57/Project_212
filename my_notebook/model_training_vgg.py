import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense,Dropout, GlobalAveragePooling2D, BatchNormalization # type: ignore
from tensorflow.keras.applications.vgg19 import VGG19 # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from data_analysis import data_generator
from tensorflow.keras.regularizers import l2 # type: ignore
from data_analysis import data_generator
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras import backend as K # type: ignore

def build_model() -> Model:
    # Charge VGG19 pré-entraîné
    base_model = VGG19(input_shape = (224,224,3),
                     include_top = False,
                     weights = 'imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    class_1 = Dense(256, activation='relu', kernel_regularizer=l2(1e-3))(x)
    dropout = Dropout(0.5)(class_1)
    class_2 = Dense(64, activation='relu', kernel_regularizer=l2(1e-3))(dropout)
    output = Dense(2, activation = 'softmax')(class_2)

    model = Model(inputs=base_model.input, outputs=output)
    # Optimiseur
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )

    return model

# Fonction pour fusionner les historiques
def merge_histories(h1, h2):
    for key in h1.history:
        h1.history[key] += h2.history[key]
    return h1

# Callbacks recommandés
def get_callbacks(filepath="../../models/model_vgg.h5"):
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    cp = ModelCheckpoint(
        filepath, monitor='val_loss', save_best_only=True,
        save_weights_only=False, mode='auto'
    )
    lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2,
        verbose=1, min_lr=1e-6
    )
    return [es, cp, lr]

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
    model = build_model()
    model.summary()

    # Chargement des données
    train_generator = data_generator("../../data/Chest X_ray/train", batch_size=16, img_size=224, shuffle=True)
    val_generator = data_generator("../../data/Chest X_ray/val", batch_size=16, img_size=224, shuffle=False)
    test_generator = data_generator(".../../data/Chest X_ray//test", batch_size=16, img_size=224, shuffle=False)

        # Class weights
    classes = train_generator.classes
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    class_weights = dict(enumerate(weights))
    print("Class Weights:", class_weights)


    # Entraînement avec data augmentation (exemple)
    history = model.fit(train_generator,
            epochs=10, 
            callbacks=get_callbacks("model_vgg.h5"),
            validation_data=val_generator,
            class_weight=class_weights
            )
    
    base_model = model.layers[0]
    base_model.trainable = True

    model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'Precision', 'Recall']
    )

    # Reprendre l'entraînement
    history_fine = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=get_callbacks("model_vgg.h5"),
        class_weight=class_weights
    )

    # Fusionner les historiques
    history_final = merge_histories(history, history_fine)
    plot_training(history_final)

    # Evaluate on test set
    test_metrics = model.evaluate(test_generator)
    print("Test set metrics:", dict(zip(model.metrics_names, test_metrics)))

    # Sauvegarde du modèle
    model.save("model_vgg.h5")

    K.clear_session() # Clear the session to free up resources


if __name__ == "__main__":
    main()
