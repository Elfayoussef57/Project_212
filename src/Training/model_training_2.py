from tensorflow.keras.models import load_model # type: ignore
from data_analysis import data_generator
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
import matplotlib.pyplot as plt

def get_callbacks(filepath="model_vgg.h5"):
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
    plt.savefig("/output/Resultat_model_2.jpg")
    plt.close()



model = load_model("model_vgg.h5")

# Geler tout d'abord
for layer in model.layers:
    layer.trainable = False

# Définir trainable=True seulement sur les dernières couches que tu veux affiner
for layer in model.layers[-4:]:  # par exemple, les 4 dernières couches
    layer.trainable = True


model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", "Precision", "Recall"]
)

# Chargement des données
train_generator = data_generator("../../data/Chest X_ray/train", batch_size=16, img_size=224, shuffle=True)
val_generator = data_generator("../../data/Chest X_ray/val", batch_size=16, img_size=224, shuffle=False)
test_generator = data_generator("../../data/Chest X_ray/test", batch_size=16, img_size=224, shuffle=False)

# Class weights
classes = train_generator.classes
weights = class_weight.compute_class_weight('balanced', classes=np.unique(classes), y=classes)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,                # nombre total souhaité
    initial_epoch=10,         # si tu avais déjà fait 10 epochs
    class_weight=class_weights,
    callbacks=get_callbacks("../../models/model_vgg.h5")
)

# Sauvegarder le modèle
model.save("../../models/model_vgg.h5")

# Afficher les courbes d'apprentissage
plot_training(history)




