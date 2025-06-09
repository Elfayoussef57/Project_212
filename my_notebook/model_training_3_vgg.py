import tensorflow as tf
from tensorflow.keras.models import load_model, Model #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix, classification_report

# 1. Chargement et préparation du modèle
base_model = load_model('../../models/model_vgg.h5')

# Correction de l'architecture si nécessaire
if base_model.layers[-1].output_shape[-1] == 2:  # Si dernière couche a 2 neurones
    # Création d'un nouveau modèle avec la bonne architecture
    x = base_model.layers[-2].output
    x = Dense(1, activation='sigmoid', name='new_dense_output')(x)  # Nouvelle couche de sortie
    model = Model(inputs=base_model.input, outputs=x)
else:
    model = base_model

# 2. Chemins des données
train_dir = '../Downloads/Project-212/data/train'
val_dir = '../Downloads/Project-212/data/val'

# 3. Data Augmentation optimisée
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 4. Création des générateurs
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Critique pour la classification binaire
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',  # Critique pour la classification binaire
    shuffle=False
)

# 5. Calcul des poids de classe
classes = train_generator.classes
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(classes),
    y=classes
)
class_weights = {0: 1.945, 1: 0.673}  # Valeurs que vous avez obtenues
print("Poids appliqués :", class_weights)

# 6. Fine-Tuning
for layer in model.layers[:-6]:
    layer.trainable = False
    
for layer in model.layers[-6:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# 7. Entraînement avec callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_vgg.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )
]

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks,
    workers=4
)

# 8. Évaluation
def evaluate_model():
    val_generator.reset()
    y_pred = (model.predict(val_generator) > 0.5).astype(int)
    y_true = val_generator.classes
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies étiquettes')
    plt.title('Matrice de Confusion')
    plt.show()
    
    # Rapport de classification
    print("\nRapport détaillé :")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))
    
    # Courbes d'apprentissage
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.show()

evaluate_model()

# 9. Fonction Grad-CAM
def grad_cam(model, img_path, layer_name='block5_conv3'):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[0]
    
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    heatmap = conv_outputs[0] @ weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Superposition
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    # Affichage
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()