import tensorflow as tf
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 2. Chemins des données
train_dir = '../../data/Chest X_ray/train'
val_dir = '../../data/Chest X_ray/train'

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

# 8. Évaluation
def evaluate_model():
    # Chargement du modèle
    model = load_model('../../models/model_vgg.h5')

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

evaluate_model()