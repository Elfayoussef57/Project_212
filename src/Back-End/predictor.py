# src/Back-End/predictor.py

import os
import numpy as np
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from model_cam import generate_gradcam, overlay_heatmap_on_image # type: ignore

IMG_SIZE = (128, 128)
MODEL_PATH = MODEL_PATH = os.path.join('..', '..', 'models', 'my_model_vggv2.h5')

def predict_and_heatmap(image_path, upload_folder, model):  # Ajouter le modèle en paramètre
    """
    Prédit la pneumonie et génère une heatmap Grad-CAM.
    Retourne (label, confidence, chemin_heatmap).
    """
    # Prétraitement
    img = load_img(image_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Prédiction (utilise le modèle passé en paramètre)
    prob = float(model.predict(arr)[0][0])
    label = 'Pneumonia' if prob > 0.5 else 'Not-Pneumonia'
    confidence = prob if prob > 0.5 else 1 - prob

    # Génération de la heatmap (utilise le même modèle)
    # Génération du chemin de sortie
    heatmap_filename = f"heatmap_{os.path.basename(image_path)}"
    heatmap_path = os.path.join(upload_folder, heatmap_filename)

    # Génération de la heatmap
    heatmap = generate_gradcam(model, arr, class_index=1, last_conv_layer="conv2d_10")
    overlay_heatmap_on_image(heatmap, image_path, save_path=heatmap_path)

    return label, confidence, heatmap_path
