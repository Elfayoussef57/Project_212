import sys
import os
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model #type: ignore
from tensorflow.keras.preprocessing import image #type: ignore
import numpy as np
from PIL import Image
from flask_cors import CORS
from flask import send_from_directory
from urllib.parse import unquote
import uuid
import tensorflow as tf
import gc


# Configuration des chemins
BASE_DIR = os.path.abspath(r'C:\Users\ASUS\Pneumonia_Project')
NOTEBOOK_DIR = os.path.join(BASE_DIR, 'my_notebook')

if NOTEBOOK_DIR not in sys.path:
    sys.path.append(NOTEBOOK_DIR)

from utils import dicom_to_png #type: ignore
from predictor import predict_and_heatmap


def sanitize_filename(original_name):
    # Garder l'extension
    ext = os.path.splitext(original_name)[-1]
    # Générer un nom unique
    return f"{uuid.uuid4().hex}{ext}"

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CORS(app)

# Charger les modèles
weights = np.load(os.path.join(BASE_DIR, 'models', 'logistic_model_weights.npz'))
w_radio = weights['w']
b_radio = float(weights['b'])
pneumonia_model = load_model(os.path.join(BASE_DIR, 'models', 'my_model_vggv2.h5'))

MODEL_RADIO_SIZE = (64, 64)
MODEL_PNEUMONIA_SIZE = (128, 128)

def cleanup_upload_folder():     #si on veux ajouter une fonction d'historisation on va devoir effacer cette fonction
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            app.logger.error(f"Erreur lors de la suppression de {file_path}: {str(e)}")


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def predict_radio(X: np.ndarray) -> float:
    x_flat = X.reshape(-1, 1)
    z = np.dot(w_radio.T, x_flat) + b_radio
    return float(sigmoid(z))

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    # Décoder le nom de fichier pour gérer les espaces et caractères spéciaux
    decoded_filename = unquote(filename)
    return send_from_directory(UPLOAD_FOLDER, decoded_filename)

@app.route('/api/analyze-scan', methods=['POST'])
def predict():
    try:
        cleanup_upload_folder()
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Sauvegarder le fichier
        ext = os.path.splitext(file.filename)[1].lower()
        filename = sanitize_filename(file.filename)  # nom sécurisé, sans espace ni accent
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)
        # Conversion DICOM -> PNG si nécessaire
        if ext == '.dcm':
            png_path = dicom_to_png(input_path, output_dir=UPLOAD_FOLDER)
            os.remove(input_path)
            input_path = png_path
            filename = os.path.basename(png_path)
            print(f"Fichier converti: {input_path}")

        # ================================
        # Étape 1 : Vérification RADIO
        # ================================
        img_radio = image.load_img(input_path, target_size=MODEL_RADIO_SIZE, color_mode='rgb')
        img_radio_array = image.img_to_array(img_radio) / 255.0
        img_radio_array = np.expand_dims(img_radio_array, axis=0)

        # Vérifier les dimensions
        expected_size = MODEL_RADIO_SIZE[0] * MODEL_RADIO_SIZE[1] * 3
        flattened_size = img_radio_array.reshape((1, -1)).shape[1]
        if flattened_size != expected_size:
            return jsonify({
                'error': f"Dimension mismatch: Image has {flattened_size} elements, model expects {expected_size}"
            }), 400

        radio_input = img_radio_array.reshape((1, -1))
        radio_prob = predict_radio(radio_input)

        if radio_prob > 0.5:
            app.logger.info(f"Image rejetée par le modèle radio. Proba: {radio_prob:.4f}")
            # Nettoyage mémoire
            del img_radio, img_radio_array, radio_input
            gc.collect()
            return jsonify({
                'result': 'Image non valide (pas une radio)',
                'confidence': round(radio_prob, 4)
            }), 200

        # ================================
        # Étape 2 : Prédiction Pneumonie
        # ================================
        img_pneumonia = image.load_img(input_path, target_size=MODEL_PNEUMONIA_SIZE, color_mode='rgb')
        img_pneumonia_array = image.img_to_array(img_pneumonia).astype(np.uint8)
        img_pil = Image.fromarray(img_pneumonia_array)
        
        pneumonia_filename = f"resized_{os.path.splitext(filename)[0]}.png"
        pneumonia_path = os.path.join(UPLOAD_FOLDER, pneumonia_filename)
        img_pil.save(pneumonia_path)

        # Prédiction + heatmap
        label, confidence, heatmap_path = predict_and_heatmap(
            pneumonia_path,
            UPLOAD_FOLDER,
            pneumonia_model
        )
        # Nettoyage mémoire
        del img_radio, img_radio_array, radio_input
        del img_pneumonia, img_pneumonia_array, img_pil
        gc.collect()


        return jsonify({
            'result': label,
            'confidence': round(confidence, 4),
            'image_url': f'/uploads/{filename}',
            'resized_image_url': f'/uploads/{pneumonia_filename}',
            'heatmap_url': f'/uploads/{os.path.basename(heatmap_path)}'
        })


    except Exception as e:
        import traceback
        app.logger.error(f"Erreur: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e),
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    expected_size = MODEL_RADIO_SIZE[0] * MODEL_RADIO_SIZE[1] * 3
    print(f"Taille attendue par le modèle radio: {expected_size} éléments")
    print(f"Dimensions des poids du modèle radio: {w_radio.shape[0]} éléments")
    
    if w_radio.shape[0] != expected_size:
        print(f"ATTENTION: Le modèle radio attend {w_radio.shape[0]} éléments, mais {MODEL_RADIO_SIZE}*3 = {expected_size}")
    
    app.run(debug=True)
