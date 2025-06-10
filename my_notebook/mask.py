import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore
from PIL import Image
import pydicom

# ---------------- Configuration ----------------
MODEL_PATH = '../models/my_model_vggv2.h5'
INPUT_SIZE = (128, 128)
DISPLAY_SIZE = (224, 224)
LUNG_MASK_PATH = '../data/JSRT/lung_mask_combined.png'  # ⬅️ combined left+right lung mask
LAST_CONV_LAYER = 'conv2d_12'  # Replace with your last conv layer name

# ---------------- Load model ----------------
model = load_model(MODEL_PATH)

# ---------------- Grad-CAM++ Setup ----------------
model_modifier = ReplaceToLinear()
gradcam = GradcamPlusPlus(model, model_modifier=model_modifier, clone=True)
score = BinaryScore([1.0])

# ---------------- Utilities ----------------
def preprocess_image(img_path):
    if img_path.endswith(".dcm"):
        dicom = pydicom.dcmread(img_path)
        image_array = dicom.pixel_array.astype(np.float32)
        image_array -= np.min(image_array)
        image_array /= np.max(image_array)
        image_array *= 255
        image = Image.fromarray(image_array.astype(np.uint8)).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")
    
    image_resized = image.resize(INPUT_SIZE)
    img_array = img_to_array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(image.resize(DISPLAY_SIZE))

def load_lung_mask():
    mask_img = Image.open(LUNG_MASK_PATH).convert('L').resize(DISPLAY_SIZE)
    mask = np.array(mask_img)
    binary_mask = (mask > 10).astype(np.float32)
    return binary_mask

def apply_gradcam_with_mask(model, input_image, original_image, mask):
    cam = gradcam(score, input_image, penultimate_layer=LAST_CONV_LAYER)
    heatmap = cam[0]
    heatmap = cv2.resize(heatmap, DISPLAY_SIZE)
    heatmap = heatmap * mask  # ⬅️ Apply lung mask
    heatmap = np.uint8(255 * heatmap / np.max(heatmap + 1e-8))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_image = cv2.resize(original_image, DISPLAY_SIZE)
    if original_image.shape[-1] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    return superimposed_img

# ---------------- Main prediction function ----------------
def predict_and_visualize(image_path):
    print(f"🖼️ Processing image: {image_path}")
    input_image, original_image = preprocess_image(image_path)
    pred = model.predict(input_image)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Not-Pneumonia"
    confidence = pred if pred > 0.5 else 1 - pred
    print(f"✅ Prediction: {label} ({confidence*100:.2f}%)")

    lung_mask = load_lung_mask()
    result = apply_gradcam_with_mask(model, input_image, original_image, lung_mask)

    # Display result
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    # Change this to your test image
    test_image_path = input("Enter image path (.png or .dcm): ").strip()
    if os.path.exists(test_image_path):
        predict_and_visualize(test_image_path)
    else:
        print("❌ Invalid image path.")
