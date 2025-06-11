import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import pydicom
from PIL import Image

from model_cam import generate_gradcam, overlay_heatmap_on_image

IMG_SIZE = 224
MODEL_PATH = "../models/my_model_vggv2.h5"
def dicom_to_png(dicom_path, output_dir="../data/RSNA/rsna_pneumonia_png_images", rescale=True):
    """
    Convert a DICOM file to PNG and save it in the specified output directory.

    Args:
        dicom_path (str): Path to the input .dcm file.
        output_dir (str): Directory to save the converted PNG image.
        rescale (bool): Whether to normalize pixel values to [0, 255].

    Returns:
        str: Path to the saved PNG file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load and normalize DICOM
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    if rescale:
        pixel_array -= np.min(pixel_array)
        pixel_array /= np.max(pixel_array)
        pixel_array *= 255.0

    image = Image.fromarray(pixel_array.astype(np.uint8))

    # Generate filename (preserve original filename but use .png)
    filename = os.path.splitext(os.path.basename(dicom_path))[0] + ".png"
    output_path = os.path.join(output_dir, filename)

    # Save image
    image.save(output_path)
    print(f"+Saved PNG to: {output_path}")
    return output_path


# 🔍 Predict and generate Grad-CAM overlay
def predict_with_heatmap(image_path):
    # Load and preprocess image
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Load model
    model = load_model(MODEL_PATH)

    # Predict
    pred_prob = model.predict(img_array)[0][0]
    pred_class = "Pneumonia" if pred_prob > 0.5 else "Not-Pneumonia"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    print(f"+Prediction: {pred_class} ({confidence*100:.2f}%)")

    # Grad-CAM
    cam_heatmap = generate_gradcam(model, img_array, class_index=1, last_conv_layer="conv2d_10")
    original_image, cam_overlay = overlay_heatmap_on_image(cam_heatmap, image_path, (224,224))


    # Show results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cam_overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"{pred_class} ({confidence*100:.2f}%)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 🏁 Entry point
if __name__ == "__main__":
    image_path = "../data/RSNA/stage_2_test_images/0041fc67-793c-4129-a952-ea3fb821b445.dcm"
    if image_path.endswith('.dcm'):
        image_path = dicom_to_png(image_path)
    predict_with_heatmap(image_path)
