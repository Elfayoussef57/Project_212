import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore

# --------- SETTINGS ---------
MODEL_PATH = "../models/my_model_vggv2.h5"
IMAGE_PATH = "../data/RSNA/rsna_pneumonia_png_images/0a0f91dc-6015-4342-b809-d19610854a21.png"
INPUT_SIZE = (128, 128)
DISPLAY_SIZE = (224, 224)
CLASS_INDEX = 1
LAST_CONV_LAYER = 'conv2d_12'


def load_trained_model(path):
    return load_model(path)


def preprocess_input_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def generate_gradcam(model, img_array, class_index, last_conv_layer):
    score = BinaryScore([float(class_index)])
    model_modifier = ReplaceToLinear()
    gradcam = GradcamPlusPlus(model, model_modifier=model_modifier, clone=True)
    cam = gradcam(score, img_array, penultimate_layer=last_conv_layer)
    return cam[0]


def overlay_heatmap_on_image(cam_heatmap, original_image_path, display_size):
    heatmap = cv2.resize(cam_heatmap, display_size)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, display_size)
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    return original_image, superimposed_img


def display_images(original, cam_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original X-ray")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM++ (Pneumonia)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    model = load_trained_model(MODEL_PATH)
    img_array = preprocess_input_image(IMAGE_PATH, INPUT_SIZE)
    cam_heatmap = generate_gradcam(model, img_array, CLASS_INDEX, LAST_CONV_LAYER)
    original_image, cam_overlay = overlay_heatmap_on_image(cam_heatmap, IMAGE_PATH, DISPLAY_SIZE)
    display_images(original_image, cam_overlay)


if __name__ == "__main__":
    main()
