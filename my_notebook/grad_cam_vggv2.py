import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

def grad_cam(model, img_array):
    last_conv_layer_name = find_last_conv_layer(model)
    print(f"✅ Using last convolutional layer: {last_conv_layer_name}")
    
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # pour la première classe (Pneumonia vs Normal)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)

    if not np.any(heatmap) or np.isnan(np.sum(heatmap)):
        print("⚠️ Heatmap is empty or invalid. Returning zeros.")
        heatmap = np.zeros_like(heatmap)
    else:
        heatmap /= np.max(heatmap)

    return heatmap

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = "../../my_notebook/my_model_vggv2.h5"
    print("📦 Loading model from:", model_path)

    model = tf.keras.models.load_model(model_path)  # sans custom_objects

    img_path = "../../data/Chest X_ray/test/PNEUMONIA/person101_bacteria_484.jpeg"

    # Charger image en RGB (3 canaux)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))  # RGB par défaut
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # (128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0) / 255.0       # (1, 128, 128, 3)

    heatmap = grad_cam(model, img_array)

    # Lire image avec OpenCV en BGR
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))

    # Vérifications
    if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
        raise ValueError("❌ Invalid heatmap format.")
    if np.any(np.isnan(heatmap)) or np.any(np.isinf(heatmap)):
        raise ValueError("❌ Heatmap contains NaN or Inf.")

    # Resize & application colormap
    heatmap = cv2.resize(heatmap.astype(np.float32), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superposition
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Affichage
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
