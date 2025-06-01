import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import cv2
import io
from grad_cam import grad_cam
import matplotlib.pyplot as plt

# Chargement du modèle
model = load_model("../../models/model_vgg.h5")

# Titre de l'application
st.title("Détection de Pneumonie à partir d'une Radiographie")

# Chargement de l'image
uploaded_file = st.file_uploader("Choisissez une image de radiographie", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()  # LIRE UNE SEULE FOIS
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    pil_img_resized = pil_img.resize((224, 224))
    st.image(pil_img, caption="Image chargée", use_column_width=True)

    if st.button("Prédire"):
        with st.spinner("Prédiction en cours..."):
            # Préparation de l'image pour le modèle
            img_array = np.array(pil_img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            prob = float(prediction[0][0])
            label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
            confidence = prob if prob > 0.5 else 1 - prob

            st.success(f"**Classe prédite : {label} ({confidence:.2f})**")

            st.write("### Probabilités pour chaque classe :")
            probabilities = {
                "NORMAL": 1 - prob,
                "PNEUMONIA": prob
            }
            for lbl, p in probabilities.items():
                st.write(f"{lbl}: {p:.2f}")

            # Graphique des probabilités
            fig, ax = plt.subplots()
            ax.bar(probabilities.keys(), probabilities.values(), color=["green", "red"])
            ax.set_ylabel("Probabilité")
            st.pyplot(fig)

            # Heatmap via Grad-CAM
            heatmap = grad_cam(model, img_array)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)

            # Conversion de l’image pour OpenCV
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img_cv = cv2.resize(img_cv, (224, 224))

            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
            st.image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB), caption="Image avec Heatmap", use_column_width=True)
