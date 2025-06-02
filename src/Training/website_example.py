import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
from PIL import Image
import io
import cv2
from grad_cam import grad_cam
import matplotlib.pyplot as plt
import os

# Imports pour le PDF
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph, Table, TableStyle


# --- Fonctions utilitaires --- #

@st.cache_resource
def load_my_model(path):
    return load_model(path)


def preprocess_image(file_bytes):
    try:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Erreur en ouvrant l’image : {e}")
        return None, None
    pil_resized = pil.resize((224, 224))
    arr = np.array(pil_resized).astype("float32") / 255.0
    return arr, pil


def predict_pneumonia(model, img_arr, seuil=0.5):
    img_batch = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_batch)
    prob = float(pred[0][0])
    label = "PNEUMONIA" if prob > seuil else "NORMAL"
    conf = prob if prob > seuil else 1 - prob
    return label, conf, pred


def compute_heatmap(model, img_arr, orig_file_bytes):
    raw_heatmap = grad_cam(model, np.expand_dims(img_arr, axis=0))
    raw_heatmap = np.maximum(raw_heatmap, 0)
    max_h = np.max(raw_heatmap) if np.max(raw_heatmap) != 0 else 1e-10
    heatmap_norm = raw_heatmap / max_h

    np_arr = np.frombuffer(orig_file_bytes, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv_img = cv2.resize(cv_img, (224, 224))

    hm_res = cv2.resize(heatmap_norm, (224, 224))
    hm_uint8 = np.uint8(255 * hm_res)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_INFERNO)
    superimposed = cv2.addWeighted(cv_img, 0.6, hm_color, 0.4, 0)
    return superimposed


# --- Fonction pour générer le PDF du rapport --- #

def generate_report_pdf(nom, prenom, pil_img, superimposed_cv, label, confidence):
    """
    Crée un PDF stylisé en mémoire (BytesIO) contenant :
      1) Titre centré “Rapport de Radiographie”
      2) Nom & Prénom du patient (à gauche) + Date (à droite)
      3) Ligne de séparation
      4) Deux images côte à côte (originale + heatmap) avec légendes
      5) Résultat de la prédiction dans un encadré
      6) Grand encadré vide pour conseils personnalisés
      7) Encadré pour conseils généraux (différents selon NORMAL/PNEUMONIA)
    Retourne : (octets du PDF, nom de fichier proposé).
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle, Image as RLImage
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4  # ~595 × 842 points

    # Marges
    margin = 50
    usable_width = width - 2 * margin
    y = height - margin

    # Styles de texte
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        alignment=1,              # centré
        fontName='Helvetica-Bold',
        fontSize=18,
        spaceAfter=12
    )
    style_header = ParagraphStyle(
        'Header',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        spaceAfter=6
    )
    style_normal = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        spaceAfter=4
    )
    style_small = ParagraphStyle(
        'Small',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        spaceAfter=3
    )

    # 1) Titre centré
    title_para = Paragraph("Rapport de Radiographie", style_title)
    w, h = title_para.wrap(usable_width, 0)
    title_para.drawOn(c, margin, y - h)
    y -= (h + 10)

    # 2) Nom/Prénom à gauche et Date à droite
    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    patient_info = Paragraph(f"<b>Patient :</b> {prenom} {nom}", style_normal)
    w_pi, h_pi = patient_info.wrap(usable_width * 0.6, 0)
    patient_info.drawOn(c, margin, y - h_pi)

    date_info = Paragraph(f"<b>Date :</b> {date_str}", style_normal)
    w_di, h_di = date_info.wrap(usable_width * 0.4, 0)
    date_info.drawOn(c, margin + usable_width - w_di, y - h_di)
    y -= (max(h_pi, h_di) + 10)

    # 3) Ligne de séparation
    c.setStrokeColor(colors.grey)
    c.setLineWidth(0.5)
    c.line(margin, y, margin + usable_width, y)
    y -= 20

    # 4) Insertion des deux images côte à côte
    pil_orig = pil_img.resize((200, 200))
    img_buf_orig = io.BytesIO()
    pil_orig.save(img_buf_orig, format="PNG")
    img_buf_orig.seek(0)

    img_rgb = cv2.cvtColor(superimposed_cv, cv2.COLOR_BGR2RGB)
    pil_hm = PILImage.fromarray(img_rgb).resize((200, 200))
    img_buf_hm = io.BytesIO()
    pil_hm.save(img_buf_hm, format="PNG")
    img_buf_hm.seek(0)

    img_data = [
        [
            RLImage(img_buf_orig, width=200, height=200),
            RLImage(img_buf_hm, width=200, height=200),
        ],
        [
            Paragraph("<b>Image originale</b>", style_small),
            Paragraph("<b>Image avec Heatmap</b>", style_small)
        ]
    ]
    tbl = Table(img_data, colWidths=[usable_width * 0.45, usable_width * 0.45], hAlign='CENTER')
    tbl.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 1), (-1, 1), 0),
    ]))
    w_tbl, h_tbl = tbl.wrap(usable_width, 0)
    tbl.drawOn(c, margin, y - h_tbl)
    y -= (h_tbl + 20)

    # 5) Résultat de la prédiction dans un encadré coloré
    result_text = f"<b>Résultat :</b> {label}  —  <b>Confiance :</b> {confidence*100:.1f}%"
    p_res = Paragraph(result_text, style_header)
    w_res, h_res = p_res.wrap(usable_width, 0)
    box_height = h_res + 12

    c.setFillColor(colors.whitesmoke)
    c.rect(margin, y - box_height + 6, usable_width, box_height, fill=1, stroke=0)
    c.setStrokeColor(colors.darkgrey)
    c.rect(margin, y - box_height + 6, usable_width, box_height, fill=0, stroke=1)

    p_res.drawOn(c, margin + 10, y - h_res + 4)
    y -= (box_height + 20)

    # 6) Encadré vide pour “Conseils personnalisés”
    pers_title = Paragraph("Conseils personnalisés :", style_header)
    w_pt, h_pt = pers_title.wrap(usable_width, 0)
    pers_title.drawOn(c, margin, y - h_pt)
    y -= (h_pt + 15)

    box_height2 = 120
    c.setStrokeColor(colors.grey)
    c.rect(margin, y - box_height2 + 6, usable_width, box_height2, fill=0, stroke=1)
    y -= (box_height2 + 20)

    # 7) Encadré “Conseils généraux”
    gen_title = Paragraph("Conseils généraux :", style_header)
    w_gt, h_gt = gen_title.wrap(usable_width, 0)
    gen_title.drawOn(c, margin, y - h_gt)
    y -= (h_gt + 15)

    if label == "PNEUMONIA":
        conseils = (
            "• Veiller à un repos suffisant et à une bonne hydratation.<br/>"
            "• Éviter le tabac et les environnements enfumés.<br/>"
            "• Suivre scrupuleusement la prescription d'antibiotiques si recommandée.<br/>"
            "• Surveiller la température et la saturation en oxygène.<br/>"
        )
    else:  # NORMAL
        conseils = (
            "• Continuez à pratiquer régulièrement une activité physique modérée.<br/>"
            "• Adoptez une hygiène de vie saine (alimentation équilibrée, sommeil régulier).<br/>"
            "• Éviter les contacts prolongés avec des personnes malades.<br/>"
            "• Se faire vacciner contre la grippe et la pneumonie si éligible.<br/>"
            "• Maintenir un environnement aéré et propre.<br/>"
        )

    text_obj = Paragraph(conseils, style_normal)
    w_cg, h_cg = text_obj.wrap(usable_width, 0)
    box_height3 = h_cg + 12

    c.setFillColor(colors.HexColor("#F9F9F9"))
    c.rect(margin, y - box_height3 + 6, usable_width, box_height3, fill=1, stroke=0)
    c.setStrokeColor(colors.darkgrey)
    c.rect(margin, y - box_height3 + 6, usable_width, box_height3, fill=0, stroke=1)

    text_obj.drawOn(c, margin + 8, y - h_cg + 4)
    y -= (box_height3 + 20)

    c.showPage()
    c.save()
    buffer.seek(0)

    filename = f"rapport_{prenom}_{nom}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return buffer.getvalue(), filename



# --- Début de l’application Streamlit --- #

st.title("Détection de Pneumonie à partir d'une Radiographie")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "..", "models", "model_vgg.h5")

# Chargement mis en cache
model = load_my_model(model_path)


# --- Étape 1 : Upload + Prédiction stockée dans st.session_state --- #

uploaded_file = st.file_uploader(
    "Choisissez une image de radiographie",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    img_arr, pil_img = preprocess_image(file_bytes)
    if img_arr is None:
        st.stop()

    if st.button("Prédire"):
        with st.spinner("Prédiction en cours…"):
            label, conf, pred = predict_pneumonia(model, img_arr, seuil=0.5)
            superimposed = compute_heatmap(model, img_arr, file_bytes)

            st.session_state["predicted"] = True
            st.session_state["label"] = label
            st.session_state["conf"] = conf
            st.session_state["pil_img"] = pil_img
            st.session_state["superimposed"] = superimposed
            st.session_state["file_bytes"] = file_bytes

    if st.session_state.get("predicted", False):
        label = st.session_state["label"]
        conf = st.session_state["conf"]
        pil_img = st.session_state["pil_img"]
        superimposed = st.session_state["superimposed"]
        file_bytes = st.session_state["file_bytes"]

        st.image(pil_img, caption="Image chargée", use_column_width=True)
        st.success(f"**Classe prédite : {label} – Confiance : {conf*100:.1f}%**")

        if label == "PNEUMONIA":
            prob_pneu = conf
        else:
            prob_pneu = 1 - conf
        prob_normal = 1 - prob_pneu

        st.write("### Probabilités pour chaque classe :")
        st.write(f"NORMAL : {prob_normal:.2f}")
        st.write(f"PNEUMONIA : {prob_pneu:.2f}")

        fig, ax = plt.subplots()
        ax.bar(["NORMAL", "PNEUMONIA"], [prob_normal, prob_pneu])
        ax.set_ylabel("Probabilité")
        st.pyplot(fig)
        plt.close(fig)

        st.image(
            cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB),
            caption="Image avec Heatmap",
            use_column_width=True
        )

        _, buffer = cv2.imencode('.png', superimposed)
        st.download_button(
            label="Télécharger Heatmap",
            data=buffer.tobytes(),
            file_name="heatmap.png",
            mime="image/png"
        )

        st.markdown("---")
        st.header("Génération du rapport complet")

        nom = st.text_input("Nom du patient :", key="nom_patient")
        prenom = st.text_input("Prénom du patient :", key="prenom_patient")

        if nom.strip() != "" and prenom.strip() != "":
            if st.button("Générer le rapport PDF", key="bouton_pdf"):
                with st.spinner("Création du rapport…"):
                    pdf_bytes, pdf_filename = generate_report_pdf(
                        nom=nom,
                        prenom=prenom,
                        pil_img=pil_img,
                        superimposed_cv=superimposed,
                        label=label,
                        confidence=conf
                    )
                    st.success("Rapport généré !")
                    st.download_button(
                        label="Télécharger le rapport complet (PDF)",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
        else:
            st.info("Veuillez renseigner le nom et le prénom du patient pour générer le rapport.")
