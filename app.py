import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
import sqlite3
from datetime import datetime
import gdown


# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="MRI Tumor Detection System",
    page_icon="🧠",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
.main {background-color: #f4f6f9;}
.title {font-size: 36px; font-weight: bold; color: #1f4e79;}
.subtitle {font-size: 18px; color: #555;}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)


# ==============================
# HEADER
# ==============================
st.markdown('<div class="title">🏥 MRI Brain Tumor Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">CNN + Database + Streamlit Cloud Ready</div>', unsafe_allow_html=True)
st.write("---")


# ==============================
# DATABASE (CLOUD SAFE)
# ==============================
DB_PATH = "/tmp/mri_reports.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        tumor_type TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

init_db()


def save_to_db(label, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    INSERT INTO reports (date, tumor_type, confidence)
    VALUES (?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, confidence))

    conn.commit()
    conn.close()


def load_reports():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY id DESC", conn)
    conn.close()
    return df


# ==============================
# SIDEBAR
# ==============================
st.sidebar.title("⚙️ Dashboard")

st.sidebar.info("""
This AI system classifies MRI scans into:
- Glioma
- Meningioma
- Pituitary
- No Tumor
""")

if st.sidebar.button("📁 Show History"):
    df = load_reports()
    st.sidebar.dataframe(df)


# ==============================
# LOAD MODEL (GOOGLE DRIVE)
# ==============================
@st.cache_resource
def load_model():

    models = {
        "final": "1GkWBUGZdxdS0nxKfvwfzyntRW0Pczq3K",
        "best": "1SRTSP1vtZ7DBBDdcDF5ZNR-7dRigFqI6"
    }

    for name, file_id in models.items():

        model_path = f"{name}.h5"
        url = f"https://drive.google.com/uc?id={file_id}"

        if not os.path.exists(model_path):
            with st.spinner(f"Downloading {name} model..."):
                gdown.download(url, model_path, quiet=False)

        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)

    return None


model = load_model()

if model is None:
    st.error("❌ Model not found!")
    st.stop()


# ==============================
# CLASS LABELS
# ==============================
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


# ==============================
# PREPROCESS IMAGE
# ==============================
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


# ==============================
# PREDICTION FUNCTION
# ==============================
def predict(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)

    index = np.argmax(preds)
    label = CLASS_NAMES[index]
    confidence = float(np.max(preds))

    return label, confidence, preds


# ==============================
# GENERATE REPORT
# ==============================
def generate_report(label, confidence):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
MRI TUMOR DETECTION REPORT
-------------------------
Date: {now}

Prediction: {label.upper()}
Confidence: {confidence:.2f}

⚠️ This result is AI-generated and not a medical diagnosis.
"""


# ==============================
# FILE UPLOAD
# ==============================
st.markdown("### 📤 Upload MRI Image")
uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "png", "jpeg"])


# ==============================
# MAIN LOGIC
# ==============================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # LEFT PANEL
    with col1:
        st.markdown("### 🧾 Original MRI")
        st.image(image, use_column_width=True)

    # PREDICT
    label, confidence, preds = predict(image)

    # SAVE TO DATABASE
    save_to_db(label, confidence)

    # RIGHT PANEL
    with col2:
        st.markdown("### 🧠 AI Diagnosis")

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f"### 🧪 Tumor Type: {label.upper()}")
        st.markdown(f"### 📊 Confidence: {confidence:.2f}")

        st.progress(int(confidence * 100))

        if label == "notumor":
            st.success("✅ No Tumor Detected")
        else:
            st.error(f"⚠️ {label.upper()} Tumor Detected")

        st.markdown('</div>', unsafe_allow_html=True)

    # ==============================
    # PROBABILITY CHART
    # ==============================
    st.markdown("### 📊 Class Probabilities")

    prob_dict = {
        CLASS_NAMES[i]: float(preds[0][i])
        for i in range(len(CLASS_NAMES))
    }

    st.bar_chart(prob_dict)

    # ==============================
    # TABLE VIEW
    # ==============================
    st.markdown("### 📋 Prediction Details")

    df = pd.DataFrame({
        "Tumor Type": CLASS_NAMES,
        "Probability": preds[0]
    })

    st.dataframe(df)

    # ==============================
    # DOWNLOAD REPORT
    # ==============================
    report_text = generate_report(label, confidence)

    st.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="mri_report.txt",
        mime="text/plain"
    )


# ==============================
# FOOTER
# ==============================
st.write("---")
st.warning("⚠️ This system is for educational purposes only. Consult a doctor for medical advice.")
st.markdown("🧠 AI Medical Dashboard | CNN + SQLite + Streamlit Cloud")