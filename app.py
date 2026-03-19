import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import pandas as pd
import sqlite3
from datetime import datetime
import gdown

# ==============================
# FIX FOR TENSORFLOW WARNINGS
# ==============================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="MRI Tumor Detection System",
    page_icon="🧠",
    layout="wide"
)

# ==============================
# CUSTOM CSS (Hospital UI)
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
st.markdown('<div class="subtitle">AI Powered Diagnosis System</div>', unsafe_allow_html=True)
st.write("---")

# ==============================
# DATABASE (Cloud Safe)
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
Tumor Classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor
""")

if st.sidebar.button("📁 Show History"):
    st.sidebar.dataframe(load_reports())

# ==============================
# LOAD MODEL (FIXED)
# ==============================
@st.cache_resource
def load_model():

    file_id = "16hk8pwHU82cSnWCNzVh6bOEGEUGLO2le"
    model_path = "model.keras"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Remove old corrupted file
    if os.path.exists(model_path):
        os.remove(model_path)

    with st.spinner("Downloading AI Model..."):
        gdown.download(url, model_path, quiet=False, fuzzy=True)

    # Check if download is valid
    if os.path.getsize(model_path) < 10000000:
        st.error("❌ Model download failed (corrupted file)")
        return None

    try:
        return keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Model Load Failed: {e}")
        return None
        
model = load_model()

if model is None:
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
# PREDICTION
# ==============================
def predict(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)

    index = np.argmax(preds)
    label = CLASS_NAMES[index]
    confidence = float(np.max(preds))

    return label, confidence, preds

# ==============================
# REPORT GENERATION
# ==============================
def generate_report(label, confidence):
    return f"""
MRI REPORT
----------
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Prediction: {label.upper()}
Confidence: {confidence:.2f}

⚠️ Not a medical diagnosis
"""

# ==============================
# FILE UPLOAD
# ==============================
st.markdown("### 📤 Upload MRI Image")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ==============================
# MAIN
# ==============================
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="MRI Scan", use_column_width=True)

    label, confidence, preds = predict(image)
    save_to_db(label, confidence)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(f"### 🧠 {label.upper()}")
        st.markdown(f"Confidence: {confidence:.2f}")

        st.progress(int(confidence * 100))

        if label == "notumor":
            st.success("No Tumor Detected")
        else:
            st.error("Tumor Detected")

        st.markdown('</div>', unsafe_allow_html=True)

    # Chart
    st.markdown("### 📊 Probabilities")
    prob_dict = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(4)}
    st.bar_chart(prob_dict)

    # Table
    st.markdown("### 📋 Details")
    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Probability": preds[0]
    })
    st.dataframe(df)

    # Download
    st.download_button(
        "📄 Download Report",
        generate_report(label, confidence),
        "report.txt"
    )

# ==============================
# FOOTER
# ==============================
st.write("---")
st.warning("⚠️ Educational use only")
st.markdown("🧠 AI Medical Dashboard")
