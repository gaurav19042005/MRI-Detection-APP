import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
import sqlite3
from datetime import datetime
import requests

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
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.title {
    font-size: 36px;
    font-weight: bold;
    color: #1f4e79;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
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
# DATABASE
# ==============================
DB_PATH = "mri_reports.db"

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
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        label,
        confidence
    ))
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
- No Tumor
- Pituitary
""")

if st.sidebar.button("📁 Show History"):
    history_df = load_reports()
    if not history_df.empty:
        st.sidebar.dataframe(history_df)
    else:
        st.sidebar.warning("No history found.")

# ==============================
# LOAD TFLITE MODEL
# ==============================
@st.cache_resource
def load_model():
    model_path = "model.tflite"

    # Raw GitHub URL
    url = "https://raw.githubusercontent.com/gaurav19042005/MRI-Detection-APP/main/model.tflite"

    # Delete old invalid model file if exists
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)

        # If file is too small, probably invalid HTML file
        if file_size < 100000:
            os.remove(model_path)

    # Download model if not exists
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI Model from GitHub..."):
            try:
                r = requests.get(url, timeout=30)

                if r.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(r.content)
                else:
                    st.error(f"Failed to download model. Status code: {r.status_code}")
                    st.stop()

            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.stop()

    # Load TensorFlow Lite model
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    except Exception as e:
        st.error(f"Failed to load TensorFlow Lite model: {e}")
        st.stop()

model = load_model()

# ==============================
# CLASS LABELS
# ==============================
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ==============================
# PREPROCESS IMAGE
# ==============================
def preprocess_image(image):
    img = image.resize((224, 224))
    img = img.convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict(image):
    img_array = preprocess_image(image)

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], img_array)
    model.invoke()

    preds = model.get_tensor(output_details[0]['index'])

    index = np.argmax(preds[0])
    label = CLASS_NAMES[index]
    confidence = float(preds[0][index])

    return label, confidence, preds

# ==============================
# REPORT GENERATOR
# ==============================
def generate_report(label, confidence):
    report = f"""
MRI REPORT
---------------------------
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Prediction: {label.upper()}
Confidence: {confidence:.2%}

Note:
This result is generated using an AI model.
This is for educational purposes only and not a medical diagnosis.
"""
    return report

# ==============================
# FILE UPLOAD
# ==============================
st.markdown("### 📤 Upload MRI Image")
uploaded_file = st.file_uploader(
    "Upload MRI Scan Image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# MAIN CONTENT
# ==============================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

        label, confidence, preds = predict(image)

        save_to_db(label, confidence)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.markdown(f"## 🧠 Prediction: {label.upper()}")
            st.markdown(f"### Confidence: {confidence:.2%}")

            st.progress(int(confidence * 100))

            if label == "notumor":
                st.success("✅ No Tumor Detected")
            else:
                st.error(f"⚠️ Tumor Detected: {label.upper()}")

            st.markdown('</div>', unsafe_allow_html=True)

        # ==============================
        # PROBABILITY CHART
        # ==============================
        st.markdown("### 📊 Prediction Probabilities")

        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": preds[0]
        })

        st.bar_chart(prob_df.set_index("Class"))

        # ==============================
        # DETAILS TABLE
        # ==============================
        st.markdown("### 📋 Detailed Results")
        st.dataframe(prob_df, use_container_width=True)

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

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ==============================
# FOOTER
# ==============================
st.write("---")
st.warning("⚠️ Educational use only. Not a substitute for professional medical diagnosis.")
st.markdown("🧠 Developed with Streamlit + TensorFlow Lite")
