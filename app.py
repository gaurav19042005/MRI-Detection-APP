import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import sqlite3
from datetime import datetime
import requests

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
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model_path = "model.tflite"
    model_url = "https://raw.githubusercontent.com/gaurav19042005/MRI-Detection-APP/main/model.tflite"

    # Delete corrupted old file
    if os.path.exists(model_path):
        try:
            temp_interpreter = tf.lite.Interpreter(model_path=model_path)
            temp_interpreter.allocate_tensors()
        except:
            os.remove(model_path)

    # Download model if not present
    if not os.path.exists(model_path):
        with st.spinner("Downloading AI model..."):
            try:
                response = requests.get(model_url, timeout=60)

                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                else:
                    st.error(f"Failed to download model. Status code: {response.status_code}")
                    st.stop()

            except Exception as e:
                st.error(f"Download error: {e}")
                st.stop()

    # Load TFLite model
    try:
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[]
        )
        interpreter.allocate_tensors()
        return interpreter

    except Exception as e:
        st.error(f"Failed to load TensorFlow Lite model: {e}")
        st.info("Please use TensorFlow 2.15.0 in requirements.txt")
        st.code("""
streamlit==1.33.0
tensorflow==2.15.0
numpy==1.26.4
pandas==2.2.2
Pillow==10.3.0
requests==2.31.0
        """)
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
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==============================
# PREDICT FUNCTION
# ==============================
def predict(image):
    img_array = preprocess_image(image)

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], img_array)
    model.invoke()

    prediction = model.get_tensor(output_details[0]['index'])[0]

    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = float(prediction[predicted_index])

    return predicted_label, confidence, prediction

# ==============================
# REPORT GENERATOR
# ==============================
def generate_report(label, confidence):
    report = f"""
MRI BRAIN TUMOR REPORT
========================

Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Predicted Tumor Type: {label.upper()}
Confidence Score: {confidence:.2%}

Conclusion:
"""

    if label == "notumor":
        report += "No tumor detected."
    else:
        report += f"Tumor detected: {label.upper()}"

    report += """

Disclaimer:
This result is generated by an AI model and is for educational purposes only.
Please consult a doctor for professional diagnosis.
"""

    return report

# ==============================
# FILE UPLOAD
# ==============================
st.markdown("### 📤 Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose an MRI Scan Image",
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

        label, confidence, prediction = predict(image)

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
        # CHART
        # ==============================
        st.markdown("### 📊 Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Tumor Type": CLASS_NAMES,
            "Probability": prediction
        })

        st.bar_chart(probability_df.set_index("Tumor Type"))

        # ==============================
        # TABLE
        # ==============================
        st.markdown("### 📋 Detailed Prediction Table")
        st.dataframe(probability_df, use_container_width=True)

        # ==============================
        # DOWNLOAD REPORT
        # ==============================
        report = generate_report(label, confidence)

        st.download_button(
            label="📄 Download Report",
            data=report,
            file_name="mri_report.txt",
            mime="text/plain"
        )

    except Exception as e:
        st.error(f"Error processing image: {e}")

# ==============================
# FOOTER
# ==============================
st.write("---")
st.warning("⚠️ This tool is for educational use only.")
st.markdown("Developed using Streamlit + TensorFlow Lite + SQLite")
