import os
import json
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
import streamlit as st

# --- Import your actual ML model helper utilities ---
from model_utils import load_model, predict

# --- Configuration & Setup ---
APP_NAME = "Brain Tumor Detection"
APP_TAG = "AI-assisted MRI screening"
LOW_CONF = 0.75          

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "brain_tumor_resnet_90.pth")

# --- Streamlit Layout Configuration ---
st.set_page_config(page_title=APP_NAME, layout="centered", page_icon="🧠")

def read_json(name, default):
    p = os.path.join(MODEL_DIR, name)
    try:
        return json.load(open(p)) if os.path.exists(p) else default
    except (OSError, ValueError):
        return default

NORMAL_KEYS = {"no", "notumor", "no_tumor", "normal", "healthy", "none"}

def pretty(k):
    kl = k.lower().replace(" ", "_")
    if kl in NORMAL_KEYS:
        return "Normal"
    if kl == "yes":
        return "Tumor"
    if kl == "pituitary":
        return "Pituitary tumor"
    return k.replace("_", " ").strip().title()

# --- Class Names Resolution ---
RAW = read_json("classes.json", ["no", "yes"])
if isinstance(RAW, dict):
    RAW = RAW.get("classes", ["no", "yes"])
CLASSES = [pretty(c) for c in RAW]

INFO = {
    "Normal": "The model found no tumor pattern in this slice. A normal AI result is not a clearance — only a doctor can rule out a tumor.",
    "Tumor": "The model found patterns similar to the tumor images it was trained on. Please have a doctor review the scan.",
    "Glioma": "Gliomas arise from glial (supporting) cells of the brain. They range from slow-growing to aggressive; diagnosis and treatment are decided by specialists using imaging and biopsy.",
    "Meningioma": "Meningiomas grow from the meninges, the membranes covering the brain. Most are benign and slow-growing; some are only monitored, others treated.",
    "Pituitary tumor": "Pituitary tumors develop in the pituitary gland at the base of the brain, which controls hormones. Most are benign and can affect hormones or vision.",
}

# --- Cache Model Weights Singleton ---
@st.cache_resource
def get_cached_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model weights file not found at {MODEL_PATH}. Make sure it is added to your repository.")
        st.stop()
    try:
        # Pass the file path and number of target classes to your custom loader
        return load_model(MODEL_PATH, len(CLASSES))
    except Exception as e:
        st.error(f"Failed loading the model weights file. Verify your helper config. Error: {e}")
        st.stop()

# Load the model cleanly into memory (Executes exactly once)
model = get_cached_model()

# --- Interactive Streamlit User Interface Layout ---
st.title(f"🧠 {APP_NAME}")
st.caption(f"*{APP_TAG}*")
st.write("---")

# Optional metadata form inputs
patient_name = st.text_input("Patient ID / Name (Optional)", max_chars=60).strip()

# File Uploader component replaces the Flask multipart file parser
uploaded_file = st.file_uploader("Upload Brain MRI Scan (JPG, PNG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Load image via stream
        img = Image.open(uploaded_file)
        img.load()
    except (UnidentifiedImageError, OSError):
        st.error("That file is not a valid image. Please select a clear JPG or PNG image.")
    else:
        if min(img.size) < 64:
            st.error("Image dimensions are too small to be evaluated reliably by the core model.")
        else:
            # Resize constraints inside bounds
            img.thumbnail((1024, 1024))
            
            # Display uploaded original scan image preview
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original MRI Scan")
                st.image(img, use_container_width=True)

            # Process inference and measure latency
            with st.spinner("🤖 AI Agent running brain slice inference..."):
                t0 = time.time()
                try:
                    idx, probs, cam = predict(model, img)
                    elapsed_seconds = round(time.time() - t0, 2)
                    
                    # Target results properties parsing
                    detected_label = CLASSES[idx]
                    confidence_pct = round(float(probs[idx]) * 100, 1)
                    is_tumor = detected_label != "Normal"
                    is_uncertain = float(probs[idx]) < LOW_CONF
                    
                    with col2:
                        st.subheader("AI Activation Map (CAM)")
                        st.image(cam, use_container_width=True)
                        
                    st.write("---")
                    st.subheader("📋 Screening Assessment Summary")
                    
                    if patient_name:
                        st.info(f"**Patient record identifier:** {patient_name}")
                        
                    if is_tumor:
                        st.error(f"**Result Status:** Tumor Pattern Detected ({detected_label})")
                    else:
                        st.success("**Result Status:** No Distinct Tumor Patterns Detected (Normal)")
                        
                    st.metric(label="Model Assessment Confidence", value=f"{confidence_pct}%")
                    
                    if is_uncertain:
                        st.warning("⚠️ **Low Confidence Flag:** The classification score is below the high-certainty target margin threshold. Manual specialist review suggested.")
                        
                    st.write(f"ℹ️ **Interpretation Guide:** {INFO.get(detected_label, '')}")
                    
                    # Full distribution metrics grid view
                    st.write("### 📊 Complete Probability Distribution Profile")
                    for name, p in zip(CLASSES, probs):
                        pct = round(float(p) * 100, 1)
                        st.write(f"**{name}**")
                        st.progress(int(pct))
                        st.caption(f"Relative activation value probability matched at {pct}%")
                        
                    st.caption(f"Inference pipeline execution completed in {elapsed_seconds} seconds.")
                    
                except Exception as ex:
                    st.error(f"Inference execution mapping failed: {ex}")
