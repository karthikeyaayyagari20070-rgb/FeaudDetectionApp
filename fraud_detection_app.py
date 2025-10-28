# ==========================================
# AI Banking Fraud Detection & Verification
# ==========================================

import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pandas as pd
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ------------------- SAFE IMPORT (DeepFace) -------------------
try:
    from deepface import DeepFace
    deepface_available = True
except Exception:
    deepface_available = False

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="AI Banking Verification Portal", layout="wide")

# ------------------- STYLING (Blue-Silver Gradient) -------------------
st.markdown("""
<style>
/* ===== Background ===== */
.stApp {
    background: linear-gradient(135deg, #eaf1fb, #f5f8fa);
    color: #0b2545;
    font-family: 'Segoe UI', sans-serif;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001F54, #004AAD);
    color: white;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-weight: 500;
}
[data-testid="stSidebar"] [aria-checked="true"] {
    background-color: #4682B4 !important;
    border-radius: 6px;
    font-weight: bold;
}

/* ===== Headers ===== */
h1, h2, h3, h4 {
    color: #002855 !important;
    font-weight: 700;
}
p, li {
    color: #0b2545 !important;
}

/* ===== Buttons ===== */
.stButton>button {
    background-color: #004aad;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 8px 18px;
}
.stButton>button:hover {
    background-color: #002855;
}

/* ===== Inputs ===== */
.stTextInput>div>div>input, .stFileUploader label {
    color: #0b2545 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ------------------- APP HEADER -------------------
st.title("🏦 AI Banking Fraud Detection & Verification System")
st.caption("A Secure AI-Powered System for Document, KYC, and Transaction Fraud Prevention")

# ------------------- SIDEBAR MENU -------------------
st.sidebar.title("🔍 Fraud Detection Modules")
option = st.sidebar.radio("Select a Module", [
    "Dashboard Home",
    "Document Tampering",
    "Signature Verification",
    "Aadhaar Fraud Detection",
    "PAN Fraud Detection",
    "AI-Based KYC Verification",
    "Unusual Pattern Detection"
])

# ------------------- FUNCTIONS -------------------
def compare_images(img1_bytes, img2_bytes):
    try:
        img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        score, diff = ssim(img1, img2, full=True)
        return round(score, 3), diff
    except Exception as e:
        st.error(f"Image comparison failed: {e}")
        return None, None

def generate_pdf_report(title, result_text, score=None):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "AI Banking Fraud Detection Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Module: {title}")
    c.drawString(50, 750, f"Result: {result_text}")
    if score is not None:
        c.drawString(50, 730, f"Similarity Score: {score}")
    c.setFont("Helvetica", 11)
    c.drawString(50, 700, "Key Fraud Detection Principles:")
    c.drawString(70, 680, "1. Image and signature verification using AI-based SSIM and ORB.")
    c.drawString(70, 660, "2. Face matching through Deep Learning (DeepFace).")
    c.drawString(70, 640, "3. Format and metadata validation for KYC documents.")
    c.drawString(70, 620, "4. Anomaly detection in transaction datasets.")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ------------------- MODULES -------------------

# Dashboard Home
if option == "Dashboard Home":
    st.markdown("""
    <h3>🏛 Welcome to Secure Bank’s AI Verification Dashboard</h3>
    <p>Use this intelligent system to verify customer documents, detect potential frauds, and ensure compliance.</p>
    <ul>
    <li>📄 Document Forgery Detection</li>
    <li>✍ Signature Verification</li>
    <li>🪪 Aadhaar & PAN Validation</li>
    <li>🧬 AI-Based KYC Verification</li>
    <li>📊 Transaction Pattern Monitoring</li>
    </ul>
    """, unsafe_allow_html=True)

# Document Tampering
elif option == "Document Tampering":
    st.header("📄 Document Forgery Detection")
    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.file_uploader("Upload Original Document", type=["jpg", "png", "jpeg"])
    with col2:
        doc2 = st.file_uploader("Upload Suspected Document", type=["jpg", "png", "jpeg"])

    if doc1 and doc2:
        score, _ = compare_images(doc1.read(), doc2.read())
        if score is not None:
            result_text = "✅ No significant alterations detected." if score > 0.85 else "⚠ Possible forgery detected."
            (st.success if score > 0.85 else st.error)(result_text)
            pdf = generate_pdf_report("Document Tampering", result_text, score)
            st.download_button("📘 Download PDF Report", pdf, file_name="Document_Report.pdf")

# Signature Verification
elif option == "Signature Verification":
    st.header("✍ Signature Verification")
    col1, col2 = st.columns(2)
    with col1:
        sig1 = st.file_uploader("Upload Original Signature", type=["jpg", "png", "jpeg"])
    with col2:
        sig2 = st.file_uploader("Upload Submitted Signature", type=["jpg", "png", "jpeg"])

    if sig1 and sig2:
        sig1_img = cv2.imdecode(np.frombuffer(sig1.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        sig2_img = cv2.imdecode(np.frombuffer(sig2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(sig1_img, None)
        kp2, des2 = orb.detectAndCompute(sig2_img, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            score = len(matches)
            result_text = "✅ Genuine Signature" if score > 50 else "❌ Forged Signature Detected"
            (st.success if score > 50 else st.error)(result_text)
            pdf = generate_pdf_report("Signature Verification", result_text)
            st.download_button("📘 Download PDF Report", pdf, file_name="Signature_Report.pdf")

# Aadhaar Verification
elif option == "Aadhaar Fraud Detection":
    st.header("🪪 Aadhaar Verification")
    aadhaar_num = st.text_input("Enter Aadhaar Number (XXXX-XXXX-XXXX):")
    if st.button("Verify Aadhaar"):
        valid = len(aadhaar_num) == 14 and aadhaar_num.count("-") == 2
        result_text = "✅ Aadhaar number format valid." if valid else "❌ Invalid Aadhaar format."
        (st.success if valid else st.error)(result_text)
        pdf = generate_pdf_report("Aadhaar Fraud Detection", result_text)
        st.download_button("📘 Download PDF Report", pdf, file_name="Aadhaar_Report.pdf")

# PAN Verification
elif option == "PAN Fraud Detection":
    st.header("💳 PAN Card Verification")
    pan = st.text_input("Enter PAN Number (ABCDE1234F):")
    if st.button("Validate PAN"):
        valid = len(pan) == 10 and pan[:5].isalpha() and pan[5:9].isdigit() and pan[-1].isalpha()
        result_text = "✅ Valid PAN Structure." if valid else "❌ Invalid PAN Format."
        (st.success if valid else st.error)(result_text)
        pdf = generate_pdf_report("PAN Fraud Detection", result_text)
        st.download_button("📘 Download PDF Report", pdf, file_name="PAN_Report.pdf")

# AI-Based KYC Verification
elif option == "AI-Based KYC Verification":
    st.header("🧬 AI-Powered KYC Face Verification")
    if not deepface_available:
        st.warning("⚠ DeepFace or TensorFlow not available. Install with:\n"
                   "pip install tensorflow>=2.16.0 keras>=2.16.0 deepface")
    else:
        col1, col2 = st.columns(2)
        with col1:
            selfie = st.file_uploader("Upload Selfie Photo", type=["jpg", "png", "jpeg"])
        with col2:
            id_photo = st.file_uploader("Upload ID Photo", type=["jpg", "png", "jpeg"])
        if selfie and id_photo:
            st.info("Running AI-based face match verification...")
            try:
                result = DeepFace.verify(np.array(Image.open(selfie)), np.array(Image.open(id_photo)))
                verified = result.get("verified", False)
                result_text = "✅ KYC Face Match Successful" if verified else "❌ KYC Face Mismatch Detected"
                (st.success if verified else st.error)(result_text)
                pdf = generate_pdf_report("AI-Based KYC Verification", result_text)
                st.download_button("📘 Download PDF Report", pdf, file_name="KYC_Report.pdf")
            except Exception as e:
                st.error(f"Error during KYC verification: {e}")

# Unusual Pattern Detection
elif option == "Unusual Pattern Detection":
    st.header("📊 Unusual Transaction Pattern Detection")
    uploaded_file = st.file_uploader("Upload Transaction Data (CSV, Excel)", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.dataframe(data.head())
            z_scores = (data - data.mean()) / data.std()
            anomalies = data[(abs(z_scores) > 3).any(axis=1)]
            st.subheader("🔎 Detected Unusual Patterns:")
            st.dataframe(anomalies)
            result_text = f"Detected {len(anomalies)} unusual patterns."
            pdf = generate_pdf_report("Unusual Pattern Detection", result_text)
            st.download_button("📘 Download PDF Report", pdf, file_name="Pattern_Report.pdf")
        except Exception as e:
            st.error(f"Error reading file: {e}")
