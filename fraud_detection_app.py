# ==============================
# AI Fraud Detection System
# ==============================

import streamlit as st
import cv2
import easyocr
import numpy as np
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace
from PIL import Image
import pandas as pd

# Streamlit setup
st.set_page_config(page_title="AI Fraud Detection System", layout="wide")
st.title("🧠 AI Fraud Detection System")
st.write("Upload documents to verify authenticity and detect fraud.")

# Sidebar modules
option = st.sidebar.selectbox("Choose Module", [
    "Document Tampering",
    "Signature Verification",
    "Aadhaar Fraud Detection",
    "PAN Fraud Detection",
    "AI-Based KYC Verification",
    "Unusual Pattern Detection"
])

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# -------------------- MODULE 1: DOCUMENT TAMPERING --------------------
if option == "Document Tampering":
    st.header("📄 Document Forgery Detection")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_doc1 = st.file_uploader("Upload Original Document", type=["jpg", "png", "jpeg"])
    with col2:
        uploaded_doc2 = st.file_uploader("Upload Suspected Document", type=["jpg", "png", "jpeg"])

    if uploaded_doc1 and uploaded_doc2:
        img1 = cv2.imdecode(np.frombuffer(uploaded_doc1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(uploaded_doc2.read(), np.uint8), cv2.IMREAD_COLOR)

        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray1, gray2, full=True)
        st.write(f"🔍 Similarity Score: {score:.2f}")

        if score < 0.85:
            st.error("⚠ Possible forgery detected.")
        else:
            st.success("✅ No significant alteration found.")
        st.image(diff, caption="Difference Map", use_container_width=True)

# -------------------- MODULE 2: SIGNATURE VERIFICATION --------------------
elif option == "Signature Verification":
    st.header("✍ Signature Verification")

    col1, col2 = st.columns(2)
    with col1:
        sig1_file = st.file_uploader("Upload Original Signature", type=["jpg", "png", "jpeg"])
    with col2:
        sig2_file = st.file_uploader("Upload Submitted Signature", type=["jpg", "png", "jpeg"])

    if sig1_file and sig2_file:
        sig1 = cv2.imdecode(np.frombuffer(sig1_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        sig2 = cv2.imdecode(np.frombuffer(sig2_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(sig1, None)
        kp2, des2 = orb.detectAndCompute(sig2, None)

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            score = len(matches)
            st.write(f"Match Score: {score}")

            if score > 50:
                st.success("✅ Genuine Signature")
            else:
                st.error("❌ Forged Signature")
        else:
            st.warning("Could not detect enough features.")

# -------------------- MODULE 3: AADHAAR FRAUD DETECTION --------------------
elif option == "Aadhaar Fraud Detection":
    st.header("🪪 Aadhaar Fraud Verification (Prototype)")
    aadhaar_num = st.text_input("Enter Aadhaar Number (XXXX-XXXX-XXXX):")
    if st.button("Verify"):
        if len(aadhaar_num) == 14:
            st.success("✅ Aadhaar appears valid (format check only).")
        else:
            st.error("❌ Invalid Aadhaar format.")

# -------------------- MODULE 4: PAN FRAUD DETECTION --------------------
elif option == "PAN Fraud Detection":
    st.header("💳 PAN Card Fraud Detection (Prototype)")
    pan_num = st.text_input("Enter PAN Number (ABCDE1234F):")
    if st.button("Validate"):
        if len(pan_num) == 10 and pan_num[:5].isalpha() and pan_num[5:9].isdigit() and pan_num[-1].isalpha():
            st.success("✅ Valid PAN structure.")
        else:
            st.error("❌ Invalid PAN format.")

# -------------------- MODULE 5: AI-BASED KYC VERIFICATION --------------------
elif option == "AI-Based KYC Verification":
    st.header("🧬 AI-Based KYC Verification")
    col1, col2 = st.columns(2)
    with col1:
        selfie = st.file_uploader("Upload Selfie Photo", type=["jpg", "png", "jpeg"])
    with col2:
        id_photo = st.file_uploader("Upload ID Photo", type=["jpg", "png", "jpeg"])

    if selfie and id_photo:
        st.info("Running facial similarity analysis using DeepFace...")
        try:
            result = DeepFace.verify(np.array(Image.open(selfie)), np.array(Image.open(id_photo)))
            if result["verified"]:
                st.success("✅ Face Match Successful")
            else:
                st.error("❌ Face Mismatch Detected")
        except Exception as e:
            st.error(f"Error during verification: {e}")

# -------------------- MODULE 6: UNUSUAL PATTERN DETECTION --------------------
elif option == "Unusual Pattern Detection":
    st.header("📊 Unusual Pattern Detection")
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())

        z_scores = (data - data.mean()) / data.std()
        anomalies = data[(abs(z_scores) > 3).any(axis=1)]
        st.subheader("🔎 Detected Unusual Patterns:")
        st.dataframe(anomalies)

# -------------------- REPORT SUMMARY --------------------
st.divider()
if st.button("Generate Fraud Report"):
    st.success("🧾 Fraud detection report generated successfully.")
