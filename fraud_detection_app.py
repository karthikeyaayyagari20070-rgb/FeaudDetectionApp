# app.py
# ==============================
# AI Fraud Detection System (fixed)
# ==============================

import streamlit as st
import cv2
import easyocr
import numpy as np
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace
from PIL import Image
import pandas as pd
import io

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

# Initialize OCR reader (easyocr may print a lot on load)
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
        # read images as numpy arrays
        arr1 = np.frombuffer(uploaded_doc1.read(), np.uint8)
        arr2 = np.frombuffer(uploaded_doc2.read(), np.uint8)
        img1 = cv2.imdecode(arr1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(arr2, cv2.IMREAD_COLOR)

        # make sure sizes match (resize suspected to original)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray1, gray2, full=True)
        st.write(f"🔍 Similarity Score: {score:.4f}")

        # normalize diff to 0-255 for display (ssim "diff" is float in [-1,1])
        diff_image = ( (1 - diff) * 255 ).astype("uint8")  # higher = more different
        # create heatmap-like visualization using applyColorMap
        heatmap = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
        st.image(heatmap, caption="Difference Heatmap", use_container_width=True)

        # threshold decision
        if score < 0.85:
            st.error("⚠ Possible forgery detected.")
        else:
            st.success("✅ No significant alteration found.")

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

        # Optional: resize both to same scale (helps feature detection)
        h = 200
        sig1 = cv2.resize(sig1, (int(sig1.shape[1] * h / sig1.shape[0]), h))
        sig2 = cv2.resize(sig2, (int(sig2.shape[1] * h / sig2.shape[0]), h))

        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(sig1, None)
        kp2, des2 = orb.detectAndCompute(sig2, None)

        if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # consider only "good" matches by distance threshold
            good_matches = [m for m in matches if m.distance < 60]  # 60 is heuristic
            score = len(good_matches)
            st.write(f"Good Match Count: {score} (total keypoints: {len(kp1)}, {len(kp2)})")

            if score > 30:
                st.success("✅ Genuine Signature (high similarity)")
            else:
                st.error("❌ Signature may be forged or not enough similarity")
        else:
            st.warning("Could not detect enough features in one or both signatures.")

# -------------------- MODULE 3: AADHAAR FRAUD DETECTION --------------------
elif option == "Aadhaar Fraud Detection":
    st.header("🪪 Aadhaar Fraud Verification (Prototype)")
    aadhaar_num = st.text_input("Enter Aadhaar Number (format XXXX-XXXX-XXXX or 12 digits):")

    if st.button("Verify Aadhaar"):
        # Accept either 12 digits or 14 chars with 2 dashes: XXXX-XXXX-XXXX
        cleaned = aadhaar_num.replace("-", "").strip()
        if len(cleaned) == 12 and cleaned.isdigit():
            st.success("✅ Aadhaar appears valid (format check only).")
        else:
            st.error("❌ Invalid Aadhaar format. Aadhaar should be 12 digits (optionally shown as XXXX-XXXX-XXXX).")

# -------------------- MODULE 4: PAN FRAUD DETECTION --------------------
elif option == "PAN Fraud Detection":
    st.header("💳 PAN Card Fraud Detection (Prototype)")
    pan_num = st.text_input("Enter PAN Number (ABCDE1234F):")

    if st.button("Validate PAN"):
        pan = pan_num.strip().upper()
        if len(pan) == 10 and pan[:5].isalpha() and pan[5:9].isdigit() and pan[-1].isalpha():
            st.success("✅ PAN structure looks valid (format check only).")
        else:
            st.error("❌ Invalid PAN format. Correct format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F).")

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
            # DeepFace accepts numpy arrays (RGB). Convert PIL->RGB->np.array
            selfie_img = np.array(Image.open(selfie).convert("RGB"))
            id_img = np.array(Image.open(id_photo).convert("RGB"))

            # DeepFace.verify returns dict with "verified" bool and distance/confidence info
            result = DeepFace.verify(selfie_img, id_img, enforce_detection=True)
            if result.get("verified"):
                st.success("✅ Face Match Successful")
                st.write(result)
            else:
                st.error("❌ Face Mismatch Detected")
                st.write(result)
        except Exception as e:
            st.error(f"Error during verification: {e}")

# -------------------- MODULE 6: UNUSUAL PATTERN DETECTION --------------------
elif option == "Unusual Pattern Detection":
    st.header("📊 Unusual Pattern Detection")
    uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(data.head())

        # Only use numeric columns for z-score anomaly detection
        numeric = data.select_dtypes(include=[np.number]).copy()
        if numeric.shape[1] == 0:
            st.warning("No numeric columns found for anomaly detection.")
        else:
            z_scores = (numeric - numeric.mean()) / numeric.std(ddof=0)
            anomalies = data[(z_scores.abs() > 3).any(axis=1)]
            st.subheader("🔎 Detected Unusual Patterns:")
            if anomalies.shape[0] == 0:
                st.write("No anomalies detected by z-score (>3).")
            else:
                st.dataframe(anomalies)

# -------------------- REPORT SUMMARY --------------------
st.divider()
if st.button("Generate Fraud Report"):
    # Simple prototype response — you can expand this to save a PDF or Excel
    st.success("🧾 Fraud detection report generated successfully.")
    st.info("Accepted export preview: jpg, png, pdf, docx, xlsx, csv (prototype - add actual generation to save files).")
