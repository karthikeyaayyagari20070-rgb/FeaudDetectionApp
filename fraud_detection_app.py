# =============================================================
# AI Fraud Detection System (Final Version with Dashboard)
# Author: Karthikeya Ayyagari
# =============================================================

import os
import sys
import subprocess

# ---------- AUTO-INSTALL MODULES ----------
required = [
    "streamlit", "opencv-python-headless", "easyocr", "numpy", "Pillow",
    "scikit-image", "pdf2image", "pytesseract", "deepface", "pandas"
]
for module in required:
    try:
        __import__(module)
    except ImportError:
        print(f"Installing {module}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

# ---------- IMPORTS ----------
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace
import easyocr
import pandas as pd
import datetime

# ---------- APP CONFIG ----------
st.set_page_config(page_title="AI Fraud Detection System", layout="wide")
st.title("🧠 AI Fraud Detection System")
st.markdown("A unified platform for detecting document forgery, signature fraud, KYC mismatch, and more.")

# ---------- HISTORY ----------
if "history" not in st.session_state:
    st.session_state["history"] = []

def log_result(module, result):
    st.session_state["history"].append({
        "Module": module,
        "Result": result,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# ---------- UTILITIES ----------
def load_image(file):
    if file.type == "application/pdf":
        pages = convert_from_path(file)
        return np.array(pages[0])
    else:
        return np.array(Image.open(file))

def detect_text(img):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(img)
    return " ".join([r[1] for r in results])

def detect_aadhaar(text):
    import re
    match = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)
    return f"Aadhaar: {match.group()}" if match else "No Aadhaar number found."

def detect_pan(text):
    import re
    match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    return f"PAN: {match.group()}" if match else "No PAN number found."

def tampering_check(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (300, 300))
    gray2 = cv2.resize(gray2, (300, 300))
    score, _ = ssim(gray1, gray2, full=True)
    return round(score * 100, 2)

def signature_check(sig1, sig2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(sig1, None)
    kp2, des2 = orb.detectAndCompute(sig2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return round(len(matches) / max(len(kp1), len(kp2)) * 100, 2)

def kyc_check(face1, face2):
    try:
        result = DeepFace.verify(face1, face2, model_name="Facenet")
        return "✅ Faces Match (KYC Passed)" if result["verified"] else "❌ Faces Do Not Match"
    except:
        return "⚠️ Face Verification Error"

def pattern_check(text):
    risky = ["copy", "forged", "duplicate", "tamper", "fake"]
    found = [word for word in risky if word in text.lower()]
    return f"⚠️ Suspicious words detected: {', '.join(found)}" if found else "✅ No suspicious patterns."

# ---------- SIDEBAR ----------
option = st.sidebar.radio(
    "Select Module:",
    [
        "Aadhaar / PAN Verification",
        "Document Tampering",
        "Signature Verification",
        "AI KYC Verification",
        "Unusual Pattern Detection",
        "📊 Fraud Analysis Dashboard"
    ]
)

# ---------- MODULES ----------
if option == "Aadhaar / PAN Verification":
    doc = st.file_uploader("Upload Document", type=["jpg", "png", "jpeg", "pdf"])
    if doc:
        img = load_image(doc)
        st.image(img, caption="Uploaded Document", use_column_width=True)
        text = detect_text(img)
        aadhaar = detect_aadhaar(text)
        pan = detect_pan(text)
        st.write(aadhaar)
        st.write(pan)
        log_result("Aadhaar/PAN Verification", f"{aadhaar}, {pan}")

elif option == "Document Tampering":
    col1, col2 = st.columns(2)
    with col1:
        original = st.file_uploader("Upload Original Document", type=["jpg", "png", "jpeg", "pdf"])
    with col2:
        suspect = st.file_uploader("Upload Suspected Document", type=["jpg", "png", "jpeg", "pdf"])
    if original and suspect:
        img1 = load_image(original)
        img2 = load_image(suspect)
        score = tampering_check(img1, img2)
        st.image([img1, img2], caption=["Original", "Suspect"], width=300)
        st.write(f"Similarity Score: **{score}%**")
        if score < 85:
            st.error("⚠️ Possible Tampering Detected!")
        else:
            st.success("✅ No Major Tampering Detected")
        log_result("Document Tampering", f"Similarity: {score}%")

elif option == "Signature Verification":
    col1, col2 = st.columns(2)
    with col1:
        sig1 = st.file_uploader("Upload Original Signature", type=["jpg", "png", "jpeg"])
    with col2:
        sig2 = st.file_uploader("Upload Suspected Signature", type=["jpg", "png", "jpeg"])
    if sig1 and sig2:
        img1 = np.array(Image.open(sig1))
        img2 = np.array(Image.open(sig2))
        score = signature_check(img1, img2)
        st.image([img1, img2], caption=["Original", "Suspected"], width=300)
        st.info(f"Signature Match Score: {score}%")
        if score < 50:
            st.warning("⚠️ Possible Forgery Detected!")
        else:
            st.success("✅ Signature Verified")
        log_result("Signature Verification", f"Match: {score}%")

elif option == "AI KYC Verification":
    col1, col2 = st.columns(2)
    with col1:
        face1 = st.file_uploader("Upload ID Proof Face", type=["jpg", "png", "jpeg"])
    with col2:
        face2 = st.file_uploader("Upload Live Photo", type=["jpg", "png", "jpeg"])
    if face1 and face2:
        st.image([Image.open(face1), Image.open(face2)], caption=["ID Face", "Live Face"], width=300)
        result = kyc_check(face1, face2)
        st.write(result)
        log_result("AI KYC Verification", result)

elif option == "Unusual Pattern Detection":
    doc = st.file_uploader("Upload Document", type=["jpg", "png", "jpeg", "pdf"])
    if doc:
        img = load_image(doc)
        text = detect_text(img)
        st.text_area("Extracted Text", text)
        result = pattern_check(text)
        st.write(result)
        log_result("Unusual Pattern Detection", result)

elif option == "📊 Fraud Analysis Dashboard":
    st.subheader("📈 Fraud Detection History")
    if st.session_state["history"]:
        df = pd.DataFrame(st.session_state["history"])
        st.dataframe(df)
    else:
        st.info("No history yet. Run a module first!")

# ---------- FOOTER ----------
st.markdown("---")
st.caption("© 2025 AI Fraud Detection | Developed by Karthikeya Ayyagari")
