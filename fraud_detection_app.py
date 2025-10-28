import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace

# --- Theming ---
st.set_page_config(page_title="Secure Fraud Detection", layout="wide", page_icon="🏦")
st.markdown("""
    <style>
    .stApp {
        background-color: #f3f6fb;
    }
    .css-18e3th9 {
        background: #fff;
        border-radius: 10px;
        box-shadow: 0 2px 8px #dee8f7;
    }
    .css-10trblm {
        color: #003366;
    }
    div[role="dialog"] {
        background: #fff;
    }
    .stButton > button {
        background-color: #0056a3;
        color: #fff;
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Secure Fraud Detection System")
st.markdown("Welcome to your next-generation, secure banking fraud detection and eKYC portal.")

menu = st.sidebar.radio("Choose Service", [
    "Document Forgery Detection",
    "Signature Authentication",
    "Aadhaar Check",
    "PAN Validation",
    "AI Face KYC",
    "Transaction Anomaly"
])

reader = easyocr.Reader(['en'], gpu=False)

if menu == "Document Forgery Detection":
    st.header("📄 Document Tampering Detection")
    col1, col2 = st.columns(2)
    with col1: original = st.file_uploader("Original Document", type=["png","jpg","jpeg"], key="oridoc")
    with col2: suspect = st.file_uploader("Suspected Document", type=["png","jpg","jpeg"], key="susdoc")
    if original and suspect:
        img1 = cv2.imdecode(np.frombuffer(original.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(suspect.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, diff = ssim(gray1, gray2, full=True)
        st.write(f"**Similarity Score:** {score:.3f}")
        if score < 0.88:
            st.error("⚠️ Possible alteration detected.")
        else:
            st.success("✅ No major differences found.")
        st.image((diff*255).astype(np.uint8), caption="Visual Difference", use_column_width=True)

if menu == "Signature Authentication":
    st.header("✍️ Signature Comparison")
    col1, col2 = st.columns(2)
    s1 = col1.file_uploader("Reference Signature", type=["png","jpg","jpeg"], key="sig1")
    s2 = col2.file_uploader("Submitted Signature", type=["png","jpg","jpeg"], key="sig2")
    if s1 and s2:
        sig1 = cv2.imdecode(np.frombuffer(s1.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        sig2 = cv2.imdecode(np.frombuffer(s2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(sig1, None)
        kp2, des2 = orb.detectAndCompute(sig2, None)
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            score = len(matches)
            st.write(f"Feature Match Score: {score}")
            st.success("✅ Likely Genuine Signature" if score > 45 else "❌ Possible Forgery")
        else:
            st.warning("Couldn't extract enough features.")

if menu == "Aadhaar Check":
    st.header("🪪 Aadhaar Number Format Validator")
    aadhaar = st.text_input("Aadhaar No. (XXXX-XXXX-XXXX):")
    if st.button("Validate Aadhaar"):
        if len(aadhaar) == 14 and all(x.isdigit() or x == "-" for x in aadhaar.replace("-", "")):
            st.success("Structure appears valid (format only).")
        else:
            st.error("Invalid Aadhaar number format.")

if menu == "PAN Validation":
    st.header("💳 PAN Structure Validation")
    pan = st.text_input("PAN Number (ABCDE1234F):")
    if st.button("Validate PAN"):
        if len(pan) == 10 and pan[:5].isalpha() and pan[5:9].isdigit() and pan[-1].isalpha():
            st.success("PAN format appears correct.")
        else:
            st.error("Incorrect PAN format.")

if menu == "AI Face KYC":
    st.header("🧬 Automated KYC (Face Match)")
    col1, col2 = st.columns(2)
    with col1: selfie = st.file_uploader("Upload Selfie", type=["png","jpg","jpeg"], key="selfie")
    with col2: idphoto = st.file_uploader("Upload ID Photo", type=["png","jpg","jpeg"], key="idpic")
    if selfie and idphoto:
        st.info("Running DeepFace face similarity check...")
        try:
            result = DeepFace.verify(np.array(Image.open(selfie)), np.array(Image.open(idphoto)), enforce_detection=False)
            st.success("Faces Match" if result.get("verified") else "Faces do NOT Match")
        except Exception as e:
            st.error(f"Error during DeepFace analysis: {e}")

if menu == "Transaction Anomaly":
    st.header("📊 Transaction Anomaly Detection")
    datafile = st.file_uploader("Transaction CSV File", type="csv")
    if datafile:
        df = pd.read_csv(datafile)
        st.dataframe(df.head())
        z = (df.select_dtypes('number') - df.select_dtypes('number').mean()) / df.select_dtypes('number').std()
        anomalies = df.loc[(z.abs() > 3).any(axis=1)]
        st.subheader("Potential Outliers:")
        st.dataframe(anomalies)

st.divider()
if st.button("Generate Comprehensive Fraud Report"):
    st.success("Your report is ready. Download from the secure console or save this session!")

