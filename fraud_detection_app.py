# app.py
# ==============================
# AI Fraud Detection System (updated)
# ==============================

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from deepface import DeepFace
from PIL import Image
import io
import re

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="AI Fraud Detection System", layout="wide")
st.title("🧠 AI Fraud Detection System")
st.write("Upload documents to verify authenticity and detect fraud.")

# ---------------- Helpers & Cached Resources ----------------
@st.cache_resource
def get_easyocr_reader():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

reader = get_easyocr_reader()

def safe_to_numeric_series(s):
    # Convert to string first, remove non-numeric chars (except . and -) and coerce
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def normalize_ssim_diff(diff):
    # SSIM diff range is typically [-1,1] or [0,1]; convert to 0-255 uint8 for visualization
    diff_image = ((1 - diff) * 255).clip(0, 255).astype("uint8")
    heatmap = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
    # convert BGR (OpenCV) to RGB (PIL/Streamlit)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap

# ---------------- Sidebar modules ----------------
option = st.sidebar.selectbox(
    "Choose Module",
    [
        "Document Tampering",
        "Signature Verification",
        "Aadhaar Fraud Detection",
        "PAN Fraud Detection",
        "AI-Based KYC Verification",
        "Unusual Pattern Detection",
    ],
)

# -------------------- MODULE 1: DOCUMENT TAMPERING --------------------
if option == "Document Tampering":
    st.header("📄 Document Forgery Detection")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_doc1 = st.file_uploader("Upload Original Document", type=["jpg", "png", "jpeg"])
    with col2:
        uploaded_doc2 = st.file_uploader("Upload Suspected Document", type=["jpg", "png", "jpeg"])

    if uploaded_doc1 and uploaded_doc2:
        file1_bytes = uploaded_doc1.read()
        file2_bytes = uploaded_doc2.read()
        arr1 = np.frombuffer(file1_bytes, np.uint8)
        arr2 = np.frombuffer(file2_bytes, np.uint8)
        img1 = cv2.imdecode(arr1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(arr2, cv2.IMREAD_COLOR)

        if img1 is None or img2 is None:
            st.error("Could not decode one of the uploaded images.")
        else:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            score, diff = ssim(gray1, gray2, full=True)
            st.write(f"🔍 Similarity Score: {score:.4f}")

            heatmap = normalize_ssim_diff(diff)
            st.image(heatmap, caption="Difference Heatmap", use_container_width=True)

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
        sig1_bytes = sig1_file.read()
        sig2_bytes = sig2_file.read()
        sig1 = cv2.imdecode(np.frombuffer(sig1_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        sig2 = cv2.imdecode(np.frombuffer(sig2_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if sig1 is None or sig2 is None:
            st.error("Could not decode one of the signature images.")
        else:
            # Normalize height for both signatures to help feature matching
            h = 200
            sig1 = cv2.resize(sig1, (int(sig1.shape[1] * h / sig1.shape[0]), h))
            sig2 = cv2.resize(sig2, (int(sig2.shape[1] * h / sig2.shape[0]), h))

            orb = cv2.ORB_create(5000)
            kp1, des1 = orb.detectAndCompute(sig1, None)
            kp2, des2 = orb.detectAndCompute(sig2, None)

            if des1 is not None and des2 is not None and len(kp1) > 0 and len(kp2) > 0:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                good_matches = [m for m in matches if m.distance < 60]  # heuristic
                score = len(good_matches)
                st.write(f"Good Match Count: {score} (keypoints: {len(kp1)}, {len(kp2)})")

                if score > 30:
                    st.success("✅ Genuine Signature (high similarity)")
                else:
                    st.error("❌ Signature may be forged or not enough similarity.")
            else:
                st.warning("Could not detect enough features in one or both signatures.")

# -------------------- MODULE 3: AADHAAR FRAUD DETECTION --------------------
elif option == "Aadhaar Fraud Detection":
    st.header("🪪 Aadhaar Fraud Verification (Prototype)")
    aadhaar_num = st.text_input("Enter Aadhaar Number (format XXXX-XXXX-XXXX or 12 digits):")
    if st.button("Verify Aadhaar"):
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
            selfie_img = np.array(Image.open(io.BytesIO(selfie.read())).convert("RGB"))
            id_img = np.array(Image.open(io.BytesIO(id_photo.read())).convert("RGB"))

            # DeepFace verification
            with st.spinner("Analyzing faces..."):
                result = DeepFace.verify(selfie_img, id_img, enforce_detection=True)
            if result.get("verified"):
                st.success("✅ Face Match Successful")
                st.write(result)
            else:
                st.error("❌ Face Mismatch Detected")
                st.write(result)
        except Exception as e:
            st.error(f"Error during verification: {e}")

# -------------------- MODULE 6: UNUSUAL PATTERN DETECTION (EXTENDED UPLOADS) --------------------
elif option == "Unusual Pattern Detection":
    st.header("📊 Unusual Pattern Detection — Extended Uploads")

    uploaded_file = st.file_uploader(
        "Upload transaction data (CSV, XLSX, JPG, PNG, PDF, DOCX)",
        type=["csv", "xls", "xlsx", "jpg", "png", "jpeg", "pdf", "docx", "doc"],
    )

    def extract_dataframe_from_csv_bytes(b):
        return pd.read_csv(io.BytesIO(b))

    def extract_dataframe_from_excel_bytes(b):
        return pd.read_excel(io.BytesIO(b))

    def ocr_image_to_dataframe(image_bytes):
        if reader is None:
            st.warning("easyocr is not available. Install easyocr to enable OCR from images.")
            return None
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image for OCR")
        with st.spinner("Running OCR on image..."):
            try:
                ocr_results = reader.readtext(img)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                return None

        lines = [t[1] for t in ocr_results]
        rows = []
        for line in lines:
            line_clean = re.sub(r"\s+", " ", line).strip()
            if "," in line_clean:
                parts = [p.strip() for p in line_clean.split(",") if p.strip()]
                rows.append(parts)
            else:
                parts = [p for p in re.split(r"[\s\|;]+", line_clean) if p]
                if len(parts) > 1:
                    rows.append(parts)

        if rows:
            counts = [len(r) for r in rows]
            target = max(set(counts), key=counts.count)
            filtered = [r for r in rows if len(r) == target]
            df = pd.DataFrame(filtered)
            for col in df.columns:
                df[col] = safe_to_numeric_series(df[col])
            return df

        return pd.DataFrame({"extracted_text": lines})

    def ocr_pdf_to_dataframe(pdf_bytes):
        try:
            from pdf2image import convert_from_bytes
        except Exception:
            st.warning("pdf2image or its system dependency (poppler) is not available. Install them for PDF table extraction.")
            return None
        try:
            with st.spinner("Converting PDF pages to images and running OCR..."):
                pages = convert_from_bytes(pdf_bytes)
                combined = []
                for page_num, page in enumerate(pages, start=1):
                    img_bytes = io.BytesIO()
                    page.save(img_bytes, format="JPEG")
                    img_b = img_bytes.getvalue()
                    df_page = ocr_image_to_dataframe(img_b)
                    if df_page is not None:
                        combined.append(df_page)
                if combined:
                    return pd.concat(combined, ignore_index=True, sort=False)
                else:
                    return None
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None

    def docx_to_dataframe(docx_bytes):
        try:
            import docx
        except Exception:
            st.warning("python-docx is not installed. Install python-docx to support DOCX text extraction.")
            return None
        try:
            doc = docx.Document(io.BytesIO(docx_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            rows = []
            for p_text in paragraphs:
                line = re.sub(r"\s+", " ", p_text).strip()
                if "," in line:
                    rows.append([c.strip() for c in line.split(",")])
                else:
                    parts = [x for x in re.split(r"[\s\|;]+", line) if x]
                    if len(parts) > 1:
                        rows.append(parts)
            if rows:
                counts = [len(r) for r in rows]
                target = max(set(counts), key=counts.count)
                filtered = [r for r in rows if len(r) == target]
                df = pd.DataFrame(filtered)
                for col in df.columns:
                    df[col] = safe_to_numeric_series(df[col])
                return df
            return pd.DataFrame({"extracted_text": paragraphs})
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return None

    if uploaded_file:
        # read bytes once (prevents stream exhaustion)
        file_bytes = uploaded_file.read()
        fname = uploaded_file.name.lower()

        data = None
        try:
            if fname.endswith(".csv"):
                data = extract_dataframe_from_csv_bytes(file_bytes)
                # attempt to coerce numeric-like columns
                data = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="ignore"))
            elif fname.endswith((".xls", ".xlsx")):
                data = extract_dataframe_from_excel_bytes(file_bytes)
                data = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="ignore"))
            elif fname.endswith((".jpg", ".jpeg", ".png")):
                data = ocr_image_to_dataframe(file_bytes)
            elif fname.endswith(".pdf"):
                data = ocr_pdf_to_dataframe(file_bytes)
                if data is None:
                    st.error("Could not extract tables from PDF. Install pdf2image and poppler for better support.")
            elif fname.endswith((".docx", ".doc")):
                data = docx_to_dataframe(file_bytes)
                if data is None:
                    st.error("Could not extract data from DOC/DOCX. Install python-docx to enable this feature.")
            else:
                st.error("Unsupported file type")
                data = None
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            data = None

        if data is not None:
            st.subheader("Preview of parsed data")
            st.dataframe(data.head())

            # Ensure numeric columns are numeric (coerce strings)
            numeric = data.select_dtypes(include=[np.number])
            # If numeric dtype not detected but columns look numeric-like, try coercion
            if numeric.shape[1] == 0:
                # attempt to coerce all columns
                coerced = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"))
                numeric = coerced.select_dtypes(include=[np.number])
                # keep coerced numeric values in data for anomaly reporting
                data = data.assign({c: coerced[c] for c in numeric.columns})

            if numeric.shape[1] == 0:
                st.warning("No numeric columns found for anomaly detection. The uploaded file was parsed as text — try CSV/XLSX for best results.")
            else:
                z_scores = (numeric - numeric.mean()) / numeric.std(ddof=0)
                anomalies = data[(z_scores.abs() > 3).any(axis=1)]
                st.subheader("🔎 Detected Unusual Patterns:")
                if anomalies.shape[0] == 0:
                    st.write("No anomalies detected by z-score (>3).")
                else:
                    st.dataframe(anomalies)
        else:
            st.info("No dataframe could be extracted from the uploaded file. Try uploading CSV/XLSX for the most accurate results.")

# -------------------- REPORT SUMMARY / EXPORT (Prototype) --------------------
st.divider()
if st.button("Generate Fraud Report"):
    # For now, prototype: you can extend this to generate a PDF/Excel summary and offer download
    st.success("🧾 Fraud detection report generated successfully.")
    st.info("Accepted export preview: jpg, png, pdf, docx, xlsx, csv (prototype - add actual generation to save files).")

# ---------------- End of app ----------------
