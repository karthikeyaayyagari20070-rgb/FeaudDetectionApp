# app.py
# AI Fraud Detection System — Professional Light Theme (white background, black text)
# Save as app.py and run: streamlit run app.py

import streamlit as st
import io
import numpy as np
import pandas as pd
import re
from PIL import Image

# Guard heavy / optional imports so the app won't crash if not installed
try:
    import cv2
except Exception:
    cv2 = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

# easyocr cached resource (optional)
@st.cache_resource
def get_easyocr_reader():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

reader = get_easyocr_reader()

# Page config
st.set_page_config(
    page_title="AI Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------- PROFESSIONAL LIGHT THEME CSS -------
css = """
<style>
/* Page background and base typography */
[data-testid="stAppViewContainer"] {
    background: #ffffff;  /* pure white background */
    color-scheme: light;
}
body, .css-18e3th9, .stApp {
    color: #111827; /* near-black text for high contrast */
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* Card style (subtle, neutral) */
.card {
    background: #ffffff;
    border-radius: 10px;
    padding: 18px;
    border: 1px solid #e6e9ee; /* faint border for structure */
    box-shadow: 0 6px 18px rgba(16,24,40,0.04); /* subtle shadow */
    margin-bottom: 18px;
}

/* Header */
.header {
    display: flex;
    align-items: center;
    gap: 16px;
}
.logo {
    width: 56px;
    height: 56px;
    border-radius: 10px;
    background: linear-gradient(180deg,#111827,#374151); /* dark professional accent */
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 0.5px;
}

/* Subtext and muted text */
.small-muted {
    color: #6b7280; /* neutral muted gray */
    font-size: 13px;
}

/* Section title */
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a; /* very dark for headings */
    margin-bottom: 6px;
}

/* Keep input/button styles crisp */
.stButton>button {
    border-radius: 8px;
    padding: 8px 12px;
}

/* Footer */
.footer {
    color: #6b7280;
    font-size: 13px;
    margin-top: 18px;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ------- Header -------
with st.container():
    st.markdown(
        """
        <div class="header">
            <div class="logo">AI</div>
            <div>
                <div style="font-size:20px;font-weight:700;color:#0f172a">AI Fraud Detection System</div>
                <div class="small-muted">Document tampering • Signature verification • KYC • Anomaly detection</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------- Sidebar -------
st.sidebar.markdown("## Modules")
module = st.sidebar.radio(
    "",
    (
        "Document Tampering",
        "Signature Verification",
        "Aadhaar / PAN Checks",
        "AI-Based KYC Verification",
        "Unusual Pattern Detection",
        "Report & Settings",
    ),
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Tips:\n\n"
    "- Best results: high-resolution JPG/PNG for images, CSV/XLSX for transaction data.\n"
    "- Install optional packages for extra features: easyocr, deepface, pdf2image, python-docx, openpyxl.\n"
)
st.sidebar.markdown("---")
st.sidebar.markdown("Contact: dev@example.com")  # edit if you want

# ------- Utilities -------
def normalize_ssim_diff(diff):
    if diff is None or cv2 is None:
        return None
    diff_img = ((1 - diff) * 255).clip(0, 255).astype("uint8")
    heat = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat

def safe_to_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def read_image_bytes(file_bytes):
    if cv2 is None:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            return np.array(img)[:, :, ::-1]  # RGB -> BGR
        except Exception:
            return None
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def card(title, content_fn):
    st.markdown(f'<div class="card"><div class="section-title">{title}</div>', unsafe_allow_html=True)
    try:
        content_fn()
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

# ------- MODULES -------

# Document Tampering
if module == "Document Tampering":
    def doc_tamper():
        col1, col2 = st.columns(2)
        with col1:
            orig = st.file_uploader("Original Document (JPG/PNG)", type=["jpg", "jpeg", "png"], key="orig_doc")
        with col2:
            suspect = st.file_uploader("Suspected Document (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sus_doc")

        st.markdown("---")
        ssim_thresh = st.slider("SSIM similarity threshold (lower => stricter)", 0.50, 0.99, 0.85, step=0.01)

        if orig and suspect:
            orig_b = orig.read()
            sus_b = suspect.read()
            img1 = read_image_bytes(orig_b)
            img2 = read_image_bytes(sus_b)
            if img1 is None or img2 is None:
                st.error("Unable to decode uploaded images. Ensure OpenCV or Pillow is available.")
                return
            if ssim is None:
                st.warning("scikit-image is not installed — SSIM comparison unavailable.")
                return

            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            with st.spinner("Computing similarity..."):
                score, diff = ssim(gray1, gray2, full=True)
            st.metric("Similarity (SSIM)", f"{score:.4f}")
            heat = normalize_ssim_diff(diff)
            if heat is not None:
                st.image(heat, caption="Difference heatmap (red = more change)", use_column_width=True)
            if score < ssim_thresh:
                st.error("⚠ Possible tampering detected (similarity below threshold).")
            else:
                st.success("✅ No significant tampering detected.")

    card("Document Forgery Detection", doc_tamper)

# Signature Verification
elif module == "Signature Verification":
    def signature_module():
        col1, col2 = st.columns(2)
        with col1:
            s1 = st.file_uploader("Original Signature (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sig1")
        with col2:
            s2 = st.file_uploader("Submitted Signature (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sig2")

        st.markdown("---")
        dist_thresh = st.slider("Match distance threshold (lower => stricter)", 10, 100, 60)
        match_count_thresh = st.slider("Good matches threshold (higher => stricter)", 5, 60, 30)

        if s1 and s2:
            b1 = s1.read(); b2 = s2.read()
            if cv2 is None:
                st.error("OpenCV (cv2) is required for signature matching.")
                return
            img1 = cv2.imdecode(np.frombuffer(b1, np.uint8), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imdecode(np.frombuffer(b2, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img1 is None or img2 is None:
                st.error("Could not decode signature images.")
                return

            h = 220
            img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
            img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

            orb = cv2.ORB_create(5000)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                st.warning("Not enough features detected. Try higher-resolution or cleaner signature images.")
                return

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            good = [m for m in matches if m.distance < dist_thresh]
            st.metric("Good matches", len(good))
            st.write(f"Keypoints: {len(kp1)} vs {len(kp2)}")

            if len(good) >= match_count_thresh:
                st.success("✅ Signature likely genuine (heuristic).")
            else:
                st.error("❌ Signature may be forged or dissimilar (heuristic).")

    card("Signature Verification", signature_module)

# Aadhaar / PAN Checks
elif module == "Aadhaar / PAN Checks":
    def id_checks():
        st.markdown("### Aadhaar format check")
        aad = st.text_input("Aadhaar (XXXX-XXXX-XXXX or 12 digits)", key="aad")
        if st.button("Verify Aadhaar"):
            cleaned = aad.replace("-", "").strip()
            if len(cleaned) == 12 and cleaned.isdigit():
                st.success("✅ Aadhaar format OK (format-only).")
            else:
                st.error("❌ Aadhaar format invalid (12 digits required).")

        st.markdown("---")
        st.markdown("### PAN format check")
        pan = st.text_input("PAN (ABCDE1234F)", key="pan")
        if st.button("Validate PAN"):
            p = pan.strip().upper()
            if len(p) == 10 and p[:5].isalpha() and p[5:9].isdigit() and p[-1].isalpha():
                st.success("✅ PAN format OK (format-only).")
            else:
                st.error("❌ PAN format invalid. Example: ABCDE1234F")

    card("Aadhaar / PAN Format Checks", id_checks)

# AI-Based KYC
elif module == "AI-Based KYC Verification":
    def kyc_module():
        st.markdown("Upload selfie and ID photo to verify faces match.")
        col1, col2 = st.columns(2)
        with col1:
            selfie = st.file_uploader("Selfie (JPG/PNG)", type=["jpg","jpeg","png"], key="selfie")
        with col2:
            idphoto = st.file_uploader("ID Photo (JPG/PNG)", type=["jpg","jpeg","png"], key="idphoto")

        if selfie and idphoto:
            if DeepFace is None:
                st.error("DeepFace not installed. Install deepface to enable facial verification.")
                return
            try:
                b1 = selfie.read(); b2 = idphoto.read()
                img1 = Image.open(io.BytesIO(b1)).convert("RGB")
                img2 = Image.open(io.BytesIO(b2)).convert("RGB")
                arr1 = np.array(img1)
                arr2 = np.array(img2)
                with st.spinner("Running face verification..."):
                    result = DeepFace.verify(arr1, arr2, enforce_detection=True)
                verified = result.get("verified", False)
                st.metric("Face match", "Verified" if verified else "Not verified")
                st.write(result)
                if verified:
                    st.success("✅ Face match successful.")
                else:
                    st.error("❌ Face mismatch detected.")
            except Exception as e:
                st.error(f"Error during verification: {e}")

    card("AI-Based KYC Verification", kyc_module)

# Unusual Pattern Detection (Extended uploads)
elif module == "Unusual Pattern Detection":
    def anomaly_module():
        st.markdown("Upload transaction files. CSV/XLSX recommended. Images/PDF/DOCX supported via OCR heuristics.")
        uploaded = st.file_uploader("Upload file (csv, xlsx, jpg, png, pdf, docx)", type=["csv","xls","xlsx","jpg","jpeg","png","pdf","docx","doc"], key="anomaly_up")
        z_cutoff = st.slider("Z-score cutoff for anomaly (|z| > ...)", 2.0, 5.0, 3.0, step=0.1)

        def extract_csv(b): return pd.read_csv(io.BytesIO(b))
        def extract_excel(b): return pd.read_excel(io.BytesIO(b))

        def ocr_image_to_df(b):
            if reader is None:
                st.warning("easyocr not installed — install it for image/PDF OCR support.")
                return None
            img = read_image_bytes(b)
            if img is None: return None
            try:
                with st.spinner("Running OCR on image..."):
                    results = reader.readtext(img)
            except Exception as e:
                st.error(f"OCR failed: {e}")
                return None
            lines = [r[1] for r in results]
            rows=[]
            for line in lines:
                line_clean = re.sub(r"\s+", " ", line).strip()
                if "," in line_clean:
                    rows.append([p.strip() for p in line_clean.split(",")])
                else:
                    parts = [p for p in re.split(r"[\s\|;]+", line_clean) if p]
                    if len(parts) > 1: rows.append(parts)
            if not rows: return pd.DataFrame({"extracted_text": lines})
            counts=[len(r) for r in rows]; target=max(set(counts), key=counts.count)
            filtered=[r for r in rows if len(r)==target]; df=pd.DataFrame(filtered)
            for col in df.columns: df[col]=safe_to_numeric_series(df[col])
            return df

        def ocr_pdf_to_df(b):
            try:
                from pdf2image import convert_from_bytes
            except Exception:
                st.warning("pdf2image/poppler not installed — install for better PDF support.")
                return None
            try:
                with st.spinner("Converting PDF pages and running OCR..."):
                    pages = convert_from_bytes(b)
                    dfs=[]
                    for p in pages:
                        buf=io.BytesIO(); p.save(buf, format="JPEG")
                        df_page=ocr_image_to_df(buf.getvalue()); 
                        if df_page is not None: dfs.append(df_page)
                    if not dfs: return None
                    return pd.concat(dfs, ignore_index=True, sort=False)
            except Exception as e:
                st.error(f"PDF processing failed: {e}"); return None

        def docx_to_df(b):
            try:
                import docx
            except Exception:
                st.warning("python-docx not installed — install for DOCX support.")
                return None
            try:
                doc = docx.Document(io.BytesIO(b)); paras=[p.text for p in doc.paragraphs if p.text.strip()]
                rows=[]
                for ptext in paras:
                    line=re.sub(r"\s+"," ",ptext).strip()
                    if "," in line: rows.append([c.strip() for c in line.split(",")])
                    else:
                        parts=[x for x in re.split(r"[\s\|;]+", line) if x]
                        if len(parts)>1: rows.append(parts)
                if not rows: return pd.DataFrame({"extracted_text": paras})
                counts=[len(r) for r in rows]; target=max(set(counts), key=counts.count)
                filtered=[r for r in rows if len(r)==target]; df=pd.DataFrame(filtered)
                for col in df.columns: df[col]=safe_to_numeric_series(df[col])
                return df
            except Exception as e:
                st.error(f"DOCX parse failed: {e}"); return None

        if uploaded:
            content = uploaded.read(); name = uploaded.name.lower(); data=None
            try:
                if name.endswith(".csv"):
                    data = extract_csv(content)
                    data = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="ignore"))
                elif name.endswith((".xls","xlsx")):
                    data = extract_excel(content)
                    data = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="ignore"))
                elif name.endswith((".jpg","jpeg","png")):
                    data = ocr_image_to_df(content)
                elif name.endswith(".pdf"):
                    data = ocr_pdf_to_df(content)
                elif name.endswith((".docx","doc")):
                    data = docx_to_df(content)
                else:
                    st.error("Unsupported file type.")
            except Exception as e:
                st.error(f"Failed to parse uploaded file: {e}"); data=None

            if data is None:
                st.info("Could not extract table from file. CSV/XLSX are best for anomaly detection.")
                return

            st.subheader("Parsed preview")
            st.dataframe(data.head(10))

            numeric = data.select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                coerced = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"))
                numeric = coerced.select_dtypes(include=[np.number])
                if numeric.shape[1] > 0:
                    for c in numeric.columns: data[c]=coerced[c]

            if numeric.shape[1] == 0:
                st.warning("No numeric columns found — anomaly detection requires numeric data.")
                return

            with st.spinner("Computing anomalies..."):
                z = (numeric - numeric.mean()) / numeric.std(ddof=0)
                mask = (z.abs() > z_cutoff).any(axis=1)
                anomalies = data[mask]

            st.subheader("Detected anomalies")
            if anomalies.shape[0] == 0:
                st.write("No anomalies found with current cutoff.")
            else:
                st.dataframe(anomalies)
                st.download_button("Download anomalies (CSV)", data=anomalies.to_csv(index=False).encode(), file_name="anomalies.csv", mime="text/csv")

    card("Unusual Pattern Detection", anomaly_module)

# Report & Settings
elif module == "Report & Settings":
    def report_settings():
        st.markdown("## Report & Export")
        st.markdown("This is a prototype area. You can implement full PDF/XLSX reports that assemble screenshots, anomaly tables, and verification results.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate sample report (prototype)"):
                st.success("Prototype: implement PDF builder to produce final report.")
        with col2:
            if st.button("Dependencies checklist"):
                st.info(
                    "- Required: streamlit, numpy, pandas, pillow\n"
                    "- Recommended: opencv-python, scikit-image, openpyxl\n"
                    "- Optional: easyocr, deepface, pdf2image, python-docx\n"
                    "- OS: poppler for pdf2image"
                )
        st.markdown("---")
        st.markdown("### Advanced (developer)")
        with st.expander("Developer options"):
            st.write("Tune or persist thresholds here in the future.")

    card("Report & Settings", report_settings)

# Footer
st.markdown('<div class="footer">Made with care — AI Fraud Detection Demo (light theme)</div>', unsafe_allow_html=True)
