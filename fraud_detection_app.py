# app.py
# AI Fraud Detection — patched: safe image bytes + robust PDF generation
# Run: streamlit run app.py

import streamlit as st
import io
import numpy as np
import pandas as pd
import re
from PIL import Image
import base64
import traceback

# -------------------- Guarded imports --------------------
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

@st.cache_resource
def get_easyocr_reader():
    try:
        import easyocr
        return easyocr.Reader(["en"], gpu=False)
    except Exception:
        return None

reader = get_easyocr_reader()

try:
    from pdf2image import convert_from_bytes
    pdf2image_available = True
except Exception:
    convert_from_bytes = None
    pdf2image_available = False

try:
    import docx
    docx_available = True
except Exception:
    docx = None
    docx_available = False

# reportlab for PDF generation (optional)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    reportlab_available = True
except Exception:
    reportlab_available = False

# -------------------- Page config + CSS --------------------
st.set_page_config(page_title="AI Fraud Detection", layout="wide")
PERSIAN_BLUE = "#1C39BB"

THEME_CSS = f"""
<style>
[data-testid="stAppViewContainer"] {{ background: #ffffff; color: #0f172a; }}
body, .stApp {{ color: #0f172a; font-family: Inter, Roboto, Arial, sans-serif; }}
[data-testid="stSidebar"] {{ background: {PERSIAN_BLUE} !important; color: #000000 !important; }}
.card {{ background: #ffffff; border-radius:10px; padding:18px; border:1px solid #eef2f7; box-shadow:0 6px 18px rgba(16,24,40,0.04); margin-bottom:18px; }}
.section-title {{ font-size:18px; font-weight:700; color:#0f172a; margin-bottom:8px; }}
.stButton>button {{ background: {PERSIAN_BLUE} !important; color: #000000 !important; border-radius:8px; padding:8px 12px; border:none; }}
input[type="range"] {{ -webkit-appearance:none; width:100%; height:8px; border-radius:8px; background:#e6e6e6; outline:none; }}
input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance:none; appearance:none; width:18px; height:18px; border-radius:50%; background:{PERSIAN_BLUE}; cursor:pointer; border:2px solid #ffffff; box-shadow:0 2px 6px rgba(0,0,0,0.15); }}
input[type="range"]::-moz-range-thumb {{ width:18px; height:18px; border-radius:50%; background:{PERSIAN_BLUE}; border:2px solid #ffffff; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio label {{ color: #000000 !important; }}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="display:flex; gap:14px; align-items:center; margin-bottom:14px;">
      <div style="width:56px;height:56px;border-radius:10px;background:{PERSIAN_BLUE};display:flex;align-items:center;justify-content:center;color:#000;font-weight:700;">AI</div>
      <div>
        <div style="font-size:20px;font-weight:700;color:#0f172a">AI Fraud Detection System</div>
        <div style="color:#6b7280;font-size:13px">Document tampering · Signature verification · KYC · Anomaly detection</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Utilities --------------------
def safe_to_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def read_image_bytes_to_bgr(file_bytes):
    if file_bytes is None:
        return None
    if cv2 is None:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            arr = np.array(img)[:, :, ::-1]
            return arr
        except Exception:
            return None
    try:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def normalize_ssim_diff(diff):
    if diff is None or cv2 is None:
        return None
    diff_img = ((1 - diff) * 255).clip(0, 255).astype("uint8")
    heat = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat

def encode_image_to_png_bytes(np_or_pil):
    """
    Accepts numpy array (RGB) or PIL.Image and returns PNG bytes.
    Returns None on failure.
    """
    try:
        if isinstance(np_or_pil, np.ndarray):
            img = Image.fromarray(np_or_pil)
        else:
            img = np_or_pil  # assume PIL
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None

def put_session_result(key, value):
    st.session_state.setdefault("results", {})
    st.session_state["results"][key] = value

def get_session_results():
    return st.session_state.get("results", {})

# ensure results container
if "results" not in st.session_state:
    st.session_state["results"] = {}

# -------------------- Sidebar & Modules --------------------
st.sidebar.markdown("## Modules", unsafe_allow_html=True)
module = st.sidebar.radio(
    "",
    [
        "Document Tampering",
        "Signature Verification",
        "Aadhaar / PAN Checks",
        "AI-Based KYC Verification",
        "Unusual Pattern Detection",
        "Report & Export"
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "- Use high-res JPG/PNG for images. CSV/XLSX recommended for transactions.\n- Install optional packages for OCR & PDF export: easyocr, deepface, pdf2image, python-docx, reportlab."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Contact: dev@example.com")

# -------------------- Document Tampering --------------------
if module == "Document Tampering":
    st.markdown('<div class="card"><div class="section-title">📄 Document Forgery Detection</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        uploaded_orig = st.file_uploader("Original Document (JPG/PNG)", type=["jpg", "jpeg", "png"], key="orig")
    with col2:
        uploaded_sus = st.file_uploader("Suspected Document (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sus")

    ssim_thresh = st.slider("SSIM similarity threshold", 0.50, 0.99, 0.85, step=0.01, key="ssim_threshold")

    if uploaded_orig and uploaded_sus:
        orig_bytes = uploaded_orig.read()
        sus_bytes = uploaded_sus.read()
        img1 = read_image_bytes_to_bgr(orig_bytes)
        img2 = read_image_bytes_to_bgr(sus_bytes)

        if img1 is None or img2 is None:
            st.error("Unable to decode images. Install OpenCV or ensure files are valid images.")
            put_session_result("document_tamper", {"status": "error", "message": "decode_error"})
        elif ssim is None:
            st.error("SSIM requires scikit-image. Install with pip install scikit-image.")
            put_session_result("document_tamper", {"status": "error", "message": "ssim_missing"})
        else:
            try:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                with st.spinner("Computing similarity..."):
                    score, diff = ssim(gray1, gray2, full=True)
                verdict = "No significant alteration" if score >= ssim_thresh else "Possible forgery detected"
                st.write(f"🔍 Similarity Score (SSIM): *{score:.4f}*")
                st.write(f"Result: *{verdict}*")
                heat = normalize_ssim_diff(diff)
                diff_png = None
                if heat is not None:
                    # encode heat numpy array -> PNG bytes for storage & PDF
                    diff_png = encode_image_to_png_bytes(heat)
                    st.image(heat, caption="Difference heatmap", use_column_width=True)
                put_session_result("document_tamper", {
                    "status": "ok",
                    "score": float(score),
                    "verdict": verdict,
                    "orig_image": orig_bytes,       # uploaded file bytes (OK)
                    "sus_image": sus_bytes,         # uploaded file bytes (OK)
                    "diff_image": diff_png          # PNG bytes or None
                })
            except Exception:
                st.error("Error computing similarity.")
                st.write(traceback.format_exc())
                put_session_result("document_tamper", {"status": "error", "message": "ssim_exception"})
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Signature Verification --------------------
elif module == "Signature Verification":
    st.markdown('<div class="card"><div class="section-title">✍ Signature Verification</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        sig_orig = st.file_uploader("Original Signature (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sig_orig")
    with col2:
        sig_sub = st.file_uploader("Submitted Signature (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sig_sub")

    dist_thresh = st.slider("Match distance threshold", 5, 120, 60, key="sig_dist")
    match_count_thresh = st.slider("Good matches threshold", 1, 200, 30, key="sig_matches")

    if sig_orig and sig_sub:
        b1 = sig_orig.read()
        b2 = sig_sub.read()
        if cv2 is None:
            st.error("OpenCV required for signature matching. Install: pip install opencv-python")
            put_session_result("signature", {"status": "error", "message": "cv2_missing"})
        else:
            try:
                img1 = cv2.imdecode(np.frombuffer(b1, np.uint8), cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imdecode(np.frombuffer(b2, np.uint8), cv2.IMREAD_GRAYSCALE)
                if img1 is None or img2 is None:
                    st.error("Could not decode signatures.")
                    put_session_result("signature", {"status": "error", "message": "decode_error"})
                else:
                    h = 220
                    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
                    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
                    orb = cv2.ORB_create(5000)
                    kp1, des1 = orb.detectAndCompute(img1, None)
                    kp2, des2 = orb.detectAndCompute(img2, None)
                    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                        st.warning("Not enough features detected in one or both signatures.")
                        put_session_result("signature", {"status": "error", "message": "not_enough_features"})
                    else:
                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)
                        good = [m for m in matches if m.distance < dist_thresh]
                        st.write(f"Good matches: {len(good)}")
                        verdict = "Genuine signature (heuristic)" if len(good) >= match_count_thresh else "Possible forgery or mismatch"
                        if len(good) >= match_count_thresh:
                            st.success(f"✅ {verdict}")
                        else:
                            st.error(f"❌ {verdict}")
                        # store images as uploaded bytes (b1,b2)
                        put_session_result("signature", {
                            "status": "ok",
                            "good_matches": len(good),
                            "verdict": verdict,
                            "orig_image": b1,
                            "sub_image": b2
                        })
            except Exception:
                st.error("Signature comparison failed.")
                st.write(traceback.format_exc())
                put_session_result("signature", {"status": "error", "message": "exception"})
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Aadhaar / PAN --------------------
elif module == "Aadhaar / PAN Checks":
    st.markdown('<div class="card"><div class="section-title">🪪 Aadhaar & PAN Format Checks</div>', unsafe_allow_html=True)
    aadhaar_num = st.text_input("Aadhaar (XXXX-XXXX-XXXX or 12 digits)", key="aad")
    if st.button("Verify Aadhaar"):
        cleaned = aadhaar_num.replace("-", "").strip()
        if len(cleaned) == 12 and cleaned.isdigit():
            st.success("✅ Aadhaar format looks valid (format-only).")
            put_session_result("aadhaar", {"status": "ok", "value": cleaned, "verdict": "valid"})
        else:
            st.error("❌ Invalid Aadhaar format.")
            put_session_result("aadhaar", {"status": "ok", "value": cleaned, "verdict": "invalid"})
    st.markdown("---")
    pan_num = st.text_input("PAN (ABCDE1234F)", key="pan")
    if st.button("Validate PAN"):
        p = pan_num.strip().upper()
        if len(p) == 10 and p[:5].isalpha() and p[5:9].isdigit() and p[-1].isalpha():
            st.success("✅ PAN format OK (format-only).")
            put_session_result("pan", {"status": "ok", "value": p, "verdict": "valid"})
        else:
            st.error("❌ PAN format invalid.")
            put_session_result("pan", {"status": "ok", "value": p, "verdict": "invalid"})
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- AI-Based KYC --------------------
elif module == "AI-Based KYC Verification":
    st.markdown('<div class="card"><div class="section-title">🧬 AI-Based KYC Verification</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        selfie_file = st.file_uploader("Upload Selfie Photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="selfie")
    with col2:
        id_file = st.file_uploader("Upload ID Photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="idphoto")

    if selfie_file and id_file:
        if DeepFace is None:
            st.error("DeepFace not installed. Install deepface to enable face verification.")
            put_session_result("kyc", {"status": "error", "message": "deepface_missing"})
        else:
            try:
                s_bytes = selfie_file.read()
                id_bytes = id_file.read()
                img1 = np.array(Image.open(io.BytesIO(s_bytes)).convert("RGB"))
                img2 = np.array(Image.open(io.BytesIO(id_bytes)).convert("RGB"))
                with st.spinner("Running face verification..."):
                    try:
                        result = DeepFace.verify(img1, img2, enforce_detection=True)
                    except Exception:
                        result = DeepFace.verify(img1, img2, enforce_detection=False)
                verified = result.get("verified", False)
                st.write(result)
                if verified:
                    st.success("✅ Face Match Successful")
                else:
                    st.error("❌ Face Mismatch Detected")
                put_session_result("kyc", {"status": "ok", "result": result, "verdict": "verified" if verified else "mismatch"})
            except Exception:
                st.error("Error during face verification.")
                st.write(traceback.format_exc())
                put_session_result("kyc", {"status": "error", "message": "exception"})
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Unusual Pattern Detection --------------------
elif module == "Unusual Pattern Detection":
    st.markdown('<div class="card"><div class="section-title">📊 Unusual Pattern Detection</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload transactions/doc (CSV, XLSX, JPG, PNG, PDF, DOCX)", type=["csv", "xls", "xlsx", "jpg", "jpeg", "png", "pdf", "docx", "doc"], key="anomaly")
    z_cutoff = st.slider("Z-score cutoff for anomaly (|z| > ...)", 2.0, 5.0, 3.0, step=0.1, key="z_cutoff")

    def extract_csv(b):
        try:
            return pd.read_csv(io.BytesIO(b))
        except Exception:
            st.error("Failed to read CSV.")
            return None

    def extract_excel(b):
        try:
            return pd.read_excel(io.BytesIO(b))
        except Exception:
            st.error("Failed to read Excel. Install openpyxl if needed.")
            return None

    def ocr_image_to_dataframe(b):
        if reader is None:
            st.error("easyocr not installed. Install easyocr for OCR.")
            return None
        img = read_image_bytes_to_bgr(b)
        if img is None:
            st.error("Couldn't decode image.")
            return None
        try:
            with st.spinner("Running OCR..."):
                ocr_results = reader.readtext(img)
        except Exception as e:
            st.error("OCR failed.")
            st.write(str(e))
            return None
        lines = [t[1] for t in ocr_results]
        rows = []
        for line in lines:
            line_clean = re.sub(r"\s+", " ", line).strip()
            if "," in line_clean:
                rows.append([p.strip() for p in line_clean.split(",") if p.strip()])
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

    def ocr_pdf_to_dataframe(b):
        if not pdf2image_available:
            st.error("pdf2image or poppler not available. Install for PDF parsing.")
            return None
        try:
            with st.spinner("Converting PDF to images..."):
                pages = convert_from_bytes(b)
                dfs = []
                for page in pages:
                    buf = io.BytesIO()
                    page.save(buf, format="JPEG")
                    dfp = ocr_image_to_dataframe(buf.getvalue())
                    if dfp is not None:
                        dfs.append(dfp)
                if dfs:
                    return pd.concat(dfs, ignore_index=True, sort=False)
                return None
        except Exception as e:
            st.error("PDF processing failed.")
            st.write(str(e))
            return None

    def docx_to_dataframe(b):
        if not docx_available:
            st.error("python-docx not installed.")
            return None
        try:
            d = docx.Document(io.BytesIO(b))
            paragraphs = [p.text for p in d.paragraphs if p.text.strip()]
            rows = []
            for ptext in paragraphs:
                line = re.sub(r"\s+", " ", ptext).strip()
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
            st.error("DOCX parse failed.")
            st.write(str(e))
            return None

    if uploaded:
        file_bytes = uploaded.read()
        name = uploaded.name.lower()
        data = None
        try:
            if name.endswith(".csv"):
                data = extract_csv(file_bytes)
                if data is not None:
                    data = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="ignore"))
            elif name.endswith((".xls", ".xlsx")):
                data = extract_excel(file_bytes)
                if data is not None:
                    data = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="ignore"))
            elif name.endswith((".jpg", ".jpeg", ".png")):
                data = ocr_image_to_dataframe(file_bytes)
            elif name.endswith(".pdf"):
                data = ocr_pdf_to_dataframe(file_bytes)
            elif name.endswith((".docx", ".doc")):
                data = docx_to_dataframe(file_bytes)
            else:
                st.error("Unsupported file type.")
        except Exception:
            st.error("Failed to parse uploaded file.")
            st.write(traceback.format_exc())

        if data is None:
            st.info("Couldn't extract structured data. CSV/XLSX recommended.")
            put_session_result("anomaly", {"status": "error", "message": "no_structured_data"})
        else:
            st.subheader("Parsed preview")
            st.dataframe(data.head(10))
            numeric = data.select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                coerced = data.apply(lambda col: pd.to_numeric(col.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce"))
                numeric = coerced.select_dtypes(include=[np.number])
                if numeric.shape[1] > 0:
                    for c in numeric.columns:
                        data[c] = coerced[c]
            if numeric.shape[1] == 0:
                st.warning("No numeric columns found — anomaly detection requires numeric data.")
                put_session_result("anomaly", {"status": "error", "message": "no_numeric"})
            else:
                with st.spinner("Computing anomalies..."):
                    z = (numeric - numeric.mean()) / numeric.std(ddof=0)
                    mask = (z.abs() > z_cutoff).any(axis=1)
                    anomalies = data[mask]
                st.subheader("Detected anomalies")
                if anomalies.shape[0] == 0:
                    st.write("No anomalies detected with the current cutoff.")
                else:
                    st.dataframe(anomalies)
                    put_session_result("anomaly", {"status": "ok", "count": int(anomalies.shape[0]), "sample": anomalies.head(5).to_dict(orient="records")})
                    st.download_button("Download anomalies (CSV)", data=anomalies.to_csv(index=False).encode(), file_name="anomalies.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Report & Export --------------------
elif module == "Report & Export":
    st.markdown('<div class="card"><div class="section-title">📁 Report & Export</div>', unsafe_allow_html=True)
    st.write("Generate a human-readable PDF report of the latest analysis from different modules.")
    st.write("Latest results (summary):")
    results = get_session_results()
    st.json(results)

    if reportlab_available:
        if st.button("Generate Fraud Report (PDF)"):
            try:
                buffer = io.BytesIO()
                c = canvas.Canvas(buffer, pagesize=A4)
                width, height = A4
                margin = 40
                y = height - margin

                c.setFont("Helvetica-Bold", 16)
                c.drawString(margin, y, "AI Fraud Detection Report")
                c.setFont("Helvetica", 10)
                y -= 20
                c.drawString(margin, y, "Generated by AI Fraud Detection System")
                y -= 30

                # iterate results
                for key, value in results.items():
                    c.setFont("Helvetica-Bold", 12)
                    if y < margin + 120:
                        c.showPage()
                        y = height - margin
                    c.drawString(margin, y, f"{key.replace('_',' ').title()}:")
                    y -= 16
                    c.setFont("Helvetica", 10)

                    if isinstance(value, dict):
                        for k2, v2 in value.items():
                            # handle image bytes specially
                            if k2 in ("orig_image", "sus_image", "diff_image", "sub_image"):
                                if v2:
                                    try:
                                        # v2 should be image file bytes (PNG/JPEG)
                                        img = Image.open(io.BytesIO(v2)).convert("RGB")
                                        # thumbnail within bounds
                                        max_w = width - 2*margin
                                        max_h = 140
                                        img.thumbnail((max_w, max_h))
                                        # page break if not enough space
                                        if y - img.size[1] < margin:
                                            c.showPage()
                                            y = height - margin
                                        ir = ImageReader(img)
                                        c.drawImage(ir, margin, y - img.size[1], width=img.size[0], height=img.size[1])
                                        y -= img.size[1] + 8
                                        continue
                                    except Exception:
                                        # fallback to text entry
                                        c.drawString(margin+8, y, f"{k2}: <image could not be rendered>")
                                        y -= 12
                                        continue
                                else:
                                    c.drawString(margin+8, y, f"{k2}: <no image provided>")
                                    y -= 12
                                    continue
                            # normal value
                            text_line = f"{k2}: {str(v2)}"
                            # wrap long lines
                            max_chars = 90
                            while len(text_line) > max_chars:
                                c.drawString(margin+8, y, text_line[:max_chars])
                                text_line = text_line[max_chars:]
                                y -= 12
                                if y < margin + 80:
                                    c.showPage()
                                    y = height - margin
                            c.drawString(margin+8, y, text_line)
                            y -= 12
                            if y < margin + 80:
                                c.showPage()
                                y = height - margin
                    else:
                        c.drawString(margin+8, y, str(value))
                        y -= 14
                    y -= 10
                    if y < margin + 80:
                        c.showPage()
                        y = height - margin

                c.showPage()
                c.save()
                buffer.seek(0)
                b64 = base64.b64encode(buffer.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="fraud_report.pdf">Download Fraud Report (PDF)</a>'
                st.markdown(href, unsafe_allow_html=True)
            except Exception:
                st.error("Failed to generate PDF report.")
                st.write(traceback.format_exc())
    else:
        st.error("PDF export not available. Install reportlab: pip install reportlab and reload the app to enable PDF report generation.")
        st.info("You can download anomalies CSV from the Unusual Pattern module.")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Footer --------------------
st.markdown("<div style='margin-top:12px;color:#6b7280;font-size:13px;'>Made with care — AI Fraud Detection (white theme, Persian Blue accents)</div>", unsafe_allow_html=True)
