# Updated Unusual Pattern Detection module — accepts images, PDFs, DOCX, Excel, CSV
# Drop this section into your Streamlit app (replace the previous Unusual Pattern Detection block).

import streamlit as st
import pandas as pd
import numpy as np
import io
import cv2
import easyocr
import re
from PIL import Image

reader = easyocr.Reader(['en'], gpu=False)

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
    # Read image bytes to numpy array
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image for OCR")

    # Run EasyOCR to extract text lines
    ocr_results = reader.readtext(img)
    # Collect plain text lines
    lines = [t[1] for t in ocr_results]
    text = "\n".join(lines)

    # Try to heuristically parse tabular data: look for lines with commas or multiple numeric tokens
    rows = []
    for line in lines:
        # replace multiple spaces with single space
        line_clean = re.sub(r"\s+", " ", line).strip()
        # if line contains commas, split by comma
        if "," in line_clean:
            parts = [p.strip() for p in line_clean.split(",") if p.strip()]
            rows.append(parts)
        else:
            # split by whitespace and keep numeric-like tokens
            parts = [p for p in re.split(r"[\s\|;]+", line_clean) if p]
            # if there are multiple tokens, keep the row
            if len(parts) > 1:
                rows.append(parts)

    # If we parsed rows with consistent column counts, convert to DataFrame
    if rows:
        # find most common column count
        counts = [len(r) for r in rows]
        target = max(set(counts), key=counts.count)
        filtered = [r for r in rows if len(r) == target]
        df = pd.DataFrame(filtered)
        # try to convert columns to numeric when possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
        return df

    # fallback: return single-column dataframe of all OCR lines
    return pd.DataFrame({"extracted_text": lines})


def ocr_pdf_to_dataframe(pdf_bytes):
    # Try to convert PDF -> images using pdf2image if available
    try:
        from pdf2image import convert_from_bytes
    except Exception as e:
        st.warning("pdf2image not available or poppler not installed. Install pdf2image & poppler for better PDF support.")
        # fallback: return None so caller can show raw error
        return None

    try:
        pages = convert_from_bytes(pdf_bytes)
        combined = []
        for page in pages:
            img_bytes = io.BytesIO()
            page.save(img_bytes, format='JPEG')
            img_b = img_bytes.getvalue()
            df = ocr_image_to_dataframe(img_b)
            combined.append(df)
        # try concat (will create NaNs where columns differ)
        return pd.concat(combined, ignore_index=True)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None


def docx_to_dataframe(docx_bytes):
    try:
        import docx
    except Exception:
        st.warning("python-docx not installed. Install python-docx to support DOCX text extraction.")
        return None

    try:
        doc = docx.Document(io.BytesIO(docx_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # try to parse paragraphs into rows similar to OCR
        rows = []
        for p in paragraphs:
            line = re.sub(r"\s+", " ", p).strip()
            if "," in line:
                rows.append([c.strip() for c in line.split(",")])
            else:
                parts = [p for p in re.split(r"[\s\|;]+", line) if p]
                if len(parts) > 1:
                    rows.append(parts)
        if rows:
            counts = [len(r) for r in rows]
            target = max(set(counts), key=counts.count)
            filtered = [r for r in rows if len(r) == target]
            df = pd.DataFrame(filtered)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col].str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
            return df
        return pd.DataFrame({"extracted_text": paragraphs})
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None


if uploaded_file:
    fname = uploaded_file.name.lower()
    try:
        if fname.endswith('.csv'):
            data = extract_dataframe_from_csv_bytes(uploaded_file.read())
        elif fname.endswith(('.xls', '.xlsx')):
            data = extract_dataframe_from_excel_bytes(uploaded_file.read())
        elif fname.endswith(('.jpg', '.jpeg', '.png')):
            data = ocr_image_to_dataframe(uploaded_file.read())
        elif fname.endswith('.pdf'):
            data = ocr_pdf_to_dataframe(uploaded_file.read())
            if data is None:
                st.error('Could not extract tables from PDF. Install pdf2image and Poppler for better support.')
        elif fname.endswith(('.docx', '.doc')):
            data = docx_to_dataframe(uploaded_file.read())
            if data is None:
                st.error('Could not extract data from DOC/DOCX. Install python-docx to enable this feature.')
        else:
            st.error("Unsupported file type")
            data = None
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        data = None

    if data is not None:
        st.subheader("Preview of parsed data")
        st.dataframe(data.head())

        # Only use numeric columns for z-score anomaly detection
        numeric = data.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            st.warning("No numeric columns found for anomaly detection. The uploaded file was parsed as text — check 'extracted_text' column or try CSV/XLSX for best results.")
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

# End of module
