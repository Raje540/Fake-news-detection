import streamlit as st
import pickle
from PyPDF2 import PdfReader
import pandas as pd

# ------------------ Load Pipeline ------------------
pipeline = pickle.load(open("../models/pipeline.pkl", "rb"))

# ------------------ Session State ------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ------------------ Page Config ------------------
st.set_page_config(page_title="📰 Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection System")
st.write("Paste a news article or upload a PDF/TXT file to check if it is REAL or FAKE.")

# ------------------ User Input ------------------
text_input = st.text_area("Or paste news text here:", height=200)
uploaded_file = st.file_uploader("Upload a PDF or TXT file:", type=["pdf", "txt"])

news_text = ""

# Extract text from uploaded file
if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/pdf":
            pdf = PdfReader(uploaded_file)
            for page in pdf.pages:
                news_text += page.extract_text() + "\n"
        elif uploaded_file.type == "text/plain":
            news_text = str(uploaded_file.read(), "utf-8")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# If text area is used, override uploaded file
if text_input.strip() != "":
    news_text = text_input

# ------------------ Prediction ------------------
if st.button("Analyze"):
    if news_text.strip() == "":
        st.warning("⚠️ Please provide some text or upload a file!")
    else:
        try:
            prediction = pipeline.predict([news_text])[0]
            prob = pipeline.predict_proba([news_text])[0]
            confidence = round(max(prob) * 100, 2)

            # Save to session history
            st.session_state.history.append({
                "text": news_text,
                "prediction": prediction,
                "confidence": confidence
            })

            # Display prediction
            if prediction == "REAL":
                st.success(f"REAL NEWS ✅ (Confidence: {confidence}%)")
            else:
                st.error(f"FAKE NEWS ❌ (Confidence: {confidence}%)")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ------------------ Prediction History ------------------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)

    # Table of Contents (TOC)
    st.markdown("### 🔗 Table of Contents")
    for idx, row in history_df.iterrows():
        st.markdown(f"- [{row['prediction']} - Confidence {row['confidence']}%](#{idx})")
    st.markdown("---")

    # Display table of history
    st.dataframe(history_df[["text", "prediction", "confidence"]], height=300)

    # ------------------ Delete Rows ------------------
    st.markdown("### 🗑 Delete a row")
    rows_to_delete = []
    for idx, row in history_df.iterrows():
        if st.button(f"Delete row {idx} - {row['prediction']}", key=f"del_{idx}"):
            rows_to_delete.append(idx)

    # Remove deleted rows after iterating
    if rows_to_delete:
        for idx in sorted(rows_to_delete, reverse=True):
            st.session_state.history.pop(idx)
        st.info("✅ Row(s) deleted successfully! The table and chart have been updated.")

    # ------------------ Chart ------------------
    st.subheader("📊 Prediction Distribution")
    chart_data = pd.DataFrame(st.session_state.history)
    if not chart_data.empty:
        st.bar_chart(chart_data['prediction'].value_counts())

    # ------------------ Download CSV ------------------
    if chart_data.shape[0] > 0:
        csv = chart_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
