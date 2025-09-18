
import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import base64
import os
from transformers import pipeline

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="AI-Powered Data Storyteller", layout="wide")

# -------------------------------
# Load Hugging Face Summarizer (LLM)
# -------------------------------
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# -------------------------------
# PDF Export Function
# -------------------------------
def export_pdf_report(df, numeric_cols, cat_cols):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Executive Summary Report", ln=1, align="C")
    pdf.ln(10)

    pdf.multi_cell(0, 10, f"Number of rows: {len(df)}")
    pdf.multi_cell(0, 10, f"Number of columns: {len(df.columns)}")

    if numeric_cols:
        pdf.multi_cell(
            0, 10,
            "Top numeric means: " + ", ".join(f"{col}={df[col].mean():.2f}" for col in numeric_cols[:3])
        )

    if cat_cols:
        common_vals = df[cat_cols].mode().astype(str).to_dict(orient='records')[0]
        formatted_common = ", ".join([f"{k}={v}" for k, v in common_vals.items()])
        pdf.multi_cell(0, 10, f"Most common categorical values: {formatted_common}")

    # ✅ Get PDF as bytes
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return pdf_bytes

# -------------------------------
# Background Logo
# -------------------------------
def add_background_logo(logo_path):
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: top right;
                background-size: 150px 150px;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# ✅ Call logo
add_background_logo("logo.png")  # ensure logo.png is in project folder

# ✅ Title
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📊 AI-Powered Data Storyteller</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# -------------------------------
# Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # --- EDA ---
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all').T)

    st.subheader("Value Counts (Top Categorical Columns)")
    cat_cols = [col for col in df.columns if df[col].dtype == "object" and df[col].nunique() < (0.4 * len(df))]
    for col in cat_cols[:3]:  # Limit: avoid performance issues
        st.write(f"**{col}**")
        st.bar_chart(df[col].value_counts())

    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=np.number)
    if len(numeric_cols.columns) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --- Visualization Plots ---
    st.subheader("Data Visualizations")
    if cat_cols:
        plt.figure()
        df[cat_cols].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {cat_cols}")
        st.pyplot(plt)

    num_cols = list(numeric_cols.columns)
    if len(num_cols) > 1:
        plt.figure()
        df[num_cols].plot(kind='line')
        plt.title(f"Line Plot of {num_cols}")
        st.pyplot(plt)

    # --- Static Insights ---
    st.subheader("Auto-Generated Insights")
    st.write(f"Number of rows: {len(df)}")
    st.write(f"Number of columns: {len(df.columns)}")
    st.write("Top numeric column means:")
    for col in num_cols[:3]:
        st.write(f"- {col}: mean = {df[col].mean():.2f}")
    if cat_cols:
        st.write(f"Most common {cat_cols}: {df[cat_cols].mode()}")

    # --- AI-Generated Insights with LLM ---
    st.subheader("AI-Generated Narrative Insights")
    if st.button("Generate AI Insights"):
        text_summary = f"""
        Dataset overview:
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        - Numeric columns: {list(numeric_cols.columns)}
        - Categorical columns: {list(cat_cols)}
        - Quick stats: {df.describe(include='all').T.to_dict()}
        """
        result = summarizer(text_summary, max_length=120, min_length=40, do_sample=False)
        st.success(result[0]['summary_text'])

    # --- PDF Export ---
    if st.button("Download Executive Summary (PDF)"):
        pdf_bytes = export_pdf_report(df, num_cols, cat_cols)
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="executive_summary.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info("Please upload a CSV file to begin analysis.")

