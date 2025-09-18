import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import torch
import io
import json
from PyPDF2 import PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Load Hugging Face pipeline for sentiment analysis (pos/neg/neu with scores)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_classifier = load_sentiment_model()
# Custom CSS for color palette and styling
st.markdown("""
    <style>
    body, .main {
        background-color: #A8E8F9;
    }
    .stApp {
        background-color: #A8E8F9;
    }
    .stTitle, .stMarkdown h1, .stMarkdown h2 {
        color: #00537A;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #F5A201;
        color: #013C58;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        padding: 0.5em 2em;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFBA42;
        color: #00537A;
    }
    .stRadio label, .stTextArea label, .stFileUploader label {
        color: #013C58;
        font-weight: bold;
    }
    .stDataFrame, .stTable {
        background-color: #FFD35B;
        border-radius: 8px;
    }
    .stSidebar {
        background-color: #00537A;
    }
    .stSidebar .stMarkdown {
        color: #FFD35B;
    }
    .stDownloadButton>button {
        background-color: #00537A;
        color: #FFD35B;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    .stDownloadButton>button:hover {
        background-color: #013C58;
        color: #F5A201;
    }
    </style>
""", unsafe_allow_html=True)
# Function for sentiment analysis and explanations
def analyze_sentiment(texts):
    results = []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10)  # For keyword extraction
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        keywords = vectorizer.get_feature_names_out()
    except:
        keywords = []
    
    for text in texts:
        if not str(text).strip():
            continue
        try:
            result = sentiment_classifier(text)[0]
            label = result['label'].upper()  # POSITIVE, NEGATIVE, NEUTRAL
            score = result['score']
            # Simple explanation: Highlight top TF-IDF words as "sentiment drivers"
            explanation = f"Key drivers: {', '.join(keywords)} (based on word importance in batch)."
            results.append({"Text": text, "Sentiment": label, "Confidence": score, "Explanation": explanation})
        except Exception as e:
            st.error(f"Error analyzing '{str(text)[:50]}...': {str(e)}")
    return pd.DataFrame(results)
def section_divider():
    st.markdown("""
        <hr style="border:1px solid #F5A201;margin:1.5em 0;">
    """, unsafe_allow_html=True)
# Streamlit App
st.markdown("""
    <div style="background-color:#00537A;padding:1.2em 0;border-radius:12px;margin-bottom:1.5em;box-shadow:0 2px 8px #013C5822;">
        <h1 style="color:#FFD35B;text-align:center;font-family:'Segoe UI',sans-serif;margin:0;">
            📝 Sentiment Analysis Dashboard
        </h1>
    </div>
""", unsafe_allow_html=True)
st.title("Sentiment Analysis Dashboard")
st.markdown("Analyze emotional tone in customer reviews, social media, or text data. Supports batch processing and comparisons.")

# Input Section
section_divider()
input_method = st.radio("Input Method", ("Direct Entry", "File Upload"))
texts = []

if input_method == "Direct Entry":
    text_input = st.text_area("Enter text (one per line for batch):")
    if text_input:
        texts = text_input.split("\n")
elif input_method == "File Upload":
    uploaded_file = st.file_uploader("Upload CSV or TXT (one text per row/line)", type=["csv", "txt"])
    if uploaded_file:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            texts = df.iloc[:, 0].tolist()  # Assume first column is text
        else:
            texts = uploaded_file.read().decode("utf-8").split("\n")

# Analysis Button
section_divider()
if st.button("Analyze Sentiment") and texts:
    with st.spinner("Processing..."):
        df_results = analyze_sentiment(texts)
        
        if not df_results.empty:
            # Display Results Table
            st.subheader("Analysis Results")
            st.dataframe(df_results)
            
            # Visualizations: Sentiment Distribution
            st.subheader("Sentiment Distribution")
            st.markdown("This pie chart shows the proportion of each sentiment (Positive, Negative, Neutral) detected in your input data.")
            fig, ax = plt.subplots()
            colors = ['#00537A', '#F5A201', '#FFD35B']
            df_results['Sentiment'].value_counts().plot.pie(
            autopct='%1.1f%%', ax=ax, colors=colors, textprops={'color':'#013C58', 'fontsize':12}
)
            ax.set_ylabel('')
            ax.set_title("Sentiment Distribution", color="#00537A", fontsize=16)
            st.pyplot(fig)
            section_divider()
            # Comparative Analysis: Avg confidence by sentiment
            st.subheader("Comparative Analysis")
            st.markdown("The bar chart below compares the average confidence scores for each sentiment category. Higher scores indicate stronger model certainty.")
            avg_conf = df_results.groupby('Sentiment')['Confidence'].mean()
            st.bar_chart(avg_conf, color="#F5A201")
            st.write("Average confidence scores across sentiments for comparison.")
            
            # Export Options
            st.subheader("Export Results")
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "results.csv", "text/csv")
            
            json_data = df_results.to_json(orient="records")
            st.download_button("Download JSON", json_data, "results.json", "application/json")
            
            # PDF Export (simple text-based)
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            y = 750
            for _, row in df_results.iterrows():
                c.drawString(100, y, f"Text: {row['Text'][:100]}...")
                c.drawString(100, y-20, f"Sentiment: {row['Sentiment']} (Confidence: {row['Confidence']:.2f})")
                c.drawString(100, y-40, row['Explanation'])
                y -= 60
                if y < 100:
                    c.showPage()
                    y = 750
            c.save()
            pdf_buffer.seek(0)
            st.download_button("Download PDF", pdf_buffer, "results.pdf", "application/pdf")

# Error Handling for Empty Input
else:
    st.info("Enter or upload text to analyze.")

# Model Limitations Section
section_divider()
st.sidebar.title("About")
st.sidebar.markdown("""
- **Confidence Threshold**: Scores < 0.6 may indicate uncertainty.
- **Limitations**: Handles English best; may misclassify sarcasm or slang.
- Built with Hugging Face and Streamlit.
""")