import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pdfplumber
import sqlite3
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.explain import get_explainer, explain_prediction
from src import config

st.set_page_config(page_title="JobFit AI | Shortlist Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling for Premium Aesthetics ---
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #58a6ff;
    }
    
    /* Custom Card for Scores */
    .score-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease-in-out;
    }
    .score-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    
    .score-title {
        font-size: 1.2rem;
        color: #8b949e;
        margin-bottom: 10px;
    }
    .score-value {
        font-size: 3rem;
        font-weight: bold;
        color: #58a6ff;
    }
    .score-value.high { color: #3fb950; }
    .score-value.med { color: #d29922; }
    .score-value.low { color: #f85149; }
</style>
""", unsafe_allow_html=True)

# --- Initialization & Model Loading ---
@st.cache_resource
def load_model():
    model_path = config.MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)

@st.cache_resource
def load_explainer(_pipeline):
    # We need a background dataset for TreeExplainer
    # In a real app we'd load a sample of the training data.
    # For now, we create a small dummy background dataset.
    dummy_background = ["Software Engineer Python SQL", "Data Scientist Machine Learning Python Pandas"]
    return get_explainer(_pipeline, dummy_background)

pipeline = load_model()

# --- Helpers ---
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def log_prediction(resume_text, job_text, proba):
    try:
        db_path = config.DATA_DIR / "jobfit.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_snippet TEXT,
                job_snippet TEXT,
                score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            INSERT INTO predictions (resume_snippet, job_snippet, score)
            VALUES (?, ?, ?)
        """, (resume_text[:100], job_text[:100], proba))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.sidebar.error(f"DB Logging Error: {e}")

# --- App Layout ---
st.title("🎯 JobFit AI : Shortlist Predictor")
st.markdown("*AI-driven objective candidate ranking and skill gap analysis.*")

if pipeline is None:
    st.error("Model not found. Please train the model first by running `python src/models/train.py`.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("📝 Input Data")
st.sidebar.markdown("Provide the Job Description and the Candidate's Resume.")

job_desc = st.sidebar.text_area("Job Description", height=200, placeholder="Paste the job description here...")

input_mode = st.sidebar.radio("Resume Input Method", ["Paste Text", "Upload PDF"])
resume_text = ""
if input_mode == "Paste Text":
    resume_text = st.sidebar.text_area("Candidate Resume", height=200, placeholder="Paste resume text here...")
else:
    pdf_file = st.sidebar.file_uploader("Upload Resume PDF", type=["pdf"])
    if pdf_file is not None:
        resume_text = extract_text_from_pdf(pdf_file)
        st.sidebar.success("PDF parsed successfully!")

# --- Main Logic ---
if st.sidebar.button("Analyze Candidate", type="primary"):
    if not job_desc or not resume_text:
        st.warning("Please provide both a Job Description and a Resume.")
    else:
        with st.spinner("Analyzing candidate profile..."):
            # Prepare unified text input
            X_instance = [resume_text + " | " + job_desc]
            
            # Predict
            proba = pipeline.predict_proba(X_instance)[0][1]
            log_prediction(resume_text, job_desc, proba)
            
            # Explain
            explainer = load_explainer(pipeline)
            shap_values = explain_prediction(explainer, pipeline, X_instance)
            
            # Feature names
            feature_names = pipeline.named_steps['features'].get_feature_names_out()
            
            # --- Results Presentation ---
            st.markdown("---")
            st.subheader("Analysis Results")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Score Card
                score_pct = int(proba * 100)
                color_class = "high" if score_pct >= 70 else "med" if score_pct >= 40 else "low"
                st.markdown(f"""
                    <div class="score-card">
                        <div class="score-title">Match Score</div>
                        <div class="score-value {color_class}">{score_pct}%</div>
                        <p style="color:#8b949e; font-size: 0.9rem;">Probability of Shortlist</p>
                    </div>
                """, unsafe_allow_html=True)
                
            with col2:
                # SHAP Waterfall Chart
                st.markdown("#### Feature Contributions (Why this score?)")
                fig, ax = plt.subplots(figsize=(8, 4))
                # For TreeExplainer, shap_values might be an Explanation object or list
                # Depending on the shap version and model type.
                # Since Random Forest is used, we get an Explanation object in shap 0.49+
                if hasattr(shap_values, "values"):
                    # Slice the explanation for the single instance
                    sv = shap_values[0]
                    # Since it's binary classification, we want the positive class (index 1)
                    if len(sv.values.shape) > 1:
                        sv.values = sv.values[:, 1]
                        sv.base_values = sv.base_values[1]
                    shap.plots.waterfall(sv, show=False)
                else:
                    # Fallback for older shap behavior
                    shap.summary_plot(shap_values[1], X_instance, feature_names=feature_names, show=False)
                st.pyplot(fig)

            st.success("Analysis complete. See history below.")

# --- History Tab ---
st.markdown("---")
with st.expander("🕒 Recent Analyses History"):
    try:
        db_path = config.DATA_DIR / "jobfit.db"
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            df_history = pd.read_sql("SELECT timestamp, score, resume_snippet, job_snippet FROM predictions ORDER BY id DESC LIMIT 10", conn)
            conn.close()
            
            if not df_history.empty:
                df_history['score'] = (df_history['score'] * 100).astype(int).astype(str) + "%"
                st.dataframe(df_history, use_container_width=True)
            else:
                st.info("No predictions logged yet.")
        else:
            st.info("Database not initialized yet.")
    except Exception as e:
        st.error(f"Could not load history: {e}")
