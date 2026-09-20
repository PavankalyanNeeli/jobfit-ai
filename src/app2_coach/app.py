import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go
import pdfplumber
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.recommender.gap_analyzer import analyze_gaps
from src.recommender.ranker import rank_recommendations
from src.features.skills import extract_skills
from src import config

st.set_page_config(page_title="JobFit AI | Elite Coach", layout="wide", initial_sidebar_state="expanded")

# --- Elite Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    .elite-header {
        text-align: center;
        background: linear-gradient(90deg, #bb86fc, #58a6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .elite-subheader {
        text-align: center;
        color: #8b949e;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 40px;
    }

    .level-up-card {
        background: linear-gradient(145deg, rgba(35, 41, 54, 0.9), rgba(13, 17, 23, 0.9));
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .level-up-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(88, 166, 255, 0.15);
        border: 1px solid rgba(88, 166, 255, 0.5);
    }
    
    .level-up-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 4px; height: 100%;
        background: linear-gradient(180deg, #3fb950, #58a6ff);
    }

    .skill-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .boost-badge {
        background: rgba(63, 185, 80, 0.15);
        color: #3fb950;
        padding: 6px 15px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 800;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    
    .sim-score {
        font-size: 1rem;
        color: #8b949e;
        margin-top: 10px;
    }
    
    .metric-box {
        text-align: center;
        padding: 20px;
        background: rgba(22, 27, 34, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization & Model Loading ---
@st.cache_resource
def load_model():
    model_path = config.MODEL_DIR / "best_model.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)

pipeline = load_model()

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# --- App Layout ---
st.markdown("<div class='elite-header'>JobFit AI Coach</div>", unsafe_allow_html=True)
st.markdown("<div class='elite-subheader'>Your personalized blueprint to land the dream job.</div>", unsafe_allow_html=True)

if pipeline is None:
    st.error("Model not found. Please train the model first by running `python src/models/train.py`.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("📄 1. Your Profile")
    input_mode = st.radio("Resume Format", ["Paste Text", "Upload PDF"])
    resume_text = ""
    if input_mode == "Paste Text":
        resume_text = st.text_area("Paste your resume here...", height=200)
    else:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf_file is not None:
            resume_text = extract_text_from_pdf(pdf_file)
            st.success("Resume Scanned! ✅")

    st.markdown("---")
    st.header("🎯 2. Target Role")
    role_mode = st.radio("Job Description", ["Select Template", "Paste JD"])
    job_desc = ""
    if role_mode == "Select Template":
        target_role = st.selectbox("Choose a role:", [
            "Data Scientist", "Machine Learning Engineer", "Data Engineer", "Frontend Developer", "Backend Engineer"
        ])
        dummy_jds = {
            "Data Scientist": "Looking for a Data Scientist with Python, Pandas, scikit-learn, SQL, and PyTorch or TensorFlow.",
            "Machine Learning Engineer": "Requires Python, PyTorch, AWS, Docker, Kubernetes, and C++.",
            "Data Engineer": "Need a Data Engineer with SQL, Python, Spark, Airflow, and AWS.",
            "Frontend Developer": "Frontend dev with React, TypeScript, JavaScript, CSS, HTML.",
            "Backend Engineer": "Backend engineer skilled in Java, Spring Boot, SQL, PostgreSQL, Docker."
        }
        job_desc = dummy_jds.get(target_role, "")
    else:
        job_desc = st.text_area("Paste the exact Job Description...", height=200)

    analyze_btn = st.button("🚀 Generate Elite Roadmap", type="primary", use_container_width=True)


# --- Main Logic ---
if analyze_btn:
    if not job_desc or not resume_text:
        st.warning("Please provide both your Resume and a Target Role.")
    else:
        with st.spinner("🧠 AI is simulating thousands of resume variations..."):
            
            # Analyze gaps
            gaps = analyze_gaps(resume_text, job_desc, pipeline)
            top_recs = rank_recommendations(gaps, top_k=5)
            
            if not top_recs:
                st.balloons()
                st.success("🏆 You are a PERFECT MATCH! You have all the required technical skills. Apply now!")
                st.stop()
                
            baseline = top_recs[0]['baseline_score'] * 100
            
            # --- Dashboard Top Metrics ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Current Hire Probability</div>
                    <div class="metric-value" style="color: {'#ff7b72' if baseline < 40 else '#d2a8ff' if baseline < 70 else '#3fb950'}">{baseline:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                total_boost = sum([r['delta'] * 100 for r in top_recs])
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Potential Gain</div>
                    <div class="metric-value" style="color: #3fb950">+{total_boost:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                max_score = baseline + total_boost
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Optimized Score</div>
                    <div class="metric-value" style="color: #58a6ff">{max_score:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # --- Skill Path ---
            st.markdown("### 🗺️ Your Skill Acquisition Path")
            st.markdown("We simulated adding different skills to your resume. Here are the exact skills you should learn, ranked by how much they increase your chances of getting hired.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display Elite Cards
            for i, rec in enumerate(top_recs):
                skill = rec['skill'].upper()
                delta = rec['delta'] * 100
                sim = rec['simulated_score'] * 100
                
                # Assign icons based on rank
                icons = ["🥇", "🥈", "🥉", "🏅", "🏅"]
                icon = icons[i] if i < len(icons) else "💡"
                
                st.markdown(f"""
                <div class="level-up-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="color: #8b949e; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Step {i+1}</div>
                            <div class="skill-title">{icon} {skill}</div>
                            <div class="sim-score">If you learn this, your total match score becomes <b>{sim:.1f}%</b>.</div>
                        </div>
                        <div style="text-align: right;">
                            <span class="boost-badge">+{delta:.1f}% Boost</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # --- Plotly Gauge Chart ---
            st.markdown("<br>### 📊 Score Projection", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = max_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Hire Probability (After Learning)", 'font': {'size': 24, 'color': '#c9d1d9'}},
                delta = {'reference': baseline, 'increasing': {'color': "#3fb950"}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#c9d1d9"},
                    'bar': {'color': "#58a6ff"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#30363d",
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(255, 123, 114, 0.2)"},
                        {'range': [40, 75], 'color': "rgba(210, 168, 255, 0.2)"},
                        {'range': [75, 100], 'color': "rgba(63, 185, 80, 0.2)"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': baseline}
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#c9d1d9"})
            st.plotly_chart(fig, use_container_width=True)
