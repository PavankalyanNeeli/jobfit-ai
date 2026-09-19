import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
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

st.set_page_config(page_title="JobFit AI | Skill-Gap Coach", layout="wide", initial_sidebar_state="expanded")

# --- Custom Styling for Premium Aesthetics ---
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 { color: #58a6ff; }
    
    .coach-card {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease-in-out;
    }
    .coach-card:hover {
        transform: translateY(-5px);
        border-color: #58a6ff;
    }
    
    .skill-name {
        font-size: 1.5rem;
        font-weight: bold;
        color: #c9d1d9;
    }
    
    .boost-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3fb950;
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
st.title("🚀 JobFit AI : Skill-Gap Coach")
st.markdown("*Identify your missing skills and see exactly how learning them boosts your resume.*")

if pipeline is None:
    st.error("Model not found. Please train the model first by running `python src/models/train.py`.")
    st.stop()

# --- Sidebar ---
st.sidebar.header("📝 Candidate Input")

input_mode = st.sidebar.radio("Resume Input Method", ["Paste Text", "Upload PDF"])
resume_text = ""
if input_mode == "Paste Text":
    resume_text = st.sidebar.text_area("Your Resume", height=200, placeholder="Paste your resume here...")
else:
    pdf_file = st.sidebar.file_uploader("Upload Resume PDF", type=["pdf"])
    if pdf_file is not None:
        resume_text = extract_text_from_pdf(pdf_file)
        st.sidebar.success("PDF parsed successfully!")

# Let candidate select a target role or paste a JD
st.sidebar.header("🎯 Target Role")
role_mode = st.sidebar.radio("Target Selection", ["Select Common Role", "Paste Specific JD"])

job_desc = ""
if role_mode == "Select Common Role":
    target_role = st.sidebar.selectbox("Choose a role:", [
        "Data Scientist", 
        "Machine Learning Engineer", 
        "Data Engineer", 
        "Frontend Developer", 
        "Backend Engineer"
    ])
    # Very basic dummy JDs for the sake of the coach
    dummy_jds = {
        "Data Scientist": "Looking for a Data Scientist with Python, Pandas, scikit-learn, SQL, and PyTorch or TensorFlow.",
        "Machine Learning Engineer": "Requires Python, PyTorch, AWS, Docker, Kubernetes, and C++.",
        "Data Engineer": "Need a Data Engineer with SQL, Python, Spark, Airflow, and AWS.",
        "Frontend Developer": "Frontend dev with React, TypeScript, JavaScript, CSS, HTML.",
        "Backend Engineer": "Backend engineer skilled in Java, Spring Boot, SQL, PostgreSQL, Docker."
    }
    job_desc = dummy_jds.get(target_role, "")
else:
    job_desc = st.sidebar.text_area("Job Description", height=200, placeholder="Paste target job description...")

# --- Main Logic ---
if st.sidebar.button("Generate Coaching Plan", type="primary"):
    if not job_desc or not resume_text:
        st.warning("Please provide both your Resume and a Target Role.")
    else:
        with st.spinner("Analyzing gaps and simulating score boosts..."):
            
            # Extract skills for visualization
            resume_skills = set(extract_skills(resume_text))
            
            # Analyze gaps
            gaps = analyze_gaps(resume_text, job_desc, pipeline)
            
            # Rank top 5 recommendations
            top_recs = rank_recommendations(gaps, top_k=5)
            
            st.markdown("---")
            
            if not top_recs:
                st.success("🎉 You have all the key skills for this role! No major technical gaps detected.")
            else:
                baseline = top_recs[0]['baseline_score'] * 100
                st.subheader(f"Current Match Score: {baseline:.1f}%")
                st.markdown("Here are the top skills you should learn to maximize your chances:")
                
                # Display Top Recommendations in Cards
                for i, rec in enumerate(top_recs):
                    skill = rec['skill']
                    delta = rec['delta'] * 100
                    sim = rec['simulated_score'] * 100
                    
                    st.markdown(f"""
                        <div class="coach-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="color: #8b949e; font-size: 1.2rem;">#{i+1} Priority</span><br/>
                                    <span class="skill-name">{skill.upper()}</span>
                                </div>
                                <div style="text-align: right;">
                                    <span style="color: #8b949e; font-size: 0.9rem;">Estimated Boost</span><br/>
                                    <span class="boost-value">+{delta:.1f}%</span>
                                </div>
                            </div>
                            <div style="margin-top: 15px; font-size: 0.9rem; color: #c9d1d9;">
                                <i>Adding {skill} to your resume increases your match probability to <b>{sim:.1f}%</b>.</i>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Plot Bar Chart of Deltas
                st.markdown("### Score Boost Impact")
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor('#0d1117')
                ax.set_facecolor('#0d1117')
                
                skills = [r['skill'] for r in top_recs]
                deltas = [r['delta'] * 100 for r in top_recs]
                
                bars = ax.barh(skills, deltas, color='#58a6ff')
                ax.set_xlabel('Score Boost (%)', color='#c9d1d9')
                ax.tick_params(axis='x', colors='#c9d1d9')
                ax.tick_params(axis='y', colors='#c9d1d9')
                ax.invert_yaxis()  # Highest boost at the top
                
                # Add values on bars
                for bar in bars:
                    width = bar.get_width()
                    ax.annotate(f'+{width:.1f}%',
                                xy=(width, bar.get_y() + bar.get_height() / 2),
                                xytext=(3, 0),  # 3 points horizontal offset
                                textcoords="offset points",
                                ha='left', va='center', color='#3fb950', fontweight='bold')
                
                st.pyplot(fig)
