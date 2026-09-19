# JobFit AI 🎯

JobFit AI is an end-to-end Machine Learning system that bridges the gap between Recruiters and Candidates using predictive scoring and explainable AI (SHAP). 

It features a custom `RandomForest` inference pipeline, two Streamlit applications, and SQLite tracking.

## Features

1. **Shortlist Predictor (For Recruiters)**
   - Batch upload or paste candidate resumes against a Job Description.
   - Instantly see a probability match score (0-100%).
   - Transparent AI: View exactly which skills contributed positively or negatively to the score via SHAP waterfall charts.

2. **Skill-Gap Coach (For Candidates)**
   - Upload your resume and select your dream target role.
   - Discover your blind spots: See the Top 5 missing skills.
   - Using *Ablation Testing*, the model simulates adding each skill to your resume and reveals exactly how much your shortlist probability will jump.

3. **ML Pipeline & Architecture**
   - Engineered purely with `scikit-learn` and `pandas`.
   - TF-IDF extraction combined with a custom Skill Knowledge Graph.
   - Implements `class_weight='balanced'` for fairness.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(If you face corporate proxy SSL issues, see `scripts/PROXY_WORKAROUND.md`)*

### 2. Train the Model
The model uses a synthetic dataset by default to ensure immediate zero-friction onboarding.
```bash
python src/models/train.py
```
*(This generates `models/best_model.pkl`)*

### 3. Run the Apps
**App 1: Shortlist Predictor (Recruiter View)**
```bash
streamlit run src/app1_shortlist/app.py
```

**App 2: Skill-Gap Coach (Candidate View)**
```bash
streamlit run src/app2_coach/app.py
```

## Portfolio
A standalone HTML portfolio is available in `portfolio_site/index.html` showcasing the architecture and case studies.
