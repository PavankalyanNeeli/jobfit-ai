"""
Central configuration — all paths, hyperparameters, and constants in ONE place.

Every module imports from here. Change once, propagate everywhere.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "jobfit.db"
MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

# ── Synthetic Data ────────────────────────────────────────────
RANDOM_SEED = 42
N_SYNTHETIC_RESUMES = 500
N_SYNTHETIC_JOBS = 100

# ── Target Roles (single source of truth) ────────────────────
TARGET_ROLES = [
    "data scientist",
    "data analyst",
    "ml engineer",
    "data engineer",
    "backend developer",
    "frontend developer",
    "full stack developer",
    "devops engineer",
    "business analyst",
    "research scientist",
]

# ── Feature Engineering ───────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.85
TFIDF_SUBLINEAR_TF = True

# ── Model Training ────────────────────────────────────────────
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_FILENAME = "best_pipeline.pkl"

# Class imbalance: always use balanced weights.
# Interview line: "Shortlist prediction is inherently imbalanced —
# I handle it at the model level with class_weight='balanced',
# and at evaluation with PR-AUC and calibration curves."
CLASS_WEIGHT = "balanced"

# ── Probability Calibration ──────────────────────────────────
# Wrap best model in CalibratedClassifierCV so the 75% gauge
# actually means 75%. Calibration curve proves it.
CALIBRATION_METHOD = "sigmoid"   # 'sigmoid' (Platt) or 'isotonic'
CALIBRATION_CV = 5

# ── SHAP ──────────────────────────────────────────────────────
SHAP_MAX_DISPLAY = 10

# ── App ───────────────────────────────────────────────────────
HISTORY_LIMIT = 10  # last N predictions shown in history tab
SCORE_PRECISION = 2  # decimal places for displayed scores
