"""
Feature engineering module for JobFit AI.
Builds sklearn-compatible transformers for the resume-to-job matching pipeline.
"""
import numpy as np
from typing import List, Optional, Set
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config
from src.features import skills


# ═══════════════════════════════════════════════════════════════
#  SKILL FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════

class SkillFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts skill-based features from text.
    Returns counts of skills per category.
    """
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: List[str]):
        features = []
        for text in X:
            extracted = skills.extract_skills(text)
            
            n_skills = len(extracted)
            counts = {
                "programming": 0,
                "ml": 0,
                "framework": 0,
                "cloud": 0,
                "soft": 0,
                "database": 0,
                "data_engineering": 0,
            }
            
            for skill in extracted:
                cat = skills.get_skill_category(skill)
                if cat in counts:
                    counts[cat] += 1
                    
            features.append([
                n_skills,
                counts["programming"],
                counts["ml"],
                counts["framework"],
                counts["cloud"],
                counts["soft"],
                counts["database"],
                counts["data_engineering"]
            ])
            
        return np.array(features)
        
    def get_feature_names_out(self, input_features=None):
        return np.array([
            "n_skills",
            "n_programming",
            "n_ml",
            "n_framework",
            "n_cloud",
            "n_soft",
            "n_database",
            "n_data_engineering"
        ])


# ═══════════════════════════════════════════════════════════════
#  SKILL COVERAGE TRANSFORMER
# ═══════════════════════════════════════════════════════════════

class SkillCoverageTransformer(BaseEstimator, TransformerMixin):
    """
    Computes skill coverage between text and a target set of job skills.
    """
    
    def __init__(self, job_skills: Optional[Set[str]] = None):
        self.job_skills = job_skills if job_skills is not None else set()
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: List[str]):
        features = []
        for text in X:
            resume_skills = set(skills.extract_skills(text))
            
            if not self.job_skills:
                coverage_ratio = 0.0
                n_matched = 0
                n_missing = 0
            else:
                matched, missing, coverage_ratio = skills.compute_skill_match(
                    resume_skills, self.job_skills
                )
                n_matched = len(matched)
                n_missing = len(missing)
                
            features.append([
                coverage_ratio,
                n_matched,
                n_missing
            ])
            
        return np.array(features)
        
    def get_feature_names_out(self, input_features=None):
        return np.array([
            "coverage_ratio",
            "n_matched",
            "n_missing"
        ])


# ═══════════════════════════════════════════════════════════════
#  TEXT FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts basic text statistics.
    """
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: List[str]):
        features = []
        for text in X:
            text_length = len(text)
            words = text.split()
            word_count = len(words)
            avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            if sentence_count == 0 and word_count > 0:
                sentence_count = 1
                
            features.append([
                text_length,
                word_count,
                avg_word_length,
                sentence_count
            ])
            
        return np.array(features)
        
    def get_feature_names_out(self, input_features=None):
        return np.array([
            "text_length",
            "word_count",
            "avg_word_length",
            "sentence_count"
        ])


# ═══════════════════════════════════════════════════════════════
#  PIPELINE BUILDER
# ═══════════════════════════════════════════════════════════════

def build_feature_pipeline(include_tfidf: bool = True) -> Pipeline:
    """
    Builds the full feature engineering pipeline.
    """
    transformers = [
        ("skills", SkillFeatureExtractor()),
        ("text_stats", TextFeatureExtractor())
    ]
    
    if include_tfidf:
        tfidf = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF
        )
        transformers.append(("tfidf", tfidf))
        
    union = FeatureUnion(transformer_list=transformers)
    return Pipeline(steps=[("features", union)])


def build_combined_features(resume_texts: List[str], job_texts: Optional[List[str]] = None) -> np.ndarray:
    """
    Convenience function to build combined features.
    If job_texts is provided, coverage features could be added (not standard in pipeline).
    """
    pipeline = build_feature_pipeline(include_tfidf=True)
    return pipeline.fit_transform(resume_texts)


if __name__ == "__main__":
    sample_resumes = [
        "I am a data scientist with experience in Python, pandas, and scikit-learn. I love machine learning.",
        "Experienced frontend developer skilled in JavaScript, React, and TypeScript.",
        "Data engineer who knows SQL, Python, Airflow, and AWS."
    ]
    
    print("Building pipeline...")
    pipeline = build_feature_pipeline(include_tfidf=True)
    
    print("Fitting and transforming sample resumes...")
    features = pipeline.fit_transform(sample_resumes)
    
    print(f"Output shape: {features.shape}")
    
    # Try to extract feature names if supported
    try:
        feature_names = pipeline.named_steps["features"].get_feature_names_out()
        print(f"Feature names ({len(feature_names)}):")
        print(feature_names[:20], "...")
    except Exception as e:
        print("Could not get feature names:", e)
