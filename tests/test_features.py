"""
Tests for the feature engineering module.
"""
import pytest
import numpy as np
from typing import List

from src.features.build import (
    SkillFeatureExtractor,
    SkillCoverageTransformer,
    TextFeatureExtractor,
    build_feature_pipeline
)
from src.features.skills import extract_skills

# ═══════════════════════════════════════════════════════════════
#  SKILL FEATURE EXTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════

def test_skill_feature_extractor(sample_resumes: List[str]) -> None:
    """Ensure SkillFeatureExtractor returns an array of the correct shape and extracts positive counts."""
    extractor = SkillFeatureExtractor()
    features = extractor.fit_transform(sample_resumes)
    
    # 10 resumes, 8 features (n_skills + 7 categories)
    assert features.shape == (10, 8)
    
    # All values should be non-negative
    assert np.all(features >= 0)
    
    # The first resume ("Data Scientist... Python, pandas...") should have > 0 skills
    assert features[0, 0] > 0
    # Should have programming skills (Python)
    assert features[0, 1] > 0

# ═══════════════════════════════════════════════════════════════
#  SKILL COVERAGE TRANSFORMER TESTS
# ═══════════════════════════════════════════════════════════════

def test_skill_coverage_transformer(sample_resumes: List[str], sample_job: str) -> None:
    """Ensure coverage ratio is bounded between 0.0 and 1.0."""
    job_skills = set(extract_skills(sample_job))
    transformer = SkillCoverageTransformer(job_skills=job_skills)
    
    features = transformer.fit_transform(sample_resumes)
    
    # 10 resumes, 3 features (coverage_ratio, n_matched, n_missing)
    assert features.shape == (10, 3)
    
    # Coverage ratio is at index 0
    coverage_ratios = features[:, 0]
    
    assert np.all(coverage_ratios >= 0.0)
    assert np.all(coverage_ratios <= 1.0)
    
    # First resume has Python, second has none of the job skills
    assert coverage_ratios[0] > 0.0
    assert coverage_ratios[1] == 0.0

# ═══════════════════════════════════════════════════════════════
#  TEXT FEATURE EXTRACTOR TESTS
# ═══════════════════════════════════════════════════════════════

def test_text_feature_extractor(sample_resumes: List[str]) -> None:
    """Ensure word counts and lengths are positive."""
    extractor = TextFeatureExtractor()
    features = extractor.fit_transform(sample_resumes)
    
    # 10 resumes, 4 features (text_length, word_count, avg_word_length, sentence_count)
    assert features.shape == (10, 4)
    
    # All features should be strictly positive for our non-empty sample resumes
    assert np.all(features > 0)
    
    # text_length
    assert features[0, 0] == len(sample_resumes[0])
    # word_count
    assert features[0, 1] == len(sample_resumes[0].split())

# ═══════════════════════════════════════════════════════════════
#  PIPELINE BUILDER TESTS
# ═══════════════════════════════════════════════════════════════

def test_build_feature_pipeline(sample_resumes: List[str]) -> None:
    """Ensure the full pipeline fits and transforms sample_resumes."""
    pipeline = build_feature_pipeline(include_tfidf=True)
    
    # Should fit and transform without errors
    # Need to convert output to dense format because TF-IDF uses sparse matrices
    features = pipeline.fit_transform(sample_resumes)
    
    # Number of rows should match number of resumes
    assert features.shape[0] == len(sample_resumes)
    
    # Should have a significant number of columns due to TF-IDF
    assert features.shape[1] > 12

if __name__ == "__main__":
    pytest.main(["-v", __file__])
