# Model Card: JobFit AI Scoring Engine

## Model Details
- **Architecture**: `RandomForestClassifier` with TF-IDF vectorization and custom skill-extraction transformers.
- **Task**: Binary classification (Shortlist / Reject)
- **Framework**: `scikit-learn`, `joblib`
- **Version**: 1.0 (Phase B)

## Intended Use
- **Primary Use Case**: Ranking resumes against job descriptions based on skill coverage, structural heuristics, and domain relevance.
- **Target Users**: Recruiters (for batch processing and ranking) and Candidates (for identifying skill gaps).
- **Out of Scope**: Automated hiring decisions without human oversight. The model is an assistive scoring tool, not a decision-maker.

## Training Data
- **Source**: Synthetic generated resumes and job descriptions (to be swapped with real-world Kaggle HR data in Phase C).
- **Features Extracted**:
  - `n_skills`, `n_programming`, `n_ml`, `n_framework`, etc. (via Knowledge Graph)
  - `coverage_ratio`, `n_matched`, `n_missing`
  - `text_length`, `word_count`, `avg_word_length`, `sentence_count`
  - 5000 max TF-IDF unigrams/bigrams

## Performance Metrics
*(To be updated after MLflow tracking captures real-world holdout data)*
- **Optimization Target**: F1-Score (balanced precision and recall)
- **Validation**: 20% stratified holdout set.

## Ethical Considerations & Bias
- **Class Balance**: The model utilizes `class_weight='balanced'` to prevent bias toward rejecting all candidates in highly imbalanced talent pools.
- **Explainability**: SHAP (SHapley Additive exPlanations) is integrated to ensure transparency for every prediction.
- **Fairness Warning**: The model operates purely on text and technical skill matching. It does not extract protected attributes (gender, age, race), but TF-IDF may inadvertently latch onto linguistic proxies. Periodic fairness audits are required.
