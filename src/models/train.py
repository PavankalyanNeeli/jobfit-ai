"""
Model training module for JobFit AI.
Loads synthetic data, builds features, trains a RandomForestClassifier,
logs to MLflow (if available), and saves the best model.
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src import config
from src.data.load import load_data
from src.features.build import build_feature_pipeline
from src.models.evaluate import evaluate_model, plot_evaluation

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def train():
    """Train the model and save it."""
    print("Loading data...")
    resumes_df, jobs_df, matches_df = load_data()
    
    # Merge datasets to get text features
    merged = matches_df.merge(resumes_df[['resume_id', 'raw_text']], on='resume_id')
    merged = merged.merge(jobs_df[['job_id', 'description']], on='job_id')
    
    # Create unified text feature
    X = [r + " | " + j for r, j in zip(merged['raw_text'], merged['description'])]
    y = merged['is_shortlisted'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    
    print("Building pipeline...")
    # Using RandomForestClassifier with balanced class weights as per config
    classifier = RandomForestClassifier(
        class_weight=config.CLASS_WEIGHT,
        random_state=config.RANDOM_SEED
    )
    
    pipeline = Pipeline([
        ('features', build_feature_pipeline(include_tfidf=True)),
        ('classifier', classifier)
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    plot_evaluation(y_test, y_pred_proba)
    
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    model_path = config.MODEL_DIR / "best_model.pkl"

    if MLFLOW_AVAILABLE:
        print("Logging to MLflow...")
        mlflow.set_experiment("JobFit_AI_Models")
        with mlflow.start_run():
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("random_seed", config.RANDOM_SEED)
            mlflow.log_param("class_weight", config.CLASS_WEIGHT)
            mlflow.log_metrics(metrics)
            # Skipped mlflow.sklearn.log_model to prevent skops UntrustedTypesFoundException
            
            # Save locally as well
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipeline, model_path)
    else:
        print("MLflow not available. Skipping MLflow logging.")
        # Save locally
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
