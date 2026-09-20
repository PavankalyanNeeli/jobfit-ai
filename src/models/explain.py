"""
Explainability module for JobFit AI.
Provides functions to generate SHAP values for model predictions.
"""
import shap
from src import config

def get_explainer(pipeline, X_background):
    """
    Returns a SHAP TreeExplainer fitted on the transformed background data.
    
    Args:
        pipeline: The trained sklearn pipeline.
        X_background: Background data (list of texts) to fit the explainer.
        
    Returns:
        shap.TreeExplainer: The fitted explainer.
    """
    import numpy as np
    X_transformed = pipeline.named_steps['features'].transform(X_background)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.array(X_transformed, dtype=np.float64)
    
    # Get the classifier step
    classifier = pipeline.named_steps['classifier']
    
    # Create and return the TreeExplainer
    return shap.TreeExplainer(classifier, X_transformed)

def explain_prediction(explainer, pipeline, X_instance):
    """
    Returns SHAP values for a single prediction.
    
    Args:
        explainer: The fitted SHAP explainer.
        pipeline: The trained sklearn pipeline.
        X_instance: A single instance (list of one text).
        
    Returns:
        shap.Explanation or numpy.ndarray: The SHAP values.
    """
    import numpy as np
    X_transformed = pipeline.named_steps['features'].transform(X_instance)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.array(X_transformed, dtype=np.float64)
    
    # Generate SHAP values
    shap_values = explainer(X_transformed)
    return shap_values

if __name__ == "__main__":
    print("Explain module loaded successfully.")
