"""
Model evaluation module for JobFit AI.
Provides functions to evaluate model performance and plot ROC curves.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

from src import config

def evaluate_model(pipeline, X_test, y_test) -> dict:
    """Evaluate the trained pipeline."""
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

from sklearn.calibration import calibration_curve

def plot_evaluation(y_test, y_pred_proba):
    """Plot ROC curve and calibration curve, then save them."""
    chart_dir = config.NOTEBOOK_DIR / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(chart_dir / "roc_curve.png")
    plt.close()

    # 2. Plot Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.savefig(chart_dir / "calibration_curve.png")
    plt.close()

if __name__ == "__main__":
    # Simple standalone test block
    print("Evaluate module loaded successfully.")
