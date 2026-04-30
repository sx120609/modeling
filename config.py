from pathlib import Path


RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

TARGET = "willing_birth"

NUMERIC_FEATURES = [
    "age",
    "income",
    "children_count",
    "housing_pressure",
    "education_cost_pressure",
    "childcare_pressure",
    "career_risk",
    "policy_awareness",
    "local_policy_support",
]

CATEGORICAL_FEATURES = [
    "gender",
    "education",
    "hukou",
    "elder_help",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_METRICS_FILE = "model_metrics.csv"
CV_RESULTS_FILE = "cross_validation.csv"
LOGISTIC_COEFFICIENTS_FILE = "logistic_regression_coefficients.csv"
RF_IMPORTANCE_FILE = "random_forest_feature_importance.csv"
XGB_IMPORTANCE_FILE = "xgboost_feature_importance.csv"

ROC_CURVE_FILE = "roc_curves.png"
PR_CURVE_FILE = "pr_curves.png"
CONFUSION_MATRIX_FILE = "confusion_matrices.png"
SHAP_SUMMARY_FILE = "shap_xgboost_summary.png"
