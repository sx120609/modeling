import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    LOGISTIC_COEFFICIENTS_FILE,
    RF_IMPORTANCE_FILE,
    SHAP_SUMMARY_FILE,
    XGB_IMPORTANCE_FILE,
)
from preprocessing import get_feature_names


def save_logistic_regression_coefficients(pipeline, output_dir):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = get_feature_names(preprocessor)
    coefficients = model.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "direction": np.where(coefficients > 0, "positive", "negative"),
        }
    ).sort_values("coefficient", ascending=False)

    coef_df.to_csv(output_dir / LOGISTIC_COEFFICIENTS_FILE, index=False, encoding="utf-8-sig")
    return coef_df


def save_tree_feature_importance(pipeline, output_dir, output_file):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = get_feature_names(preprocessor)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(output_dir / output_file, index=False, encoding="utf-8-sig")
    return importance_df


def run_xgboost_shap_analysis(pipeline, X_test, output_dir, max_samples=600):
    try:
        import shap
    except ImportError:
        print("未检测到 shap，已跳过 SHAP 分析。安装命令: python -m pip install shap")
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_test)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = get_feature_names(preprocessor)
    X_sample = X_transformed[: min(max_samples, X_transformed.shape[0])]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_dir / SHAP_SUMMARY_FILE, dpi=180, bbox_inches="tight")
    plt.close()

    return shap_values


def save_interpretability_results(fitted_models, X_test, output_dir):
    results = {}

    if "logistic_regression" in fitted_models:
        results["logistic_coefficients"] = save_logistic_regression_coefficients(
            fitted_models["logistic_regression"],
            output_dir,
        )

    if "random_forest" in fitted_models:
        results["random_forest_importance"] = save_tree_feature_importance(
            fitted_models["random_forest"],
            output_dir,
            RF_IMPORTANCE_FILE,
        )

    if "xgboost" in fitted_models:
        results["xgboost_importance"] = save_tree_feature_importance(
            fitted_models["xgboost"],
            output_dir,
            XGB_IMPORTANCE_FILE,
        )
        run_xgboost_shap_analysis(fitted_models["xgboost"], X_test, output_dir)

    return results
