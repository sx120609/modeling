import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import (
    CONFUSION_MATRIX_FILE,
    CV_RESULTS_FILE,
    MODEL_METRICS_FILE,
    PR_CURVE_FILE,
    ROC_CURVE_FILE,
)
from models import build_model_pipeline


def fit_and_evaluate_models(models, X_train, X_test, y_train, y_test, output_dir):
    fitted_models = {}
    metric_rows = []

    for model_name, model in models.items():
        pipeline = build_model_pipeline(model)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fitted_models[model_name] = pipeline
        metric_rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "average_precision": average_precision_score(y_test, y_prob),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(output_dir / MODEL_METRICS_FILE, index=False, encoding="utf-8-sig")

    return fitted_models, metrics_df


def run_cross_validation(models, X, y, output_dir, cv_folds=5, random_state=42):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }

    cv_rows = []
    for model_name, model in models.items():
        scores = cross_validate(
            build_model_pipeline(model),
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        row = {"model": model_name}
        for metric_name in scoring:
            values = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = values.mean()
            row[f"{metric_name}_std"] = values.std()
        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows).sort_values("roc_auc_mean", ascending=False)
    cv_df.to_csv(output_dir / CV_RESULTS_FILE, index=False, encoding="utf-8-sig")

    return cv_df


def plot_roc_curves(fitted_models, X_test, y_test, output_dir):
    plt.figure(figsize=(8, 6))

    for model_name, pipeline in fitted_models.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_value = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / ROC_CURVE_FILE, dpi=180)
    plt.close()


def plot_pr_curves(fitted_models, X_test, y_test, output_dir):
    plt.figure(figsize=(8, 6))

    for model_name, pipeline in fitted_models.items():
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f"{model_name} (AP={average_precision:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / PR_CURVE_FILE, dpi=180)
    plt.close()


def plot_confusion_matrices(fitted_models, X_test, y_test, output_dir):
    n_models = len(fitted_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, pipeline) in zip(axes, fitted_models.items()):
        y_pred = pipeline.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)

        ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

        ax.set_title(model_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], labels=["0", "1"])
        ax.set_yticks([0, 1], labels=["0", "1"])

    plt.tight_layout()
    plt.savefig(output_dir / CONFUSION_MATRIX_FILE, dpi=180)
    plt.close()


def save_all_evaluation_plots(fitted_models, X_test, y_test, output_dir):
    plot_roc_curves(fitted_models, X_test, y_test, output_dir)
    plot_pr_curves(fitted_models, X_test, y_test, output_dir)
    plot_confusion_matrices(fitted_models, X_test, y_test, output_dir)
