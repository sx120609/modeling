from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


def build_logistic_regression(random_state=42):
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )


def build_random_forest(random_state=42):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(random_state=42):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("未检测到 xgboost，已跳过 XGBoost。安装命令: python -m pip install xgboost")
        return None

    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_models(random_state=42):
    model_dict = {
        "logistic_regression": build_logistic_regression(random_state),
        "random_forest": build_random_forest(random_state),
    }

    xgb_model = build_xgboost(random_state)
    if xgb_model is not None:
        model_dict["xgboost"] = xgb_model

    return model_dict


def build_model_pipeline(model):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
