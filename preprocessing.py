import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def validate_columns(df):
    required_columns = [TARGET] + ALL_FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"数据缺少必要字段: {missing_columns}")


def split_features_target(df):
    validate_columns(df)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int)
    return X, y


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def build_preprocessor():
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor):
    numeric_names = list(
        preprocessor.named_transformers_["num"].get_feature_names_out(NUMERIC_FEATURES)
    )
    categorical_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
    )
    return numeric_names + categorical_names


def make_train_test_split(X, y, test_size, random_state):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def load_real_data(data_path):
    return pd.read_csv(data_path)
