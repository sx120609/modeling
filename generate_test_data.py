import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, OUTPUT_DIR, RANDOM_STATE, TARGET


DEFAULT_TEST_DATA_PATH = OUTPUT_DIR / "test_data.csv"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_missing_values(df, missing_rate=0.03, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    result = df.copy()

    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        mask = rng.random(len(result)) < missing_rate
        result.loc[mask, col] = np.nan

    return result


def generate_test_data(n_samples=3000, random_state=RANDOM_STATE, missing_rate=0.03):
    """
    Generate synthetic test data that matches the required schema.

    This file is for pipeline testing only. It should not be used as empirical
    evidence for actual fertility willingness research.
    """
    rng = np.random.default_rng(random_state)

    age = np.clip(rng.normal(34, 6.5, n_samples).round(), 22, 49).astype(int)
    gender = rng.choice(["female", "male"], n_samples, p=[0.52, 0.48])
    education = rng.choice(
        ["junior_or_below", "high_school", "college", "bachelor", "master_or_above"],
        n_samples,
        p=[0.12, 0.22, 0.26, 0.30, 0.10],
    )
    hukou = rng.choice(["rural", "urban"], n_samples, p=[0.38, 0.62])
    children_count = rng.choice([0, 1, 2, 3], n_samples, p=[0.10, 0.48, 0.34, 0.08])

    education_income_effect = pd.Series(education).map(
        {
            "junior_or_below": -0.35,
            "high_school": -0.15,
            "college": 0.05,
            "bachelor": 0.25,
            "master_or_above": 0.45,
        }
    ).to_numpy()
    hukou_income_effect = np.where(hukou == "urban", 0.22, -0.08)
    log_income = rng.normal(11.2 + education_income_effect + hukou_income_effect, 0.55)
    income = np.round(np.exp(log_income), -3)
    income = np.clip(income, 30000, 600000)
    income_scaled = (np.log(income) - np.mean(np.log(income))) / np.std(np.log(income))

    housing_pressure = np.clip(
        rng.normal(3.3 + 0.35 * (hukou == "urban") - 0.25 * income_scaled, 0.9, n_samples),
        1,
        5,
    ).round()
    education_cost_pressure = np.clip(
        rng.normal(3.1 + 0.20 * children_count - 0.18 * income_scaled, 0.85, n_samples),
        1,
        5,
    ).round()
    childcare_pressure = np.clip(
        rng.normal(2.8 + 0.35 * children_count - 0.25 * (age > 40), 0.9, n_samples),
        1,
        5,
    ).round()
    career_risk = np.clip(
        rng.normal(2.8 + 0.65 * (gender == "female") + 0.15 * (education == "bachelor"), 0.95, n_samples),
        1,
        5,
    ).round()

    elder_help = rng.choice(["no", "yes"], n_samples, p=[0.44, 0.56])
    policy_awareness = np.clip(
        rng.normal(2.7 + 0.35 * (education == "bachelor") + 0.55 * (education == "master_or_above"), 0.95, n_samples),
        1,
        5,
    ).round()
    local_policy_support = np.clip(
        rng.normal(2.6 + 0.18 * (hukou == "urban") + rng.normal(0, 0.25, n_samples), 0.8, n_samples),
        1,
        5,
    ).round()

    two_child_policy_sensitivity = (children_count == 2) * (
        0.35 * policy_awareness + 0.45 * local_policy_support
    )
    no_child_or_one_child_effect = np.where(children_count <= 1, 0.25, 0)
    three_child_penalty = np.where(children_count >= 3, -1.25, 0)

    linear_score = (
        1.10
        - 0.045 * (age - 34)
        + 0.42 * income_scaled
        - 0.33 * housing_pressure
        - 0.44 * education_cost_pressure
        - 0.38 * childcare_pressure
        - 0.30 * career_risk
        + 0.60 * (elder_help == "yes")
        + 0.28 * policy_awareness
        + 0.36 * local_policy_support
        + 0.14 * (hukou == "rural")
        - 0.25 * (gender == "female")
        + no_child_or_one_child_effect
        + three_child_penalty
        + 0.25 * two_child_policy_sensitivity
        + rng.normal(0, 0.45, n_samples)
    )

    willing_birth = rng.binomial(1, sigmoid(linear_score))

    df = pd.DataFrame(
        {
            TARGET: willing_birth,
            "age": age,
            "gender": gender,
            "education": education,
            "hukou": hukou,
            "income": income.astype(int),
            "children_count": children_count,
            "housing_pressure": housing_pressure.astype(int),
            "education_cost_pressure": education_cost_pressure.astype(int),
            "childcare_pressure": childcare_pressure.astype(int),
            "career_risk": career_risk.astype(int),
            "elder_help": elder_help,
            "policy_awareness": policy_awareness.astype(int),
            "local_policy_support": local_policy_support.astype(int),
        }
    )

    return add_missing_values(df, missing_rate=missing_rate, random_state=random_state)


def parse_args():
    parser = argparse.ArgumentParser(description="生成用于测试建模流程的合成 CSV 数据")
    parser.add_argument("--output-path", type=str, default=str(DEFAULT_TEST_DATA_PATH), help="测试数据输出路径。")
    parser.add_argument("--n-samples", type=int, default=3000, help="测试数据样本量。")
    parser.add_argument("--missing-rate", type=float, default=0.03, help="解释变量缺失比例。")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="随机种子。")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_test_data(
        n_samples=args.n_samples,
        random_state=args.random_state,
        missing_rate=args.missing_rate,
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"测试数据已生成: {output_path.resolve()}")
    print(f"样本量: {len(df)}")
    print(f"willing_birth=1 比例: {df[TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
