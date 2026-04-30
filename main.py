import argparse
import warnings
from pathlib import Path

from config import (
    CV_FOLDS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from evaluation import (
    fit_and_evaluate_models,
    run_cross_validation,
    save_all_evaluation_plots,
)
from interpretability import save_interpretability_results
from models import build_models
from preprocessing import load_real_data, make_train_test_split, split_features_target

warnings.filterwarnings("ignore", category=UserWarning)


def run_pipeline(
    data_path,
    output_dir=OUTPUT_DIR,
    test_size=TEST_SIZE,
    cv_folds=CV_FOLDS,
    random_state=RANDOM_STATE,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_real_data(data_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    models = build_models(random_state=random_state)

    fitted_models, metrics_df = fit_and_evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir,
    )
    cv_df = run_cross_validation(
        models,
        X,
        y,
        output_dir,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    save_all_evaluation_plots(fitted_models, X_test, y_test, output_dir)
    save_interpretability_results(fitted_models, X_test, output_dir)

    return metrics_df, cv_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="三孩政策背景下再生育意愿影响因素建模流程"
    )
    parser.add_argument("--data-path", type=str, required=True, help="真实数据 CSV 路径。")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="结果输出目录。")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="测试集比例。")
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS, help="交叉验证折数。")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="随机种子。")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_df, cv_df = run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
    )

    print("\n测试集评估结果:")
    print(metrics_df.to_string(index=False))
    print("\n五折交叉验证结果:")
    print(cv_df.to_string(index=False))
    print(f"\n结果已保存到: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
