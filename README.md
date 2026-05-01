# 三孩政策背景下生育意愿影响因素建模项目

## 项目介绍

本项目面向“三孩政策背景下居民再生育意愿影响因素分析”这一研究问题，基于问卷或调查数据构建二分类预测模型，用于判断个体是否具有再生育意愿，并进一步识别影响生育意愿的关键因素。项目将年龄、收入、已有子女数、住房压力、教育成本压力、托育压力、职业风险、政策知晓度、地方政策支持、性别、学历、户口类型、老人帮带等变量纳入分析框架，帮助研究者从经济压力、家庭支持、政策认知和人口特征等角度理解再生育意愿的形成机制。

技术实现上，项目提供了完整的机器学习建模流程，包括数据读取与字段校验、缺失值填补、数值变量标准化、分类变量 One-Hot 编码、训练集与测试集划分、模型训练、测试集评估、交叉验证、可视化评估和模型解释。当前支持逻辑回归、随机森林和 XGBoost 三类模型，既可以通过逻辑回归系数和优势比分析变量方向，也可以通过树模型特征重要性和 SHAP 分析识别非线性模型中的主要影响因素。

项目输出包括模型评估指标、交叉验证结果、ROC 曲线、PR 曲线、混淆矩阵、特征重要性表和 SHAP 可解释性图，可用于课程论文、实证研究、政策分析报告或建模流程验证。需要注意的是，模型结果依赖输入数据质量；项目中的测试数据生成脚本仅用于流程联调，不应用于正式研究结论。

项目结构：

```text
fertility_modeling/
├── main.py                  # 主程序
├── generate_test_data.py    # 生成测试数据，仅用于验证程序能否正常运行
├── config.py                # 变量配置、输出文件名、随机种子
├── preprocessing.py         # 缺失值处理、标准化、One-Hot、数据拆分
├── models.py                # 逻辑回归、随机森林、XGBoost
├── evaluation.py            # 模型评估、ROC、PR、混淆矩阵、交叉验证
├── interpretability.py      # 逻辑回归系数、特征重要性、SHAP
├── outputs/                 # 输出结果
└── requirements.txt
```

## 安装依赖

```powershell
cd fertility_modeling
python -m pip install -r requirements.txt
```

如果缺少 `xgboost` 或 `shap`，代码不会崩溃，会跳过对应部分并给出安装提示。

## 输入数据要求

本项目不生成模拟数据，必须提供真实 CSV 数据。每一行代表一个调查样本，因变量为 `willing_birth`，其余字段为解释变量。

CSV 必须包含以下列名：

| 字段 | 含义 | 类型 / 建议取值 |
| --- | --- | --- |
| `willing_birth` | 是否有再生育意愿 | 0/1，1 表示有意愿 |
| `age` | 年龄 | 数值型，例如 20-50 |
| `gender` | 性别 | 分类变量，例如 `female`、`male` |
| `education` | 学历 | 分类变量，例如 `junior_or_below`、`high_school`、`college`、`bachelor`、`master_or_above` |
| `hukou` | 城乡 / 户口类型 | 分类变量，例如 `urban`、`rural` |
| `income` | 家庭年收入 | 数值型，建议使用人民币年度收入 |
| `children_count` | 已有子女数 | 数值型整数，例如 0、1、2、3 |
| `housing_pressure` | 住房压力 | 数值型，建议 1-5，数值越大压力越高 |
| `education_cost_pressure` | 教育成本压力 | 数值型，建议 1-5，数值越大压力越高 |
| `childcare_pressure` | 托育压力 | 数值型，建议 1-5，数值越大压力越高 |
| `career_risk` | 职业风险 | 数值型，建议 1-5，数值越大风险越高 |
| `elder_help` | 是否有老人帮带 | 分类变量，例如 `yes`、`no` |
| `policy_awareness` | 三孩政策知晓度 | 数值型，建议 1-5，数值越大知晓度越高 |
| `local_policy_support` | 地方政策支持强度 | 数值型，建议 1-5，数值越大支持越强 |

说明：

- `willing_birth` 不应缺失，且应能转换为整数 0/1。
- 解释变量允许少量缺失值：数值变量会用中位数填补，分类变量会用众数填补。
- 分类变量可以使用中文标签，但同一字段内部要保持一致。例如 `gender` 不要同时混用 `female`、`F`、`女` 三套写法，除非你确实希望模型把它们当作不同类别。
- CSV 建议使用 UTF-8 编码。如果是 Excel 文件，建议先另存为 CSV。

## 生成测试数据

如果你暂时没有真实数据，可以单独运行测试数据生成脚本，生成一份符合字段要求的 CSV，用来检查建模流程是否能正常运行。

注意：该数据是合成测试数据，只用于程序联调，不用于正式研究结论。

```powershell
cd fertility_modeling
python generate_test_data.py
```

默认生成：

```text
outputs/test_data.csv
```

也可以自定义样本量和输出路径：

```powershell
python generate_test_data.py --n-samples 3000 --output-path outputs/test_data.csv
```

## 运行

```powershell
cd fertility_modeling
python main.py --data-path your_real_data.csv
```

使用测试数据运行：

```powershell
python main.py --data-path outputs/test_data.csv
```

也可以指定输出目录：

```powershell
python main.py --data-path your_real_data.csv --output-dir outputs_real
```

## 输出结果

默认保存到 `fertility_modeling/outputs/`：

- `model_metrics.csv`
- `cross_validation.csv`
- `logistic_regression_coefficients.csv`
- `random_forest_feature_importance.csv`
- `xgboost_feature_importance.csv`
- `roc_curves.png`
- `pr_curves.png`
- `confusion_matrices.png`
- `shap_xgboost_summary.png`
