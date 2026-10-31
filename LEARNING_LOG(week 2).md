# Intern Engineering Reflection Journal
**Intern Name:** Abhishek Adbhute  
**Repository:** https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship  
**Cohort:** ML Engineer Internship (Cohort 2026)  

---

### Day 1 : Environment Scaffold & Git Hygiene
- **Concepts Explored:** Virtual environment isolation, dependency pinning with `requirements.txt`, atomic commits with Conventional Commit prefixes (`feat:`, `fix:`, `test:`, `docs:`), and the difference between the `origin` remote (my repo) and `upstream` (the Attenomy starter repo).
- **Key Realizations:** Pinning exact versions means a fresh clone rebuilds the same environment, so I verified it with a clean clone instead of trusting my own machine. My repo history also contains the starter commits, so my own work is the commits on top of them. On Windows PowerShell 5, `&&` is not valid, so commands have to run one per line.
- **Commit Links:** [Pin dependencies and add scipy](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/aa56a7a) | [Add models dir and notebooks placeholder](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/e337049)

### Day 2 : Utility Engineering & Unit Testing
- **Concepts Explored:** Defensive data loading (clear errors for a missing file, an empty file, an unsupported type or missing columns), stratified train/validation/test splitting, index disjointness checks, doctests, and linting with `ruff`.
- **Key Realizations:** The utilities I started from did not match the task specification (different function signature and different keys in the health summary), and rewriting them against the spec and its tests showed how a test suite pins down behaviour. The split uses whole-number sizes because a float ratio like `1 - 0.7` rounds to the wrong row count. The tests assert that the train, validation and test index sets never overlap, which is the formal check for data leakage through the split.
- **Code Links:** [`src/utils.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/src/utils.py) | [`tests/test_utils.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/tests/test_utils.py)
- **Commit Links:** [Align tests and cleaning with the Task 1.1 utils spec](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/2c6c73b)

### Day 3 : Data Cleaning & Parquet Transformation
- **Concepts Explored:** Deterministic cleaning versus fitted preprocessing, missing-value strategies (median for skewed numbers, `UNKNOWN` for categories), IQR outlier flagging, columnar storage with Parquet and Snappy compression, and leak-free preprocessing with a scikit-learn `Pipeline` and `ColumnTransformer`.
- **Key Realizations:** The leakage rule changed my design. The starting cleaning script filled missing numbers with a median computed on the whole dataset, so I moved that step out and now fit the imputer, scaler and encoders on training rows only. In `error_code`, 91.8% of values were missing, and these almost certainly mean "no error happened", so I filled them with `NO_ERROR`, not `UNKNOWN`, and documented it as an assumption. Parquet stored the data in 0.69 MB against 2.63 MB for the CSV, and in-memory size fell from 3.99 MB to 1.96 MB.
- **Code Links:** [`src/cleaning.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/src/cleaning.py) | [`notebooks/01_cleaning.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/01_cleaning.ipynb) | [`reports/data_dictionary.md`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/reports/data_dictionary.md)
- **Commit Links:** [Rewrite cleaning pipeline](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/840d0db) | [Add cleaned parquet](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/a01eecf) | [Add data dictionary](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/e3a3915) | [Add cleaning notebook](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/e2ff719)

### Day 4 : Statistical EDA & Hypothesis Testing
- **Concepts Explored:** Chi-square test of independence, Mann-Whitney U test for skewed data, Bonferroni correction for running several tests, effect sizes (Cramer's V and AUC), and Spearman rank correlation.
- **Key Realizations:** A small p-value does not mean a large effect. Latency was significantly different between classes (p = 0.0028), but the AUC was only 0.534 and the medians were 73.8 ms and 70.5 ms, so the difference is tiny. In contrast, `error_code` had Cramer's V = 0.285, with 2.3% anomalies when there was no error and 21% to 24% when there was one. That strong link could also mean label leakage, so it went into the risk section of the memo. I used Mann-Whitney instead of a t-test because latency is strongly right-skewed (skewness 10.9).
- **Code Links:** [`notebooks/02_eda.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/02_eda.ipynb) | [`reports/hypothesis_results.csv`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/reports/hypothesis_results.csv)
- **Commit Links:** [Add EDA notebook with 8 figures](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/fe649ba) | [Add hypothesis tests](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/3a19966)

### Day 5 : Deliverables Packaging & Gate 1 Review
- **Concepts Explored:** Executive memo writing, turning statistical results into risks and a modeling plan, and a final reproducibility check from a fresh clone (`pip install -r requirements.txt`, `pytest`, `ruff check .`).
- **Key Realizations:** The memo is only useful if each risk (imbalance, label leakage, collinearity) is tied to evidence from the analysis and has a concrete mitigation for Week 2. Writing down the open questions (is `error_code` known before the label, is `NO_ERROR` the right reading) was as important as the findings.
- **Code Links:** [`reports/week1_insights_memo.pdf`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/reports/week1_insights_memo.pdf)
- **Commit Links:** [Final submission commits](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commits/main)

---

# Week 2 : Classical ML and Anti-Fraud Pipeline

### Day 6 : Baselines and a Leak-Free Pipeline
- **Concepts Explored:** Majority-class and stratified dummy baselines, stratified k-fold cross-validation, PR-AUC as the main metric for imbalanced data, and a single preprocessing pipeline that is refit inside every fold.
- **Key Realizations:** A model that always says "normal" scores 96.1% accuracy but only 0.039 PR-AUC, so accuracy hides a useless model. Every later model had to beat that PR-AUC floor of about 0.04 on the same folds.
- **Code Links:** [`src/pipeline.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/src/pipeline.py) | [`notebooks/03_baselines.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/03_baselines.ipynb)

### Day 7 : Comparing Four Classical Models Fairly
- **Concepts Explored:** Benchmarking models under identical folds and identical preprocessing, an ablation with and without one feature (`error_code`), and a rerun check that the results are reproducible.
- **Key Realizations:** A comparison is only fair if every model sees the same splits and the same pipeline, and removing one feature is the quickest way to test whether a model depends on it.
- **Code Links:** [`notebooks/03_baselines.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/03_baselines.ipynb)

### Day 8 : Feature Engineering Without Leakage
- **Concepts Explored:** Stateless features (rates, time of day, account history built only from earlier calls) versus stateful ones (frequency and target encoding of `account_id`, quantile bins) that must be refit inside every fold, and custom scikit-learn transformers.
- **Key Realizations:** Target encoding needs cross-fitting so a row never sees its own label. The ablation showed no clear gain from the new features (42 of 52 add-or-drop comparisons had no clear effect, and the largest change was +0.015 PR-AUC; tuned models scored 0.269 to 0.272 on every feature set), so I kept the nine raw features and reported the negative result.
- **Code Links:** [`src/features.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/src/features.py) | [`tests/test_stateful.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/tests/test_stateful.py)
- **Commit Links:** [Stateless features](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/d1c7bb4) | [Feature tests](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/0fffab8) | [Pipeline loads features](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/2d14814)

### Day 9 : Tuning and Statistical Model Comparison
- **Concepts Explored:** Randomized hyperparameter search inside the leak-free pipeline, repeated cross-validation with the Nadeau-Bengio corrected interval, paired tests between models, and saving a model that reloads in a fresh process.
- **Key Realizations:** The tuned XGBoost beat the logistic-regression baseline by +0.045 PR-AUC (95% interval +0.020 to +0.071, p = 0.002), a real but modest gain that held on validation (0.246 against 0.203). With about 680 positive training rows the fold-to-fold spread is about 0.04, so I reported intervals instead of chasing a spread target that the data cannot support.
- **Code Links:** [`notebooks/04_tuning.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/04_tuning.ipynb) | [`scripts/verify_model.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/scripts/verify_model.py)
- **Commit Links:** [Tuning and repeated-CV comparison](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/aaea819) | [Save and verify the model](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/6938d6e)

### Day 10 : Explainability, Calibration, Threshold and Audit
- **Concepts Explored:** SHAP values, probability calibration (Platt scaling versus isotonic regression), choosing a decision threshold on out-of-fold scores, error analysis, and a subgroup audit with Wilson intervals.
- **Key Realizations:** The model's skill comes almost entirely from one column: without `error_code` PR-AUC falls by 54%, and the one-line rule "alert whenever an error code is present" reaches precision 0.207 and recall 0.422 against 0.227 and 0.422 for the model. Raw scores were not probabilities (mean 0.375 against a true rate of 0.039) until calibrated, and 58% of anomalies have no error code and are almost never caught.
- **Code Links:** [`notebooks/05_explainability.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/05_explainability.ipynb) | [`notebooks/06_calibration_threshold.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/06_calibration_threshold.ipynb) | [`notebooks/07_error_analysis_audit.ipynb`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/notebooks/07_error_analysis_audit.ipynb) | [`reports/week2_report.pdf`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/reports/week2_report.pdf)
- **Commit Links:** [SHAP](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/0e8b363) | [Calibration and threshold](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/198d5de) | [Saved calibrator and figures](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/cb1f253) | [Error analysis and audit](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/b86b9bc)
