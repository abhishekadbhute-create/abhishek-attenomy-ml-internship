# Intern Engineering Reflection Journal**Intern Name:** Abhishek Adbhute  
**Repository:** https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship  
**Cohort:** ML Engineer Internship (Cohort 2026)  

---

### Day 1 : Environment Scaffold & Git Hygiene
- **Concepts Explored:** Virtual environment isolation, dependency pinning with `requirements.txt`, atomic commits with Conventional Commit prefixes (`feat:`, `fix:`, `test:`, `docs:`), and the difference between the `origin` remote (my repo) and `upstream` (the Attenomy starter repo).
- **Key Realizations:** Pinning exact versions means a fresh clone rebuilds the same environment, so I verified it with a clean clone instead of trusting my own machine. My repo history also contains the starter commits, so my own work is the commits on top of them. On Windows PowerShell 5, `&&` is not valid, so commands have to run one per line.
- **Commit Links:** [Pin dependencies and add scipy](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/aa56a7a) | [Add models dir and notebooks placeholder](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/e337049)

### Day 2 : Utility Engineering & Unit Testing
- **Concepts Explored:** Defensive data loading (clear errors for a missing file, an empty file, an unsupported type or missing columns), stratified train/validation/test splitting, index disjointness checks, doctests, and linting with `ruff`.
- **Key Realizations:** My first version of the utilities did not match the task specification (different function signature and different keys in the health summary), and rewriting it against the spec and its tests showed how a test suite pins down behaviour. The split uses whole-number sizes because a float ratio like `1 - 0.7` rounds to the wrong row count. The tests assert that the train, validation and test index sets never overlap, which is the formal check for data leakage through the split.
- **Code Links:** [`src/utils.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/src/utils.py) | [`tests/test_utils.py`](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/blob/main/tests/test_utils.py)
- **Commit Links:** [Align tests and cleaning with the Task 1.1 utils spec](https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship/commit/2c6c73b)

### Day 3 : Data Cleaning & Parquet Transformation
- **Concepts Explored:** Deterministic cleaning versus fitted preprocessing, missing-value strategies (median for skewed numbers, `UNKNOWN` for categories), IQR outlier flagging, columnar storage with Parquet and Snappy compression, and leak-free preprocessing with a scikit-learn `Pipeline` and `ColumnTransformer`.
- **Key Realizations:** The leakage rule changed my design. My first cleaning script filled missing numbers with a median computed on the whole dataset, so I moved that step out and now fit the imputer, scaler and encoders on training rows only. In `error_code`, 91.8% of values were missing, and these almost certainly mean "no error happened", so I filled them with `NO_ERROR`, not `UNKNOWN`, and documented it as an assumption. Parquet stored the data in 0.69 MB against 2.63 MB for the CSV, and in-memory size fell from 3.99 MB to 1.96 MB.
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
ek 1 final submission commit]
