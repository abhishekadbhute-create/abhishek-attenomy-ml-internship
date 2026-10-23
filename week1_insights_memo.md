# Week 1 Insights Memo: Attenomy Telemetry Anomaly Dataset

Author: Abhishek Adbhute (ML Engineering Intern) | Date: 23 October 2026 | For: Attenomy ML Lead | Repository: https://github.com/abhishekadbhute-create/abhishek-attenomy-ml-internship

## 1. Summary

- The dataset has 25,000 service calls and a binary target, **is_anomaly**, which is positive for only 3.9% of rows. It was cleaned and exported reproducibly to Parquet.
- The strongest signal is whether an error was recorded: calls with any error code are anomalous 21% to 24% of the time, against 2.3% for calls with no error. This is also the biggest risk, because the field may reveal the label.
- Every numeric feature is weak on its own (best rank correlation +0.06, best class separation AUC 0.58). Numeric features are not collinear (all pairwise rank correlations are 0.02 or below).
- Week 2 should build a leak-free, stratified pipeline, compare models with and without **error_code**, and judge them by precision-recall metrics, not accuracy.

## 2. Data and cleaning

- **Checked, nothing found:** no duplicate rows, no malformed or repeated transaction IDs, no unparseable timestamps.
- **Fixed:** channel casing (7 raw values became 4); 809 missing client_tier values became UNKNOWN; 91.8% missing error_code values became NO_ERROR (assumption: missing means no error occurred); 60 negative session durations were set to missing.
- **Left for the training split:** 1,032 missing session durations (4.1%) are filled with the median fitted on training rows only, because the metric is right-skewed.
- **Outliers** are flagged with the IQR rule, not removed (latency reaches 4,546 ms against a median of 71 ms), because extreme values may be the anomalies.
- **Efficiency:** Parquet is 0.69 MB against 2.63 MB for the CSV; memory fell from 3.99 MB to 1.96 MB (51%).
- **Leakage control:** the data is split 70/15/15 (stratified, seed 42) before any fitted step. The imputer, scaler and encoders are fit on training rows only, and all EDA used the training split (17,500 rows, 685 anomalies).

## 3. Findings

| Hypothesis | Test and statistic | Effect size | Result and meaning |
|---|---|---|---|
| H1: anomaly rate depends on error_code | Chi-square, chi2 = 1426.2, dof = 5, p < 1e-300 | Cramer's V = 0.285 | Rejected. Strong link: 2.3% anomalies with no error, 20.9% to 23.8% with any error. |
| H2: latency differs by class | Mann-Whitney U = 6,146,454, dof N/A, p = 0.0028 | AUC = 0.534 | Rejected, but tiny: median 73.8 ms vs 70.5 ms. |
| H3: anomaly rate depends on client_tier | Chi-square, chi2 = 1.75, dof = 4, p = 0.782 | Cramer's V = 0.010 | Not rejected. No evidence that the plan level matters. |
| H4: session duration differs by class | Mann-Whitney U = 5,163,200, dof N/A, p = 0.392 | AUC = 0.490 | Not rejected. Sessions look the same in both classes. |
| H5: anomalies have more retries (one-sided) | Mann-Whitney U = 6,679,158, dof N/A, p = 2.5e-17 | AUC = 0.580 | Rejected. Anomalous calls retry more, a modest effect. |

All tests use the training split. The decision threshold is 0.01 (Bonferroni: 0.05 divided by 5 tests). Mann-Whitney was used instead of a t-test because the features are strongly right-skewed (skewness: latency 10.9, payload 5.2, retry count 2.5, session duration 1.4); it compares ranks, so outliers do not distort it. Mann-Whitney has no degrees of freedom (non-parametric). A significant p-value is not enough on its own: with 17,500 rows even tiny differences are significant, so effect sizes are reported.

- The five error types are almost equally risky (20.9% to 23.8%), so what matters is that an error happened, not which one.
- Anomaly rates barely vary by client tier (3.5% to 4.1%) or channel (3.5% to 4.2%). The one hot spot is cli_tool with batch_sync at 7.0% against 3.9% overall; the cell is small, so treat it as a lead to check, not a conclusion.
- Latency outliers are spread evenly: the highest share is 2.0% of calls (channel rest_api, region ap-south-1).

## 4. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| Target imbalance | 3.9% positives. A model that always says 'normal' scores 96.1% accuracy. | Stratified splits, class weights, PR-AUC and recall at a fixed precision as the main metrics, threshold tuned on validation only. |
| Label leakage | error_code gives a 10x lift, while the best numeric feature has AUC 0.58, so models will lean on it. IQR outlier flags were computed on all rows. | Confirm with the data owner that error_code is known before the anomaly decision. Train with and without it. Keep the outlier flags out of the model, or recompute them on training rows. |
| Collinearity | Low among numeric features (all pairs 0.02 or below). One-hot error codes overlap with NO_ERROR. | Drop one level when encoding, use regularisation, check VIF after encoding. Tree models are unaffected. |
| Few positives | Only 685 anomalies in training, so results will be noisy. | Stratified 5-fold cross-validation on training rows, report spread, touch the test set once at the end. |
| Cleaning assumptions | NO_ERROR for missing error_code and median imputation of 4.1% of durations. | Confirm the NO_ERROR meaning with the data owner; test the median choice against a missing-indicator column. |

## 5. Week 2 classical modeling plan

1. **Baselines:** a dummy classifier and logistic regression (balanced class weights) on the existing train-only ColumnTransformer.
2. **Tree models:** random forest and histogram gradient boosting, tuned with stratified cross-validation on training rows.
3. **Leakage ablation:** compare all models with error_code, with only a has_error flag, and without it, to measure how much performance depends on it.
4. **Feature work (fit on training rows only):** hour and weekday from the timestamp, log transforms of skewed features, outlier flags recomputed on training rows, and per-account history features that use only earlier timestamps.
5. **Evaluation:** PR-AUC as the primary metric, plus recall at a precision target agreed with the mentor, F1 and the confusion matrix on validation. Save the winning pipeline in models/ and add tests for any new functions.

**Questions for review:** (1) Should clean.parquet itself have zero nulls, or is filling them after the split acceptable? (2) Is error_code known before the anomaly label is set? (3) Is NO_ERROR the right reading of a missing error_code?
