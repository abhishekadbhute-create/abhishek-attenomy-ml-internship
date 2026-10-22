# Data Dictionary: Attenomy Telemetry Dataset

Source: `data/raw/dataset.csv` (25,000 rows, 13 columns).
Cleaned output: `data/clean/clean.parquet` (25,000 rows, 16 columns, Snappy-compressed).
Target: `is_anomaly` (3.9% positive, so the classes are imbalanced).

## Schema

| Column | Type (clean) | Business meaning | Allowed range / values | Missing-value policy | Governance |
|---|---|---|---|---|---|
| transaction_id | string | Unique ID of one service interaction. Primary key. | Format `TRX_<digits>`, unique | No nulls found. Rows with a malformed or duplicate key are dropped (0 found). | Internal identifier. No personal data. |
| account_id | string | Customer account that made the call (1,501 distinct accounts). | Format `ACT_<digits>` | No nulls found. Blank values would become `UNKNOWN`. | Pseudonymous identifier. Do not join with external personal data. |
| timestamp | datetime64 (UTC) | When the interaction happened. | Valid ISO-8601 UTC timestamp | Rows with an unparseable timestamp are dropped (0 found). | Operational metadata. |
| interaction_type | category | Kind of service call. | api_query, doc_extraction, agent_orchestration, model_inference, batch_sync | No nulls found. Blanks would become `UNKNOWN`. | Internal. |
| channel | category | Access channel used. | web_app, python_sdk, rest_api, cli_tool | No nulls found. Casing variants were merged (7 raw values became 4). | Internal. |
| session_duration_sec | float64 | Session length in seconds. Right-skewed. | 0 to about 833 (observed). Negative values are invalid. | 972 raw nulls plus 60 negative values set to NaN (4.13% of rows). Kept as NaN in the parquet and imputed with the **median fit on the training split only**. | Internal. |
| payload_size_kb | float64 | Size of the request payload in kilobytes. | 0.31 to 917.07 (observed). Must be >= 0. | No nulls. Outliers are flagged, not removed. | Internal. |
| latency_ms | float64 | Response latency in milliseconds. | 10 to 4,706 (observed). Must be >= 0. | No nulls. Outliers are flagged, not removed. | Internal. |
| retry_count | int32 | Number of retries before the call finished. | 0 to 7 (observed). Must be >= 0. | No nulls. | Internal. |
| error_code | category | Error returned by the call. A missing value means **no error occurred**. | NO_ERROR, err_auth_expired, err_rate_limit, err_payload_too_large, err_upstream_unavailable, err_timeout | 22,954 raw nulls (91.8%) are mapped to `NO_ERROR`. This is an assumption: nulls mean "no error", not "unknown". | Internal. |
| client_tier | category | Customer plan level. | starter, pro, enterprise, vip, UNKNOWN | 809 raw nulls are filled with `UNKNOWN` (a domain constant). | Commercial data. Internal use only. |
| region | category | Cloud region that served the call. | us-east-1, us-west-2, eu-central-1, ap-south-1 | No nulls found. | Internal. |
| is_anomaly | int32 | **Target.** 1 if the interaction was flagged as an anomaly or abuse, else 0. | 0 or 1 | No nulls. Never imputed. | Label. Never used as a feature. |
| session_duration_sec_is_outlier | int8 | Derived flag: 1 if the value is outside the IQR fences (1.5 x IQR). | 0 or 1 | No nulls. | Derived feature. |
| payload_size_kb_is_outlier | int8 | Derived flag, same rule. | 0 or 1 | No nulls. | Derived feature. |
| latency_ms_is_outlier | int8 | Derived flag, same rule. | 0 or 1 | No nulls. | Derived feature. |

## Cleaning decisions

- Text columns are stripped, lowercased and normalized. Blank, `nan` and `none` strings become `UNKNOWN`.
- Primary keys and timestamps were validated: 0 corrupted keys and 0 duplicate rows were found.
- Invalid numeric values (negative) are set to NaN, not guessed.
- Outliers are flagged with the IQR rule and kept, because extreme latency or payload values may be the anomalies we want to detect.
- Memory: 3.99 MB to 1.96 MB (51% saved) by downcasting int64 to int32 and converting low-cardinality strings to category.

## Preprocessing plan (fit on the training split only)

| Step | Columns | Method |
|---|---|---|
| Imputation | session_duration_sec | Median, because the metric is right-skewed |
| Scaling | session_duration_sec, payload_size_kb, latency_ms, retry_count | StandardScaler (fit on train only) |
| Nominal encoding | interaction_type, channel, error_code, region | One-hot |
| Ordinal mapping | client_tier | UNKNOWN=0, starter=1, pro=2, enterprise=3, vip=4 (assumed plan order) |
| Dropped from modeling | transaction_id, account_id, timestamp | Identifiers and time. Timestamp may be used for engineered features. |