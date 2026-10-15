# Model card: fraud_detector

Generated 2026-10-15 19:43 from the model registry. All training data is synthetic.

| Field | Value |
|---|---|
| Purpose | Scores booking sessions for bots, touts, account takeover and card testing, with reason codes. |
| Intended use | Step-up verification or blocking at booking. High-risk: promotions need human approval. |
| Owner | security-ai |
| Risk tier | high |
| Autonomous actions allowed | retrain, tighten_guardrail, notify |
| Primary metric | recall (floor 0.8) |
| Champion version | v1, created 2026-10-15 19:42 |
| Training rows | 8000 |
| Data fingerprint | b8e7e40d1b21 |
| Fairness check | fpr parity across `region_segment` |

## Validated metrics

| Metric | Value |
|---|---|
| recall | 1.0 |
| precision | 0.9796 |
| f1 | 0.9897 |
| fpr | 0.0013 |
| pr_auc | 0.9996 |
| n | 2400 |
| challenge_at | 0.22 |
| block_at | 0.95 |

## Inputs

| Feature | Type | Allowed range | Upstream pipeline |
|---|---|---|---|
| requests_per_min | num | 0 to 1000000.0 | session_telemetry |
| avg_inter_click_ms | num | 0 to 1000000.0 | session_telemetry |
| form_fill_seconds | num | 0 to 1000000.0 | session_telemetry |
| captcha_solve_seconds | num | 0 to 1000000.0 | session_telemetry |
| accounts_on_device | num | 0 to 1000000.0 | session_telemetry |
| ips_last_24h | num | 0 to 1000000.0 | session_telemetry |
| bookings_last_24h | num | 0 to 1000000.0 | session_telemetry |
| tatkal_window_hit | num | 0 to 1000000.0 | session_telemetry |
| payment_failures_24h | num | 0 to 1000000.0 | session_telemetry |
| passengers_per_booking | num | 0 to 1000000.0 | session_telemetry |
| account_age_days | num | 0 to 1000000.0 | session_telemetry |
| headless_signal | num | 0 to 1000000.0 | session_telemetry |
| device_type | cat | mobile, desktop | session_telemetry |
| region_segment | cat | metro, tier2, tier3 | session_telemetry |

## Lineage

| Version | Status | Parent | Notes |
|---|---|---|---|
| v1 | champion | - | bootstrap: trained on synthetic historical data |

## Limitations

Synthetic attack profiles are more separable than real ones; thresholds are cost-based on policy values.
