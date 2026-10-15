# Model card: fare_demand_forecaster

Generated 2026-10-15 19:43 from the model registry. All training data is synthetic.

| Field | Value |
|---|---|
| Purpose | Forecasts the share of seats sold by day; a deterministic tariff converts it to fare and booking-window advice. |
| Intended use | Fare forecasts and savings suggestions. Tariff parameters are illustrative, not official. |
| Owner | revenue-ai |
| Risk tier | medium |
| Autonomous actions allowed | retrain, promote, rollback, quarantine_feed, tighten_guardrail, scale_out, notify |
| Primary metric | mae (floor 0.08) |
| Champion version | v1, created 2026-10-15 19:42 |
| Training rows | 8000 |
| Data fingerprint | d1d2a6ecef6f |
| Fairness check | - parity across `-` |

## Validated metrics

| Metric | Value |
|---|---|
| mae | 0.0295 |
| r2 | 0.9771 |
| n | 2000 |

## Inputs

| Feature | Type | Allowed range | Upstream pipeline |
|---|---|---|---|
| train_type | cat | Rajdhani, Duronto, Shatabdi, Vande Bharat, Garib Rath, Superfast, Express | prs_inventory |
| travel_class | cat | 1A, 2A, 3A, SL, CC, EC, 2S | prs_inventory |
| days_before_departure | num | 0 to 120 | prs_inventory |
| month | num | 1 to 12 | calendar |
| day_of_week | num | 0 to 6 | calendar |
| demand_index | num | 0.3 to 3.0 | calendar |
| route_popularity | num | 0 to 1 | prs_inventory |

## Lineage

| Version | Status | Parent | Notes |
|---|---|---|---|
| v1 | champion | - | bootstrap: trained on synthetic historical data |

## Limitations

Demand curves are synthetic; no real inventory feed.
