# Model card: confirmation_predictor

Generated 2026-10-15 19:43 from the model registry. All training data is synthetic.

| Field | Value |
|---|---|
| Purpose | Predicts whether a waitlisted or RAC ticket will be confirmed, RAC or still waitlisted at charting. |
| Intended use | Passenger guidance and journey planning. Not a guarantee of travel. |
| Owner | passenger-ai |
| Risk tier | medium |
| Autonomous actions allowed | retrain, promote, rollback, quarantine_feed, tighten_guardrail, scale_out, notify |
| Primary metric | accuracy (floor 0.7) |
| Champion version | v1, created 2026-10-15 19:42 |
| Training rows | 8000 |
| Data fingerprint | 979beef1ab72 |
| Fairness check | accuracy parity across `quota` |

## Validated metrics

| Metric | Value |
|---|---|
| accuracy | 0.8925 |
| macro_f1 | 0.8047 |
| brier_cnf | 0.0541 |
| ece_cnf | 0.0241 |
| mean_predicted_cnf | 0.3289 |
| actual_cnf_rate | 0.3267 |
| roc_auc_cnf | 0.9799 |
| n | 2400 |

## Inputs

| Feature | Type | Allowed range | Upstream pipeline |
|---|---|---|---|
| train_type | cat | Rajdhani, Duronto, Shatabdi, Vande Bharat, Garib Rath, Superfast, Express | prs_bookings |
| travel_class | cat | 1A, 2A, 3A, SL, CC, EC, 2S | prs_bookings |
| quota | cat | GN, TQ, LD, SS | prs_bookings |
| wl_type | cat | GNWL, RLWL, PQWL, TQWL | prs_bookings |
| current_status | cat | RAC, WL | prs_bookings |
| position | num | 1 to 2000 | prs_bookings |
| days_before_departure | num | 0 to 120 | prs_bookings |
| day_of_week | num | 0 to 6 | calendar |
| month | num | 1 to 12 | calendar |
| is_festival_window | num | 0 to 1 | calendar |
| route_popularity | num | 0 to 1 | prs_inventory |
| hist_cancellation_rate | num | 0 to 1 | prs_bookings |

## Lineage

| Version | Status | Parent | Notes |
|---|---|---|---|
| v1 | champion | - | bootstrap: trained on synthetic historical data |

## Limitations

Trained on synthetic bookings; no real-time cancellation feed, so seasonal collapses in cancellations are learned only after retraining (see docs/experiments.md).
