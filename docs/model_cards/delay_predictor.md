# Model card: delay_predictor

Generated 2026-10-15 19:43 from the model registry. All training data is synthetic.

| Field | Value |
|---|---|
| Purpose | Forecasts arrival delay with a conformal 80% interval, feeding alerts, alternatives and the planner. |
| Intended use | Passenger alerts and itinerary risk. Not for operational dispatching. |
| Owner | operations-ai |
| Risk tier | medium |
| Autonomous actions allowed | retrain, promote, rollback, quarantine_feed, tighten_guardrail, scale_out, notify |
| Primary metric | mae (floor 25.0) |
| Champion version | v1, created 2026-10-15 19:42 |
| Training rows | 8000 |
| Data fingerprint | 353d6c655238 |
| Fairness check | - parity across `-` |

## Validated metrics

| Metric | Value |
|---|---|
| mae | 7.7207 |
| r2 | 0.8515 |
| interval_coverage_80 | 0.7983 |
| interval_width_min | 25.2718 |
| conformal_margin_min | 1.6327 |
| n | 3000 |

## Inputs

| Feature | Type | Allowed range | Upstream pipeline |
|---|---|---|---|
| train_type | cat | Rajdhani, Duronto, Shatabdi, Vande Bharat, Garib Rath, Superfast, Express | timetable |
| distance_km | num | 10 to 5000 | timetable |
| num_stops | num | 0 to 300 | timetable |
| departure_hour | num | 0 to 23 | timetable |
| month | num | 1 to 12 | calendar |
| day_of_week | num | 0 to 6 | calendar |
| northern_route | num | 0 to 1 | timetable |
| fog_index | num | 0 to 1 | weather_feed |
| rain_index | num | 0 to 1 | weather_feed |
| route_congestion | num | 0 to 1 | ntes_running_status |
| upstream_delay_min | num | 0 to 1440 | ntes_running_status |
| maintenance_block | num | 0 to 1 | engineering_blocks |

## Lineage

| Version | Status | Parent | Notes |
|---|---|---|---|
| v1 | champion | - | bootstrap: trained on synthetic historical data |

## Limitations

Synthetic running-status and weather inputs; station congestion is a heuristic index.
