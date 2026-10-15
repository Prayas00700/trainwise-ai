# Model card: train_recommender

Generated 2026-10-15 19:43 from the model registry. All training data is synthetic.

| Field | Value |
|---|---|
| Purpose | Versioned ranking policy over cost, duration, availability, punctuality and personal fit. |
| Intended use | Train suggestions and the planner's ranking. Exposure is monitored against the approved baseline. |
| Owner | passenger-ai |
| Risk tier | medium |
| Autonomous actions allowed | retrain, promote, rollback, quarantine_feed, tighten_guardrail, scale_out, notify |
| Primary metric | - (floor -) |
| Champion version | v1, created 2026-10-15 19:42 |
| Training rows | - |
| Data fingerprint | - |
| Fairness check | - parity across `-` |

## Ranking weights

| Component | Weight |
|---|---|
| cost | 0.26 |
| duration | 0.16 |
| availability | 0.26 |
| punctuality | 0.16 |
| personal | 0.16 |
| premium boost | 0.0 |

## Lineage

| Version | Status | Parent | Notes |
|---|---|---|---|
| v1 | champion | - | default multi-objective weights |

## Limitations

Weights are hand-set defaults or learned from simulated feedback.
