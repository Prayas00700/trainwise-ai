# Model card: intent_classifier

Generated 2026-10-15 19:43 from the model registry. All training data is synthetic.

| Field | Value |
|---|---|
| Purpose | Routes support messages in English, Hindi and Hinglish to the right handler. |
| Intended use | Internal routing for the support assistant. |
| Owner | support-ai |
| Risk tier | medium |
| Autonomous actions allowed | retrain, promote, rollback, quarantine_feed, tighten_guardrail, scale_out, notify |
| Primary metric | accuracy (floor 0.85) |
| Champion version | v1, created 2026-10-15 19:42 |
| Training rows | 222 |
| Data fingerprint | a17326b460a6 |
| Fairness check | - parity across `-` |

## Validated metrics

| Metric | Value |
|---|---|
| accuracy | 0.9107 |
| macro_f1 | 0.8141 |
| n | 56 |

## Inputs

| Feature | Type | Allowed range | Upstream pipeline |
|---|---|---|---|
| text | text | text | chat_channel |

## Lineage

| Version | Status | Parent | Notes |
|---|---|---|---|
| v1 | champion | - | bootstrap: trained on synthetic historical data |

## Limitations

Trained on about 600 templated utterances; see docs/support_eval.md for a held-out check.
