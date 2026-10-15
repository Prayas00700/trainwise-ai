"""Generate a model card for every registered model: the AI inventory an auditor can read.

Usage:  python scripts/model_cards.py [--out docs/model_cards]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainwise.config import model_policy  # noqa: E402
from trainwise.platform import Platform  # noqa: E402

TEXT = {
    "confirmation_predictor": {
        "purpose": "Predicts whether a waitlisted or RAC ticket will be confirmed, RAC or still waitlisted at charting.",
        "use": "Passenger guidance and journey planning. Not a guarantee of travel.",
        "limits": "Trained on synthetic bookings; no real-time cancellation feed, so seasonal collapses in cancellations are learned only after retraining (see docs/experiments.md).",
    },
    "delay_predictor": {
        "purpose": "Forecasts arrival delay with a conformal 80% interval, feeding alerts, alternatives and the planner.",
        "use": "Passenger alerts and itinerary risk. Not for operational dispatching.",
        "limits": "Synthetic running-status and weather inputs; station congestion is a heuristic index.",
    },
    "fare_demand_forecaster": {
        "purpose": "Forecasts the share of seats sold by day; a deterministic tariff converts it to fare and booking-window advice.",
        "use": "Fare forecasts and savings suggestions. Tariff parameters are illustrative, not official.",
        "limits": "Demand curves are synthetic; no real inventory feed.",
    },
    "fraud_detector": {
        "purpose": "Scores booking sessions for bots, touts, account takeover and card testing, with reason codes.",
        "use": "Step-up verification or blocking at booking. High-risk: promotions need human approval.",
        "limits": "Synthetic attack profiles are more separable than real ones; thresholds are cost-based on policy values.",
    },
    "intent_classifier": {
        "purpose": "Routes support messages in English, Hindi and Hinglish to the right handler.",
        "use": "Internal routing for the support assistant.",
        "limits": "Trained on about 600 templated utterances; see docs/support_eval.md for a held-out check.",
    },
    "train_recommender": {
        "purpose": "Versioned ranking policy over cost, duration, availability, punctuality and personal fit.",
        "use": "Train suggestions and the planner's ranking. Exposure is monitored against the approved baseline.",
        "limits": "Weights are hand-set defaults or learned from simulated feedback.",
    },
}


def card(platform, name: str) -> str:
    reg, pol = platform.registry, model_policy(name)
    champion = reg.champion(name)
    info = reg.info(name, champion)
    obj = reg.model(name, champion)
    text = TEXT.get(name, {"purpose": "", "use": "", "limits": ""})
    lines = [f"# Model card: {name}", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the model registry. All training data is synthetic.", "",
             "| Field | Value |", "|---|---|",
             f"| Purpose | {text['purpose']} |", f"| Intended use | {text['use']} |",
             f"| Owner | {pol.get('owner', '-')} |", f"| Risk tier | {pol.get('risk_tier', '-')} |",
             f"| Autonomous actions allowed | {', '.join(pol.get('autonomous_actions', []))} |",
             f"| Primary metric | {pol.get('primary_metric', '-')} (floor {pol.get('floor', '-')}) |",
             f"| Champion version | {champion}, created {time.strftime('%Y-%m-%d %H:%M', time.localtime(info['created_at']))} |",
             f"| Training rows | {info.get('trained_rows') or '-'} |", f"| Data fingerprint | {info.get('data_fingerprint') or '-'} |",
             f"| Fairness check | {pol.get('fairness_metric', '-')} parity across `{pol.get('fairness_segment', '-')}` |", ""]
    if info.get("metrics"):
        lines += ["## Validated metrics", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in info["metrics"].items()]
        lines.append("")
    if hasattr(obj, "features") and obj.features:
        lines += ["## Inputs", "", "| Feature | Type | Allowed range | Upstream pipeline |", "|---|---|---|---|"]
        for f in obj.features:
            rng = f"{f.lo} to {f.hi}" if f.kind == "num" else (", ".join(f.categories) if f.categories else "text")
            lines.append(f"| {f.name} | {f.kind} | {rng} | {f.pipeline} |")
        lines.append("")
    if hasattr(obj, "weights"):
        lines += ["## Ranking weights", "", "| Component | Weight |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in obj.weights.items()]
        lines += [f"| premium boost | {getattr(obj, 'premium_boost', 0)} |", ""]
    lines += ["## Lineage", "", "| Version | Status | Parent | Notes |", "|---|---|---|---|"]
    lines += [f"| {v} | {m['status']} | {m.get('parent') or '-'} | {m.get('notes', '')} |" for v, m in reg.versions(name).items()]
    lines += ["", "## Limitations", "", text["limits"], ""]
    return "\n".join(lines)


def write_model_cards(platform, out_dir) -> list:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in platform.registry.index:
        path = out_dir / f"{name}.md"
        path.write_text(card(platform, name), encoding="utf-8")
        written.append(path)
    index = ["# Model cards", "", "One card per registered model, generated from the registry by `scripts/model_cards.py`.", ""]
    index += [f"- [{p.stem}]({p.name})" for p in written]
    (out_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "model_cards"))
    ap.add_argument("--scale", default="full", choices=["full", "small"])
    args = ap.parse_args()
    files = write_model_cards(Platform(scale=args.scale), args.out)
    print(f"Wrote {len(files)} model cards to {args.out}")


if __name__ == "__main__":
    main()
