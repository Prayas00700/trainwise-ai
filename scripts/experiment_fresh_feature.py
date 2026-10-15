"""Experiment: does a fresh (this week's) cancellation rate fix festive-season over-prediction?

The retrained confirmation model still over-predicts confirmations in the festive regime because no
input reflects this season's collapse in cancellations. This script trains two models on the same
historical data, with and without a `recent_cancellation_rate` feature that a real-time PRS feed would
provide, and compares them on unseen festive-season tickets.

Usage:  python scripts/experiment_fresh_feature.py        (writes docs/experiments.md and two charts)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainwise.config import SEED  # noqa: E402
from trainwise.data.generators import make_confirmation_data  # noqa: E402
from trainwise.models.base import FeatureSpec  # noqa: E402
from trainwise.models.confirmation import ConfirmationPredictor  # noqa: E402


class FreshFeatureConfirmationPredictor(ConfirmationPredictor):
    name = "confirmation_predictor_fresh"
    features = ConfirmationPredictor.features + (
        FeatureSpec("recent_cancellation_rate", "num", 0, 1, pipeline="prs_bookings_realtime"),)


def reliability(model, df, bins=10):
    p = model.proba(df)["CNF"].to_numpy()
    y = (df["final_status"] == "CNF").to_numpy().astype(float)
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum() >= 20:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return xs, ys


def main() -> None:
    hist = make_confirmation_data(12000, SEED)
    normal_test = make_confirmation_data(3000, 98)
    festive = make_confirmation_data(3000, 99, regime="festival_surge")
    results = {}
    for label, cls in (("Baseline: historical rate only", ConfirmationPredictor),
                       ("With this week's cancellation rate", FreshFeatureConfirmationPredictor)):
        model = cls().fit(hist)
        results[label] = {"model": model, "normal": model.evaluate(normal_test), "festive": model.evaluate(festive)}
    actual = results[next(iter(results))]["festive"]["actual_cnf_rate"]

    labels = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, [results[k]["festive"]["accuracy"] for k in labels], color=["#9aa5b1", "#1b4f9c"])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title("Accuracy on unseen festive-season tickets")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].bar(labels, [results[k]["festive"]["mean_predicted_cnf"] for k in labels], color=["#9aa5b1", "#1b4f9c"])
    axes[1].axhline(actual, color="#c8102e", linestyle="--", label=f"actual confirmation rate {actual:.0%}")
    axes[1].set_title("Average predicted P(confirm)")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "fresh_feature_effect.png", dpi=130)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#9aa5b1", linestyle="--", label="perfect calibration")
    for (label, res), color in zip(results.items(), ("#c8102e", "#1b4f9c")):
        xs, ys = reliability(res["model"], festive)
        ax.plot(xs, ys, marker="o", color=color, label=f"{label} (ECE {res['festive']['ece_cnf']:.3f})")
    ax.set_xlabel("predicted P(confirm)")
    ax.set_ylabel("observed confirmation rate")
    ax.set_title("Calibration on festive-season tickets")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "calibration_festive.png", dpi=130)

    lines = ["# Experiment: a fresh cancellation-rate feature", "",
             "Two confirmation models trained on the same 12,000 historical tickets, evaluated on 3,000 unseen "
             "normal-season and 3,000 unseen festive-season tickets. The second model adds "
             "`recent_cancellation_rate`, the cancellation rate actually observed in the last week, which a "
             "real-time PRS feed would provide.", "",
             "| Model | Accuracy (normal) | Accuracy (festive) | ECE (festive) | Mean predicted P(confirm), festive | Actual rate |",
             "|---|---|---|---|---|---|"]
    for label, res in results.items():
        lines.append(f"| {label} | {res['normal']['accuracy']:.3f} | {res['festive']['accuracy']:.3f} | "
                     f"{res['festive']['ece_cnf']:.3f} | {res['festive']['mean_predicted_cnf']:.1%} | {actual:.1%} |")
    lines += ["", "![Effect of the fresh feature](images/fresh_feature_effect.png)", "",
              "![Calibration on festive tickets](images/calibration_festive.png)", "",
              "Reading: retraining alone cannot fix a shift that no input can see. A fresh feature lets the same "
              "model family track the regime change, which is a data-engineering decision rather than a modelling one. "
              "The production model keeps the original feature set so the governance drill still demonstrates concept "
              "drift; this experiment shows what the fix would be.", ""]
    (ROOT / "docs" / "experiments.md").write_text("\n".join(lines), encoding="utf-8")
    for label, res in results.items():
        print(f"{label}: festive accuracy {res['festive']['accuracy']:.3f}, ECE {res['festive']['ece_cnf']:.3f}, "
              f"predicted {res['festive']['mean_predicted_cnf']:.1%} vs actual {actual:.1%}")


if __name__ == "__main__":
    main()
