"""Evaluate the support assistant on a labelled question set in English, Hindi and Hinglish.

Usage:  python scripts/evaluate_support.py [--scale small] [--out docs/support_eval.md]
The questions in data/eval/support_eval.json are written differently from the training templates,
so the numbers are a regression harness, not a claim about real passenger traffic.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainwise.platform import Platform  # noqa: E402


def evaluate(platform, items: list) -> dict:
    rows = []
    for it in items:
        t0 = time.perf_counter()
        r = platform.support.chat(it["text"], session_id="eval", synthetic=True)
        rows.append({**it, "predicted": r["intent"], "confidence": r["confidence"], "correct": r["intent"] == it["intent"],
                     "groundedness": r["guard"]["groundedness"], "flagged": r["guard"]["flagged"],
                     "escalated": r["escalated"], "reply_source": r["reply_source"],
                     "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
    by_lang = defaultdict(list)
    for row in rows:
        by_lang[row["lang"]].append(row["correct"])
    errors = Counter((row["intent"], row["predicted"]) for row in rows if not row["correct"])
    lat = sorted(row["latency_ms"] for row in rows)
    return {"n": len(rows), "intent_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
            "by_language": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_lang.items())},
            "groundedness_pass_rate": round(sum(not r["flagged"] for r in rows) / len(rows), 4),
            "escalation_rate": round(sum(r["escalated"] for r in rows) / len(rows), 4),
            "reply_sources": dict(Counter(r["reply_source"] for r in rows)),
            "p50_latency_ms": lat[len(lat) // 2], "p95_latency_ms": lat[int(0.95 * (len(lat) - 1))],
            "top_confusions": [{"expected": e, "predicted": p, "count": c} for (e, p), c in errors.most_common(5)],
            "mode": platform.support.mode, "rows": rows}


def to_markdown(res: dict) -> str:
    lines = ["# Support assistant evaluation", "",
             f"{res['n']} labelled questions (English, Hindi, Hinglish), assistant in `{res['mode']}` mode. "
             "Questions are paraphrased differently from the training templates; this is a regression harness, "
             "not a measurement of real passenger traffic.", "",
             "| Metric | Value |", "|---|---|",
             f"| Intent accuracy | {res['intent_accuracy']:.1%} |"]
    lines += [f"| Intent accuracy ({k}) | {v:.1%} |" for k, v in res["by_language"].items()]
    lines += [f"| Groundedness pass rate | {res['groundedness_pass_rate']:.1%} |",
              f"| Escalated to a person | {res['escalation_rate']:.1%} |",
              f"| p50 / p95 latency | {res['p50_latency_ms']} ms / {res['p95_latency_ms']} ms |",
              f"| Reply sources | {', '.join(f'{k}: {v}' for k, v in res['reply_sources'].items())} |", ""]
    if res["top_confusions"]:
        lines += ["Most common confusions:", ""] + [f"- expected `{c['expected']}`, predicted `{c['predicted']}` ({c['count']})"
                                                   for c in res["top_confusions"]] + [""]
    lines += ["<details><summary>Every question</summary>", "", "| Lang | Question | Expected | Predicted | OK |", "|---|---|---|---|---|"]
    lines += [f"| {r['lang']} | {r['text']} | {r['intent']} | {r['predicted']} | {'yes' if r['correct'] else 'no'} |" for r in res["rows"]]
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", default="full", choices=["full", "small"])
    ap.add_argument("--out", default=str(ROOT / "docs" / "support_eval.md"))
    args = ap.parse_args()
    items = json.loads((ROOT / "data" / "eval" / "support_eval.json").read_text(encoding="utf-8"))
    res = evaluate(Platform(scale=args.scale), items)
    Path(args.out).write_text(to_markdown(res), encoding="utf-8")
    print(f"{res['n']} questions: intent accuracy {res['intent_accuracy']:.1%} "
          f"({', '.join(f'{k} {v:.0%}' for k, v in res['by_language'].items())}), "
          f"groundedness pass {res['groundedness_pass_rate']:.0%}, escalated {res['escalation_rate']:.0%}")
    for c in res["top_confusions"]:
        print(f"  confusion: {c['expected']} -> {c['predicted']} x{c['count']}")
    print(f"Report: {args.out}")


if __name__ == "__main__":
    main()
