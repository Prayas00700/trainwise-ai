"""Train every model on synthetic data and register it (the API and dashboard also do this on first start).

Usage:  python scripts/bootstrap.py [--fresh] [--scale small]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainwise.platform import Platform  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true", help="delete existing artifacts and retrain")
    ap.add_argument("--scale", default="full", choices=["full", "small"])
    args = ap.parse_args()
    p = Platform(scale=args.scale, fresh=args.fresh)
    rep = p.bootstrap_report
    print(f"Bootstrap {rep['status']}" + (f" in {rep['seconds']} s" if "seconds" in rep else " (models already registered)"))
    for row in p.registry.table():
        print(f"  {row['model']:<24} {row['version']:<4} {row['status']:<10} {row['metrics']}")
    print(f"Artifacts: {p.artifact_dir}")


if __name__ == "__main__":
    main()
