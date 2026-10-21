"""CLI entry point.

  python -m ciq run            # full pipeline: generate -> ingest -> tag -> transform -> test -> train -> mmm -> report
  python -m ciq <step>         # run a single step
  python -m ciq ask "question" # AI analyst
"""
from __future__ import annotations

import sys
import time

from . import analyst, db, generate, ingest, mmm, model, quality, report, tagging, transform

STEPS = [
    ("generate", "Generate synthetic raw exports", generate.main),
    ("ingest", "Land raw data in the warehouse", ingest.main),
    ("tag", "AI content tagging", tagging.main),
    ("transform", "Build SQL staging + marts", transform.main),
    ("test", "Data quality tests", quality.main),
    ("train", "Train breakout prediction model", model.main),
    ("mmm", "Fit marketing mix model", mmm.main),
    ("report", "Render dashboard", report.main),
]


def run_all():
    t0 = time.perf_counter()
    print(f"Content Intelligence Platform | engine: {db.ENGINE}\n")
    for i, (_, desc, fn) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {desc}")
        fn()
        print()
    print(f"pipeline finished in {time.perf_counter() - t0:.1f}s -> open docs/index.html")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        print("steps: " + ", ".join(s[0] for s in STEPS))
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        run_all()
    elif cmd == "ask":
        analyst.main(rest)
    else:
        steps = {name: fn for name, _, fn in STEPS}
        if cmd not in steps:
            sys.exit(f"unknown command: {cmd}")
        steps[cmd]()


if __name__ == "__main__":
    main()
