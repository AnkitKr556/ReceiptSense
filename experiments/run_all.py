"""Run every experiment from scratch and write results/summary.md.

    python experiments/run_all.py            # everything (about 40 minutes on 2 cores)
    python experiments/run_all.py --keep     # reuse cached OCR runs

Needs the SROIE sample: python data/download_sroie.py
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time

from common import CACHE, DATA, RESULTS, ROOT


def step(name, fn):
    t = time.time()
    print(f"\n=== {name}", flush=True)
    fn()
    print(f"=== {name} done in {time.time() - t:.0f}s", flush=True)


def run_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--cov", "--cov-report=term-missing"],
                         cwd=ROOT, capture_output=True, text=True)
    (RESULTS / "test_report.txt").write_text(out.stdout + out.stderr)
    summary = next((l for l in reversed(out.stdout.splitlines()) if re.search(r"\d+ (passed|failed)", l)), "?")
    total = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+%)", out.stdout, re.M)
    (RESULTS / "tests.json").write_text(json.dumps({"summary": summary.strip("= "),
                                                    "coverage": total.group(1) if total else "?"}))
    print(summary)


def main():
    if "--keep" not in sys.argv:
        shutil.rmtree(CACHE, ignore_errors=True)
        shutil.rmtree(DATA / "synthetic", ignore_errors=True)
        CACHE.mkdir(parents=True)
    import exp_categorize
    import exp_fields
    import exp_learning
    import make_report
    import tune_on_dev
    step("dev tuning", tune_on_dev.main)
    step("field extraction + ablation + latency", exp_fields.main)
    step("categorization (5 seeds)", exp_categorize.offline)
    step("categorization end to end", exp_categorize.end_to_end)
    step("learning loop", exp_learning.main)
    step("tests", run_tests)
    step("report", make_report.main)


if __name__ == "__main__":
    main()
