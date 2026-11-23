"""Download a sample of real scanned receipts from the public SROIE dataset.

SROIE (ICDAR 2019, Huang et al.) has 626 labelled receipts with the company name,
date, address and total. We use it only to test field extraction on real scans.

Split used in this project (fixed, so results are reproducible):
  dev  = receipts 000-049, used while writing and tuning the parser rules
  test = 100 receipts sampled with seed 7 from 050-625, used only for the final numbers

Usage:  python data/download_sroie.py
Images are not included in the project zip; run this script to fetch them.
"""
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data"
OUT = Path(__file__).resolve().parent / "sroie"


def split_ids():
    dev = [f"{i:03d}" for i in range(50)]
    rest = [f"{i:03d}" for i in range(50, 626)]
    test = sorted(random.Random(7).sample(rest, 100))
    return dev, test


def fetch(rid: str) -> str:
    img_path = OUT / "img" / f"{rid}.jpg"
    key_path = OUT / "key" / f"{rid}.json"
    if img_path.exists() and key_path.exists():
        return rid
    for url, path in ((f"{BASE}/img/{rid}.jpg", img_path), (f"{BASE}/key/{rid}.json", key_path)):
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return rid


def main():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    (OUT / "key").mkdir(parents=True, exist_ok=True)
    dev, test = split_ids()
    with ThreadPoolExecutor(8) as pool:
        done = list(pool.map(fetch, dev + test))
    (OUT / "split.json").write_text(json.dumps({"dev": dev, "test": test}, indent=1))
    print(f"downloaded {len(done)} receipts to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
