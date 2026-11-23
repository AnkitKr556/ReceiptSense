"""Shared helpers for the experiments: datasets, image sets and a cached batch runner."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zlib
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one Tesseract thread per worker process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from expense_tracker.evaluate import field_scores  # noqa: E402
from expense_tracker.pipeline import ReceiptPipeline  # noqa: E402
from expense_tracker.preprocess import RAW, PreprocessConfig  # noqa: E402
from expense_tracker.synthetic import build_dataset, render_receipt  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
FIGURES = RESULTS / "figures"
for d in (RESULTS, CACHE, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 2))
LEVELS = ["clean", "medium", "hard"]
DATASET_SEED = 0

# Preprocessing variants compared in the ablation. "full" is the default pipeline.
CONFIGS = {
    "raw (no preprocessing)": RAW,
    "full pipeline": PreprocessConfig(),
    "without upscale": PreprocessConfig(upscale=False),
    "without light flattening": PreprocessConfig(flatten_light=False),
    "without deskew": PreprocessConfig(deskew=False),
    "without denoise": PreprocessConfig(denoise="never"),
    "denoise every image": PreprocessConfig(denoise="always"),
    "plus Otsu binarization": PreprocessConfig(binarize="otsu"),
    "plus adaptive binarization": PreprocessConfig(binarize="adaptive"),
}
DEFAULT_CONFIG = "full pipeline"


def dataset():
    return build_dataset(seed=DATASET_SEED)


# ------------------------------------------------------------------ image sets

def synthetic_images(split: str, n_receipts: int) -> list[dict]:
    """Render `n_receipts` receipts from the val or test split at every difficulty level.

    Half the receipts come from merchants seen in training and half from new ones.
    The same receipts are rendered at all three levels, so levels compare like for like.
    """
    out_dir = DATA / "synthetic" / split
    index_file = out_dir / "index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
        if len(index) == n_receipts * len(LEVELS):
            return index
    ds = dataset()
    receipts = ds.val if split == "val" else ds.test
    known = ds.val_is_known if split == "val" else ds.test_is_known
    rng = np.random.default_rng(100 if split == "val" else 200)
    k_idx = [i for i, k in enumerate(known) if k]
    n_idx = [i for i, k in enumerate(known) if not k]
    chosen = sorted(rng.choice(k_idx, n_receipts // 2, replace=False).tolist()
                    + rng.choice(n_idx, n_receipts - n_receipts // 2, replace=False).tolist())
    index = []
    for level in LEVELS:
        (out_dir / level).mkdir(parents=True, exist_ok=True)
        for i in chosen:
            r = receipts[i]
            path = out_dir / level / f"{split}_{i:04d}.jpg"
            img = render_receipt(r, np.random.default_rng(zlib.crc32(f"{split}-{level}-{i}".encode())), level)
            cv2.imwrite(str(path), img)
            truth = r.truth()
            truth["known_merchant"] = bool(known[i])
            index.append({"id": f"{split}_{i:04d}", "level": level, "path": str(path.relative_to(ROOT)),
                          "truth": truth})
    index_file.write_text(json.dumps(index, indent=1))
    return index


def sroie_images(split: str) -> list[dict]:
    d = DATA / "sroie"
    if not (d / "split.json").exists():
        raise SystemExit("SROIE data not found. Run: python data/download_sroie.py")
    ids = json.loads((d / "split.json").read_text())[split]
    out = []
    for rid in ids:
        key = json.loads((d / "key" / f"{rid}.json").read_text())
        out.append({"id": f"sroie_{rid}", "level": "sroie", "path": str((d / "img" / f"{rid}.jpg").relative_to(ROOT)),
                    "truth": {"merchant": key.get("company"), "date": key.get("date"), "total": key.get("total")}})
    return out


# ------------------------------------------------------------------ batch runner

def _job(args):
    item, cfg_name, psm = args
    pipe = ReceiptPipeline(categorizer=None, config=CONFIGS[cfg_name], psm=psm)
    res = pipe.process_file(ROOT / item["path"])
    scores = field_scores(res.fields, item["truth"])
    return {
        "id": item["id"], "level": item["level"], "config": cfg_name, "psm": psm,
        "fields": asdict(res.fields), "scores": scores, "timings": res.timings,
        "skew": res.info["skew"], "ocr_text": [l.text for l in res.ocr_lines],
        "mean_conf": float(np.mean([l.conf for l in res.ocr_lines])) if res.ocr_lines else 0.0,
    }


def run_batch(items: list[dict], cfg_name: str, psm: int = 4, tag: str = "") -> list[dict]:
    """Run the pipeline (without categorizer) over images, caching results on disk."""
    key = hashlib.md5(json.dumps([i["id"] + i["level"] for i in items] + [cfg_name, psm]).encode()).hexdigest()[:10]
    cache_file = CACHE / f"{tag}_{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    with Pool(WORKERS) as pool:
        out = pool.map(_job, [(i, cfg_name, psm) for i in items], chunksize=2)
    cache_file.write_text(json.dumps(out))
    return out


def summarize_fields(rows: list[dict]) -> dict:
    keys = ["merchant_exact", "merchant_fuzzy", "date", "total", "all_three"]
    out = {k: round(100 * float(np.mean([r["scores"][k] for r in rows])), 1) for k in keys}
    if rows and "item_recall" in rows[0]["scores"]:
        out["item_recall"] = round(100 * float(np.mean([r["scores"]["item_recall"] for r in rows])), 1)
    out["n"] = len(rows)
    return out


def save_json(name: str, obj) -> None:
    (RESULTS / name).write_text(json.dumps(obj, indent=2))


def load_json(name: str):
    return json.loads((RESULTS / name).read_text())
