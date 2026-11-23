"""Field extraction experiments.

E1 Accuracy of merchant, date and total on
   - 240 synthetic test images (80 receipts x clean / medium / hard), and
   - 100 real scanned receipts from the SROIE test sample.
E2 Ablation: each preprocessing step switched off in turn, on the same images.
E6 Latency per stage.
"""
from __future__ import annotations

import numpy as np

from common import (CONFIGS, DEFAULT_CONFIG, LEVELS, load_json, run_batch, save_json, sroie_images,
                    summarize_fields, synthetic_images)


def chosen_setting():
    try:
        chosen = load_json("dev_tuning.json")["chosen"]
        return chosen["config"], chosen["psm"]
    except FileNotFoundError:
        return DEFAULT_CONFIG, 4


def main():
    cfg, psm = chosen_setting()
    print(f"using '{cfg}', psm {psm} (chosen on dev)")
    syn = synthetic_images("test", 80)
    sroie = sroie_images("test")

    # E1 main results
    syn_rows = run_batch(syn, cfg, psm, tag="test_syn")
    sroie_rows = run_batch(sroie, cfg, psm, tag="test_sroie")
    main_table = {level: summarize_fields([r for r in syn_rows if r["level"] == level]) for level in LEVELS}
    main_table["synthetic (all)"] = summarize_fields(syn_rows)
    main_table["SROIE real scans"] = summarize_fields(sroie_rows)
    for k, v in main_table.items():
        print(f"{k:18s} {v}")

    # skew estimation error on synthetic images is not known exactly (random angle is not stored),
    # so we report the distribution of estimated angles per level instead
    skew = {level: {"mean_abs_deg": float(np.mean([abs(r["skew"]) for r in syn_rows if r["level"] == level]))}
            for level in LEVELS}

    # E2 ablation
    ablation = []
    for name in CONFIGS:
        s = run_batch(syn, name, psm, tag="test_syn")
        r = run_batch(sroie, name, psm, tag="test_sroie")
        row = {"config": name, "synthetic": summarize_fields(s), "sroie": summarize_fields(r)}
        for level in LEVELS:
            row[level] = summarize_fields([x for x in s if x["level"] == level])
        ablation.append(row)
        print(f"ablation {name:28s} synthetic {row['synthetic']['all_three']:5.1f}  "
              f"hard {row['hard']['all_three']:5.1f}  SROIE {row['sroie']['all_three']:5.1f}")

    # E6 latency
    latency = {}
    for label, rows in (("synthetic", syn_rows), ("SROIE", sroie_rows)):
        latency[label] = {}
        for stage in ("preprocess", "ocr", "parse", "total"):
            v = np.array([r["timings"][stage] for r in rows]) * 1000
            latency[label][stage] = {"mean_ms": float(v.mean()), "median_ms": float(np.median(v)),
                                     "p95_ms": float(np.percentile(v, 95)), "max_ms": float(v.max())}

    # per-receipt rows for anyone who wants to look at individual failures
    per_receipt = [{"id": r["id"], "level": r["level"], **r["scores"], "pred": r["fields"]} for r in syn_rows + sroie_rows]
    save_json("fields.json", {"setting": {"config": cfg, "psm": psm}, "main": main_table, "skew": skew,
                              "ablation": ablation, "latency": latency})
    save_json("fields_per_receipt.json", per_receipt)


if __name__ == "__main__":
    main()
