"""Choose the preprocessing settings and Tesseract page mode on the dev sets only.

Dev sets: 50 SROIE dev receipts (000-049) and 40 synthetic validation receipts
rendered at 3 difficulty levels (120 images). Test images are not touched here.
"""
from common import CONFIGS, run_batch, save_json, sroie_images, summarize_fields, synthetic_images


def main():
    syn = synthetic_images("val", 40)
    sroie = sroie_images("dev")
    rows = []
    for psm in (4, 6):
        for name in CONFIGS:
            if psm == 6 and name not in ("full pipeline", "raw (no preprocessing)"):
                continue
            s = summarize_fields(run_batch(syn, name, psm, tag="dev_syn"))
            r = summarize_fields(run_batch(sroie, name, psm, tag="dev_sroie"))
            combined = round((s["all_three"] + r["all_three"]) / 2, 1)
            rows.append({"config": name, "psm": psm, "synthetic_dev": s, "sroie_dev": r,
                         "mean_all_three": combined})
            print(f"psm {psm} | {name:28s} | synthetic all-three {s['all_three']:5.1f} "
                  f"| SROIE all-three {r['all_three']:5.1f} | mean {combined}")
    best = max(rows, key=lambda r: r["mean_all_three"])
    print("best on dev:", best["config"], "psm", best["psm"])
    save_json("dev_tuning.json", {"rows": rows, "chosen": {"config": best["config"], "psm": best["psm"]}})


if __name__ == "__main__":
    main()
