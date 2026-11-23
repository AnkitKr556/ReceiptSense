"""Categorization experiments.

E3  Rules only vs ML only vs hybrid on the same test sets, 5 random seeds.
    Test receipts are half from shops seen in training, half from new shops.
E4  The confidence gate tau: how many receipts go to the user for review, and
    how accurate the rest are. tau is chosen on the validation split.
E3b End to end: the same three methods on fields read by OCR from the
    synthetic test images (run exp_fields.py first).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from common import LEVELS, load_json, run_batch, save_json, synthetic_images

from expense_tracker.categorizer import UNKNOWN, HybridCategorizer, MLCategorizer, RuleCategorizer
from expense_tracker.synthetic import CATEGORIES, build_dataset

SEEDS = [0, 1, 2, 3, 4]
C_GRID = [1.0, 3.0, 10.0, 30.0]
TAU_GRID = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
TARGET_AUTO_ACCURACY = 0.97  # pick the smallest tau that reaches this on validation


def records(receipts):
    return [r.to_expense() for r in receipts]


def score(preds, truth, known):
    y = np.array([e.category for e in truth])
    yhat = np.array([p.category for p in preds])
    known = np.array(known)
    correct = y == yhat
    return {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, yhat, labels=CATEGORIES, average="macro", zero_division=0)),
        "accuracy_known_shops": float(correct[known].mean()),
        "accuracy_new_shops": float(correct[~known].mean()),
        "coverage": float(np.mean(yhat != UNKNOWN)),
        "per_class_f1": dict(zip(CATEGORIES, f1_score(y, yhat, labels=CATEGORIES, average=None,
                                                      zero_division=0).round(4).tolist())),
    }


def predict_all(model, data):
    return [model.predict(e.merchant, e.items) for e in data]


def choose_C(train, val, val_known):
    best = None
    for C in C_GRID:
        m = MLCategorizer(C=C).fit(train)
        f1 = score(predict_all(m, val), val, val_known)["macro_f1"]
        if best is None or f1 > best[1]:
            best = (C, f1)
    return best[0]


def tau_table(model: HybridCategorizer, data):
    """For each tau: share sent to review, accuracy of auto-saved ones, accuracy after review."""
    base = predict_all(model, data)
    y = [e.category for e in data]
    rows = []
    for tau in TAU_GRID:
        flagged = np.array([p.source == "ml" and p.confidence < tau for p in base])
        correct = np.array([p.category == t for p, t in zip(base, y)])
        auto = ~flagged
        rows.append({
            "tau": tau,
            "review_rate": float(flagged.mean()),
            "auto_accuracy": float(correct[auto].mean()) if auto.any() else 1.0,
            # flagged receipts are fixed by the user, so they end up correct
            "accuracy_after_review": float((correct | flagged).mean()),
        })
    return rows


def choose_tau(rows):
    ok = [r for r in rows if r["auto_accuracy"] >= TARGET_AUTO_ACCURACY]
    return (ok[0] if ok else rows[-1])["tau"]


def offline():
    per_seed = []
    for seed in SEEDS:
        ds = build_dataset(seed=seed)
        train, val, test = records(ds.train), records(ds.val), records(ds.test)
        C = choose_C(train, val, ds.val_is_known)
        rules = RuleCategorizer().fit(train)
        ml = MLCategorizer(C=C).fit(train)
        hybrid = HybridCategorizer(C=C).fit(train)
        tau = choose_tau(tau_table(hybrid, val))
        hybrid.tau = tau
        out = {"seed": seed, "C": C, "tau": tau}
        preds = {}
        for name, model in (("Rules only", rules), ("ML only", ml), ("Hybrid", hybrid)):
            preds[name] = predict_all(model, test)
            out[name] = score(preds[name], test, ds.test_is_known)
        hp = preds["Hybrid"]
        out["hybrid_sources"] = {s: float(np.mean([p.source == s for p in hp])) for s in ("rule", "ml")}
        out["tau_test"] = tau_table(hybrid, test)
        chosen = next(r for r in out["tau_test"] if r["tau"] == tau)
        out["hybrid_with_gate"] = chosen
        if seed == 0:
            y = [e.category for e in test]
            out["confusion_hybrid"] = confusion_matrix(y, [p.category for p in hp], labels=CATEGORIES).tolist()
            # timing of the categorizer alone
            t = time.perf_counter()
            for e in test:
                hybrid.predict(e.merchant, e.items)
            out["predict_ms_per_receipt"] = 1000 * (time.perf_counter() - t) / len(test)
            t = time.perf_counter()
            HybridCategorizer(C=C).fit(train)
            out["train_seconds"] = time.perf_counter() - t
        per_seed.append(out)
        print(f"seed {seed}: C={C} tau={tau} | " + " | ".join(
            f"{n}: acc {out[n]['accuracy']:.3f} F1 {out[n]['macro_f1']:.3f}" for n in ("Rules only", "ML only", "Hybrid")))

    summary = {}
    for name in ("Rules only", "ML only", "Hybrid"):
        summary[name] = {}
        for k in ("accuracy", "macro_f1", "accuracy_known_shops", "accuracy_new_shops", "coverage"):
            vals = [s[name][k] for s in per_seed]
            summary[name][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        summary[name]["per_class_f1"] = {c: float(np.mean([s[name]["per_class_f1"][c] for s in per_seed]))
                                         for c in CATEGORIES}
    gate = {k: {"mean": float(np.mean([s["hybrid_with_gate"][k] for s in per_seed])),
                "std": float(np.std([s["hybrid_with_gate"][k] for s in per_seed]))}
            for k in ("review_rate", "auto_accuracy", "accuracy_after_review")}
    tau_curve = [{k: float(np.mean([s["tau_test"][i][k] for s in per_seed])) for k in per_seed[0]["tau_test"][0]}
                 for i in range(len(TAU_GRID))]
    save_json("categorization.json", {
        "seeds": SEEDS, "summary": summary, "gate_at_chosen_tau": gate, "tau_curve_test_mean": tau_curve,
        "chosen": [{"seed": s["seed"], "C": s["C"], "tau": s["tau"]} for s in per_seed],
        "hybrid_sources_mean": {k: float(np.mean([s["hybrid_sources"][k] for s in per_seed])) for k in ("rule", "ml")},
        "confusion_hybrid_seed0": per_seed[0]["confusion_hybrid"], "labels": CATEGORIES,
        "predict_ms_per_receipt": per_seed[0]["predict_ms_per_receipt"],
        "train_seconds": per_seed[0]["train_seconds"], "per_seed": per_seed,
    })


def end_to_end():
    """Categorize from OCR output of the synthetic test images (dataset seed 0)."""
    cat = load_json("categorization.json")
    C, tau = cat["chosen"][0]["C"], cat["chosen"][0]["tau"]
    ds = build_dataset(seed=0)
    train = records(ds.train)
    models = {"Rules only": RuleCategorizer().fit(train), "ML only": MLCategorizer(C=C).fit(train),
              "Hybrid": HybridCategorizer(C=C, tau=tau).fit(train)}
    from exp_fields import chosen_setting
    cfg, psm = chosen_setting()
    items = synthetic_images("test", 80)
    rows = run_batch(items, cfg, psm, tag="test_syn")
    truth = {(i["id"], i["level"]): i["truth"] for i in items}
    out = {}
    for level in LEVELS + ["text (no OCR)"]:
        out[level] = {}
        for name, model in models.items():
            correct, known = [], []
            for r in rows:
                if level != "text (no OCR)" and r["level"] != level:
                    continue
                if level == "text (no OCR)" and r["level"] != "clean":
                    continue
                t = truth[(r["id"], r["level"])]
                if level == "text (no OCR)":
                    merchant, its = t["merchant"], t["items"]
                else:
                    merchant, its = r["fields"]["merchant"] or "", r["fields"]["items"]
                p = model.predict(merchant, its)
                correct.append(p.category == t["category"])
                known.append(t["known_merchant"])
            correct, known = np.array(correct), np.array(known)
            out[level][name] = {"accuracy": float(correct.mean()), "known": float(correct[known].mean()),
                                "new": float(correct[~known].mean()), "n": int(len(correct))}
        print(level, {n: round(v["accuracy"], 3) for n, v in out[level].items()})
    save_json("categorization_end_to_end.json", out)


if __name__ == "__main__":
    offline()
    end_to_end()
