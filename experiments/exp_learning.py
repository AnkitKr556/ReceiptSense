"""E5: does the system get better as one user keeps correcting it?

Cold start: the app ships with a model trained on only 40 receipts (5 per
category) from shops the user never visits. One simulated user then scans 600
receipts from 60 shops they visit, some often and some rarely (Zipf weights).

For every receipt we record whether the first suggestion was right. The user
corrects every wrong suggestion and confirms every flagged one; both count as
feedback. Methods that learn use that feedback.
"""
from __future__ import annotations

import numpy as np

from common import load_json, save_json

from expense_tracker.categorizer import HybridCategorizer, MLCategorizer, RuleCategorizer
from expense_tracker.synthetic import CATEGORIES, build_dataset, make_receipt

SEEDS = [0, 1, 2, 3, 4]
N_STREAM = 600
N_SHOPS = 60
SEED_PER_CATEGORY = 5
WINDOW = 50


def make_stream(seed):
    ds = build_dataset(seed=seed)
    rng = np.random.default_rng(1000 + seed)
    seed_set = []
    for cat in CATEGORIES:
        pool = [r for r in ds.train if r.merchant.category == cat]
        seed_set += [pool[i].to_expense() for i in rng.choice(len(pool), SEED_PER_CATEGORY, replace=False)]
    shops_pool = ds.merchants["val_new"] + ds.merchants["test_new"]
    shops = [shops_pool[i] for i in rng.choice(len(shops_pool), N_SHOPS, replace=False)]
    weights = 1.0 / np.arange(1, N_SHOPS + 1)
    weights /= weights.sum()
    stream = [make_receipt(rng, shops[rng.choice(N_SHOPS, p=weights)]).to_expense() for _ in range(N_STREAM)]
    return seed_set, stream


class MLWithRetraining:
    """ML only, retrained on feedback, but with no memory of shops."""

    def __init__(self, C, every):
        self.model, self.every, self.data, self.pending = MLCategorizer(C=C), every, [], 0

    def fit(self, data):
        self.data = list(data)
        self.model.fit(self.data)
        return self

    def predict(self, merchant, items):
        return self.model.predict(merchant, items)

    def feedback(self, merchant, items, category):
        from expense_tracker.models import Expense
        self.data.append(Expense(merchant, list(items), category))
        self.pending += 1
        if self.pending >= self.every:
            self.model.fit(self.data)
            self.pending = 0


class RulesWithMemory:
    """Rules baseline that also remembers corrected shops."""

    def __init__(self):
        self.model = RuleCategorizer()

    def fit(self, data):
        self.model.fit(data)
        return self

    def predict(self, merchant, items):
        return self.model.predict(merchant, items)

    def feedback(self, merchant, items, category):
        self.model.merchants.add(merchant, category)


def simulate(model, stream, learns: bool):
    """Returns per-receipt correctness, a first-visit mask, and counts of user actions."""
    first_correct, first_visit, seen, corrections, confirmations = [], [], set(), 0, 0
    for e in stream:
        p = model.predict(e.merchant, e.items)
        ok = p.category == e.category
        first_correct.append(ok)
        first_visit.append(e.merchant not in seen)
        seen.add(e.merchant)
        if not ok:
            corrections += 1
        elif p.needs_review:
            confirmations += 1
        if learns and (not ok or p.needs_review):
            model.feedback(e.merchant, e.items, e.category)
    return np.array(first_correct), np.array(first_visit), corrections, confirmations


def main():
    cat = load_json("categorization.json")
    C = cat["chosen"][0]["C"]
    tau = cat["chosen"][0]["tau"]
    methods = {
        "Static ML (never learns)": lambda: (MLCategorizer(C=C), False),
        "ML retrained every 10 (no shop memory)": lambda: (MLWithRetraining(C, 10), True),
        "Rules + remembered shops": lambda: (RulesWithMemory(), True),
        "Hybrid, ML frozen (shop memory only)": lambda: (HybridCategorizer(C=C, tau=tau, retrain_every=0), True),
        "Hybrid, retrain every 10 (proposed)": lambda: (HybridCategorizer(C=C, tau=tau, retrain_every=10), True),
    }
    results = {name: {"curves": [], "corrections": [], "confirmations": [], "last200": [], "first100": [],
                      "first_visit": [], "first_visit_late": []} for name in methods}
    for seed in SEEDS:
        seed_set, stream = make_stream(seed)
        for name, make in methods.items():
            model, learns = make()
            model.fit(seed_set)
            correct, first, corr, conf = simulate(model, stream, learns)
            rolling = np.convolve(correct, np.ones(WINDOW) / WINDOW, mode="valid")
            r = results[name]
            r["curves"].append(rolling.tolist())
            r["corrections"].append(corr)
            r["confirmations"].append(conf)
            r["first100"].append(float(correct[:100].mean()))
            r["last200"].append(float(correct[-200:].mean()))
            r["first_visit"].append(float(correct[first].mean()))
            late = first.copy()
            late[:N_STREAM // 2] = False  # new shops met in the second half of the stream
            r["first_visit_late"].append(float(correct[late].mean()) if late.any() else float("nan"))
        print(f"seed {seed}: " + " | ".join(f"{n}: last200 {results[n]['last200'][-1]:.3f}" for n in methods))

    summary = {}
    for name, r in results.items():
        summary[name] = {
            "first100_accuracy": float(np.mean(r["first100"])),
            "last200_accuracy": float(np.mean(r["last200"])), "last200_std": float(np.std(r["last200"])),
            "corrections": float(np.mean(r["corrections"])), "confirmations": float(np.mean(r["confirmations"])),
            "first_visit_accuracy": float(np.mean(r["first_visit"])),
            "first_visit_accuracy_second_half": float(np.nanmean(r["first_visit_late"])),
            "mean_curve": np.mean(np.array(r["curves"]), axis=0).round(4).tolist(),
        }
    save_json("learning.json", {"window": WINDOW, "n_stream": N_STREAM, "n_shops": N_SHOPS,
                                "seed_set_size": SEED_PER_CATEGORY * len(CATEGORIES), "tau": tau, "C": C,
                                "summary": summary})


if __name__ == "__main__":
    main()
