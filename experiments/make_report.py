"""Turn the JSON results into figures (results/figures/*.png) and results/summary.md."""
from __future__ import annotations

import platform
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import FIGURES, LEVELS, RESULTS, load_json  # noqa: E402

# Colors: categorical slots 1-5 and a one-hue blue ramp for the ordered difficulty levels.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
RAMP = {"clean": "#86b6ef", "medium": "#2a78d6", "hard": "#184f95"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})


def _bar_labels(ax, bars, fmt="{:.0f}", inside=False):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK2)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ figures

def fig_fields(f):
    fields = [("merchant_fuzzy", "Merchant"), ("date", "Date"), ("total", "Total"), ("all_three", "All three")]
    groups = [(lv, f"Synthetic {lv}", RAMP[lv]) for lv in LEVELS] + [("SROIE real scans", "SROIE real scans", ORANGE)]
    x = np.arange(len(fields))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (key, label, color) in enumerate(groups):
        vals = [f["main"][key][k] for k, _ in fields]
        bars = ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=color, label=label)
        _bar_labels(ax, bars)
    ax.set_xticks(x, [l for _, l in fields])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Receipts with the field right (%)")
    ax.set_title("Field extraction: strong on clean and medium photos, weaker on hard photos and real scans")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, "fields_accuracy.png")


def fig_ablation(f):
    rows = f["ablation"]
    names = [r["config"] for r in rows]
    y = np.arange(len(rows))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    b1 = ax.barh(y - h / 2, [r["synthetic"]["all_three"] for r in rows], h * 0.92, color=BLUE, label="Synthetic photos (240)")
    b2 = ax.barh(y + h / 2, [r["sroie"]["all_three"] for r in rows], h * 0.92, color=ORANGE, label="SROIE real scans (100)")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_width():.0f}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Receipts with merchant, date and total all right (%)")
    ax.set_title("Preprocessing ablation: big gains on phone photos, small on flat scans")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.13))
    _save(fig, "preprocessing_ablation.png")


def fig_categorization(c):
    methods = [("Rules only", BLUE), ("ML only", ORANGE), ("Hybrid", AQUA)]
    groups = [("accuracy_known_shops", "Shops seen in training"), ("accuracy_new_shops", "New shops"),
              ("accuracy", "All test receipts")]
    x = np.arange(len(groups))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, (m, color) in enumerate(methods):
        vals = [100 * c["summary"][m][k]["mean"] for k, _ in groups]
        errs = [100 * c["summary"][m][k]["std"] for k, _ in groups]
        bars = ax.bar(x + (i - 1) * w, vals, w * 0.9, color=color, label=m, yerr=errs,
                      error_kw={"elinewidth": 1, "ecolor": INK2, "capsize": 2})
        for b, e in zip(bars, errs):
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height() + e),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [g for _, g in groups])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Correct category (%), mean of 5 seeds")
    ax.set_title("Categorization: hybrid keeps rules' accuracy on known shops and ML's on new ones")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    _save(fig, "categorization_methods.png")


def fig_confusion(c):
    labels = c["labels"]
    m = np.array(c["confusion_hybrid_seed0"], dtype=float)
    pct = 100 * m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if pct[i, j] > 55 else INK)
    short = [l.replace(" & ", " &\n") for l in labels]
    ax.set_xticks(range(len(labels)), short, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), short, fontsize=9)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("True category")
    ax.grid(False)
    ax.set_title("Hybrid categorizer, confusion matrix (seed 0, 600 test receipts)", fontsize=11)
    _save(fig, "confusion_matrix_hybrid.png")


def fig_tau(c):
    curve = c["tau_curve_test_mean"]
    taus = [r["tau"] for r in curve]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(taus, [100 * r["review_rate"] for r in curve], color=BLUE, lw=2, marker="o", ms=4)
    ax.plot(taus, [100 * r["auto_accuracy"] for r in curve], color=ORANGE, lw=2, marker="o", ms=4)
    ax.annotate("Sent to the user for review", (taus[-1], 100 * curve[-1]["review_rate"]), xytext=(-5, 8),
                textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.annotate("Accuracy of the receipts saved automatically", (taus[0], 100 * curve[0]["auto_accuracy"]),
                xytext=(0, 8), textcoords="offset points", color=INK2, fontsize=9)
    chosen = [r["tau"] for r in c["chosen"]]
    ax.axvspan(min(chosen), max(chosen), color=GRID, alpha=0.6, lw=0)
    ax.annotate("τ chosen on validation\n(range over 5 seeds)", (np.mean(chosen), 50), ha="center", fontsize=9, color=INK2)
    ax.set_xlabel("Confidence threshold τ")
    ax.set_ylabel("% of test receipts")
    ax.set_ylim(0, 105)
    ax.set_title("Confidence gate: a higher τ asks the user more often, so fewer mistakes get saved")
    _save(fig, "confidence_threshold.png")


def fig_learning(l):
    colors = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    w = l["window"]
    for (name, s), color in zip(l["summary"].items(), colors):
        y = 100 * np.array(s["mean_curve"])
        x = np.arange(w, w + len(y))
        ax.plot(x, y, color=color, lw=2, label=name)
    ax.set_xlabel("Receipts scanned by the user")
    ax.set_ylabel(f"Correct first suggestion (%), last {w}")
    ax.set_ylim(40, 102)
    ax.set_title("Learning from corrections: cold start with 40 training receipts")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9)
    _save(fig, "learning_curve.png")


def fig_latency(f):
    stages = [("preprocess", "Preprocess", BLUE), ("ocr", "OCR", ORANGE), ("parse", "Parse (about 1 ms)", AQUA)]
    data = f["latency"]
    names = list(data)
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    left = np.zeros(len(names))
    for key, label, color in stages:
        vals = np.array([data[n][key]["mean_ms"] for n in names])
        ax.barh(names, vals, left=left, color=color, label=label, edgecolor=SURFACE, linewidth=2, height=0.55)
        for i, v in enumerate(vals):
            if v > 120:
                ax.text(left[i] + v / 2, i, f"{v:.0f} ms", ha="center", va="center", fontsize=9,
                        color="#ffffff" if color != AQUA else INK)
        left += vals
    for i, n in enumerate(names):
        ax.text(left[i] + 20, i, f"{data[n]['total']['mean_ms']:.0f} ms total", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, left.max() * 1.3)
    ax.set_xlabel("Mean time per receipt (ms), one CPU core")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the time goes: preprocessing and OCR; parsing is negligible")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38))
    _save(fig, "latency.png")


# ------------------------------------------------------------------ summary.md

ALGORITHM_FLOW = """\
## Algorithm flow

How one receipt goes from a photo to a saved, categorized expense. Each stage is one module in `src/expense_tracker/`, and `pipeline.py` chains them.

```
 receipt photo
      │
      ▼
 ┌──────────────────────────┐
 │ 1. PREPROCESS            │  preprocess.py   (OpenCV)
 │   grayscale              │
 │   upscale to ≥ 1000 px   │
 │   flatten lighting       │
 │   deskew (−6°…+6°)       │
 │   median filter if soft  │
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 2. OCR                   │  ocr.py   (Tesseract 5, LSTM, psm 4)
 │   words + boxes + conf   │
 │   → group into lines     │
 │   → merge same-row lines │
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 3. PARSE                 │  parser.py   (regex + layout rules)
 │   merchant, date, total, │
 │   items, GSTIN           │
 └────────────┬─────────────┘
              ▼
 ┌──────────────────────────┐
 │ 4. CATEGORIZE (hybrid)   │  categorizer.py
 │   shop list → ML → gate  │
 └────────────┬─────────────┘
              ▼
   category + confidence ──► needs review?  ── yes ──► user picks / corrects
                                   │                          │
                                   no                         ▼
                                   ▼                   5. LEARN (feedback)
                                 saved  ◄──────────────  update shop list,
                                                         retrain every k = 10
```

### 1. Preprocess (`preprocess.py`)

```
function PREPROCESS(img):
    g ← grayscale(img)
    sharpness ← variance(Laplacian(g))                        # measured on the original photo
    if width(g) < 1000: g ← resize(g, width = 1000)          # letters ~20–30 px tall
    bg ← morphological_close(g, kernel wider than strokes)    # estimate of the blank paper
    g ← g / bg · 255                                          # remove shadows / uneven light
    # deskew, projection profile on an Otsu-binarized copy:
    score(θ) ← Σ (diff(row_sums(rotate(bw, θ))))²             # sharp text/gap alternation
    θ* ← best θ in −6°…+6° (0.5° steps), then refine ±0.4° (0.1° steps)
    if |θ*| ≥ 0.2°: g ← rotate(g, θ*)
    if sharpness < 1500: g ← median_filter(g, 3×3)            # only soft images
    return g                                                  # no binarization (ablation, §3)
```

### 2. OCR (`ocr.py`)

```
function OCR(g):
    words ← tesseract(g, psm = 4)               # text, bounding box, confidence per word
    lines ← group words by Tesseract's (block, paragraph, line)
    sort lines by vertical centre
    join lines whose centres differ by ≤ half a line height and that don't overlap horizontally
                                                # keeps "TOTAL ...... 450.00" on one line
    return lines top to bottom
```

### 3. Parse (`parser.py`)

```
function PARSE(lines):
    fix O/D/Q→0, l/I/|→1, S→5, B→8, Z→2 only inside tokens that are mostly digits
    TOTAL:
        candidates ← lines with a total keyword, excluding Sub Total / Total Qty / Total Tax
        stop at "GST SUMMARY" tables
        prefer strong keywords (Grand Total, Net Amount, Amount Payable) over plain "Total"
        among equals take the last one before Cash / Change
        if none: largest amount that is not cash tendered or change
    DATE:
        try 7 formats (dd/mm/yyyy, dd.mm.yy, dd-Mon-yyyy, yyyy-mm-dd, ...), day first
        fall back to month first if day-first is impossible
        score each date: +2 if its line says "Date"/"Dt", +1 for a 4-digit year, +1 if year ≥ 2015
        take the highest score
    MERCHANT:
        plausible ← first 6 lines that are mostly letters and not a header, address or phone line
        return the first plausible line with a business word (Store, Medicals, Restaurant, Sdn Bhd, ...)
        else the first plausible line
    ITEMS:
        lines between merchant and total that end in a price, with price and qty removed
    return (merchant, date, total, items, gstin)
```

### 4. Categorize: hybrid (`categorizer.py`)

Rules only, ML only and hybrid implement the same `fit` / `predict` interface (Strategy pattern), so they are compared on identical data (§4).

```
function CATEGORIZE(merchant, items):
    m ← normalize(merchant)                     # lowercase, strip punctuation, Pvt/Ltd/M/s
    if m ∈ ShopList:                    return (ShopList[m], 1.0, RULE)
    if fuzzy(m, s) ≥ 0.9 and fuzzy(first_word(m), first_word(s)) ≥ 0.75 for some s ∈ ShopList:
                                        return (ShopList[s], 1.0, RULE)
    x ← [ TF-IDF word 1–2 grams + char 3–5 grams ](m)  ⊕  [ same ](items)
    (c, p) ← argmax LogisticRegression(x)       # C chosen on validation
    if p < τ:                           return (c, p, NEEDS_REVIEW)   # ask the user
    return (c, p, ML)
```

Character n-grams make the model tolerant of OCR typos (`paracetam0l`) and POS abbreviations (`PNR BTR MSL`); the first-word check stops generic names (`Sri Enterprises` / `Sai Enterprises`) from being treated as the same shop.

### 5. Learn from the user (`HybridCategorizer.feedback`)

```
function ON_FEEDBACK(merchant, items, c*):      # user corrected or confirmed a category
    ShopList[normalize(merchant)] ← c*          # takes effect on the very next receipt
    TrainSet ← TrainSet ∪ {(merchant, items, c*)}
    pending ← pending + 1
    if pending ≥ k (= 10):
        retrain LogisticRegression on TrainSet
        pending ← 0
```

The shop list handles shops the user returns to; retraining is what improves guesses at shops never seen before (§7).

### In the web app (`api.py`, `storage.py`)

```
POST  /api/receipts         image → PREPROCESS → OCR → PARSE → CATEGORIZE → save to SQLite
PATCH /api/expenses/{id}    new category → ON_FEEDBACK → save model to disk
GET   /api/summary          totals by category and by month for the dashboard
```
"""



def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def write_summary(f, c, e2e, l, dev, cov):
    s = []
    a = s.append
    a("# Results\n")
    a("Every number here was produced by `python experiments/run_all.py` on the code in this folder. "
      f"Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}, "
      f"Tesseract {subprocess.run(['tesseract', '--version'], capture_output=True, text=True).stdout.split()[1]}.\n")
    a(ALGORITHM_FLOW)
    a("## Data\n")
    a("- **Synthetic receipts.** Indian shop bills generated with known answers (see `src/expense_tracker/synthetic.py`). "
      "8 categories, 480 shops, split 60/20/20 into shops seen in training, new shops for validation and new shops for test. "
      "1,500 training receipts, 300 validation, 600 test (half from known shops, half from new ones). Item names held out of "
      "training, abbreviated POS-style names, generic shop names and mixed baskets make categorization realistic rather than trivial.")
    a("- **Synthetic photos.** 80 test receipts rendered at three levels (clean; medium: tilt up to 2.5°, blur, smaller, uneven light; "
      "hard: tilt 2.5–5°, more blur and noise, faded ink, low resolution) = 240 images. Validation: 40 receipts × 3 = 120 images.")
    a("- **Real scans.** SROIE (ICDAR 2019) Malaysian receipts with labelled company, date and total. Receipts 000–049 were used "
      "while writing the parser rules (dev); 100 others, sampled with a fixed seed, are used only for the final numbers (test).\n")

    a("## 1. Choosing the settings (dev sets only)\n")
    a("| Preprocessing | Page mode | Synthetic dev, all 3 fields right | SROIE dev, all 3 right |")
    a("|---|---|---|---|")
    for r in dev["rows"]:
        a(f"| {r['config']} | psm {r['psm']} | {r['synthetic_dev']['all_three']:.1f}% | {r['sroie_dev']['all_three']:.1f}% |")
    a(f"\nChosen: **{dev['chosen']['config']}, psm {dev['chosen']['psm']}** (best average of the two dev sets).\n")

    a("## 2. Field extraction on the test sets\n")
    a("Merchant counts as right when it matches the label at ≥ 90% character similarity (OCR often drops a letter); "
      "exact matches are shown too. Date and total must match exactly.\n")
    a("| Test set | n | Merchant (fuzzy) | Merchant (exact) | Date | Total | All three | Item recall |")
    a("|---|---|---|---|---|---|---|---|")
    for k, v in f["main"].items():
        a(f"| {k} | {v['n']} | {v['merchant_fuzzy']}% | {v['merchant_exact']}% | {v['date']}% | {v['total']}% | "
          f"{v['all_three']}% | {str(v.get('item_recall', '–')) + ('%' if 'item_recall' in v else '')} |")
    a("\n![](figures/fields_accuracy.png)\n")
    a("The parser gets every synthetic receipt right when given perfect text (checked in the tests), so the drop on hard photos "
      "is OCR errors. The synthetic receipts share print conventions with the parser rules, so they flatter it; the SROIE "
      "numbers are the honest real-world estimate.\n")

    a("## 3. Preprocessing ablation (test sets)\n")
    a("| Setting | Synthetic all 3 | clean | medium | hard | SROIE all 3 |")
    a("|---|---|---|---|---|---|")
    for r in f["ablation"]:
        a(f"| {r['config']} | {r['synthetic']['all_three']}% | {r['clean']['all_three']}% | {r['medium']['all_three']}% | "
          f"{r['hard']['all_three']}% | {r['sroie']['all_three']}% |")
    a("\n![](figures/preprocessing_ablation.png)\n")
    a("On phone-style photos every step helps; upscaling matters most, and light flattening is what makes the hard "
      "photos readable at all. On SROIE's flat, evenly lit scans the steps add little (39% raw vs 42% full). "
      "Binarization helps sharp clean images but wrecks the soft ones, so it is left off.\n")

    a("## 4. Categorization: rules only vs ML only vs hybrid\n")
    a("Same training data for all three. Mean ± standard deviation over 5 random seeds (each seed generates new shops and receipts). "
      "`C` (regularization) and τ were chosen on the validation split of each seed.\n")
    a("| Method | Accuracy | Macro-F1 | Known shops | New shops | Coverage |")
    a("|---|---|---|---|---|---|")
    for m, v in c["summary"].items():
        a(f"| {m} | {pct(v['accuracy']['mean'])} ± {pct(v['accuracy']['std'])} | {pct(v['macro_f1']['mean'])} ± {pct(v['macro_f1']['std'])} | "
          f"{pct(v['accuracy_known_shops']['mean'])} | {pct(v['accuracy_new_shops']['mean'])} | {pct(v['coverage']['mean'])} |")
    a("\nCoverage is the share of receipts that get a category at all; the rules give up on shops they don't recognise. "
      f"The hybrid answered {pct(c['hybrid_sources_mean']['rule'])}% of test receipts from its shop list and the rest with the model.\n")
    a("![](figures/categorization_methods.png)\n")
    a("Per-category F1 (mean of 5 seeds):\n")
    a("| Category | Rules only | ML only | Hybrid |")
    a("|---|---|---|---|")
    for cat in c["labels"]:
        a(f"| {cat} | " + " | ".join(pct(c['summary'][m]['per_class_f1'][cat]) for m in ("Rules only", "ML only", "Hybrid")) + " |")
    a("\n![](figures/confusion_matrix_hybrid.png)\n")

    g = c["gate_at_chosen_tau"]
    a("## 5. The confidence gate τ\n")
    a("τ was chosen as the smallest value that keeps the accuracy of automatically saved receipts at or above 97% on validation. "
      f"Chosen τ per seed: {', '.join(str(x['tau']) for x in c['chosen'])}.\n")
    a(f"On test: **{pct(g['review_rate']['mean'])}%** of receipts were sent to the user, the ones saved automatically were "
      f"**{pct(g['auto_accuracy']['mean'])}%** correct, and after the user fixes the flagged ones overall accuracy is "
      f"**{pct(g['accuracy_after_review']['mean'])}%** (mean of 5 seeds).\n")
    zero = sum(x["tau"] == 0 for x in c["chosen"])
    at05 = next(r for r in c["tau_curve_test_mean"] if abs(r["tau"] - 0.5) < 1e-9)
    if zero:
        a(f"In {zero} of 5 seeds the hybrid already reached 97% on validation without asking the user, so τ = 0 was chosen "
          "and the gate stayed off. The curve shows the trade-off if you want fewer mistakes saved: at τ = 0.5 about "
          f"{pct(at05['review_rate'])}% of receipts go to the user and {pct(at05['auto_accuracy'])}% of the rest are right.\n")
    a("![](figures/confidence_threshold.png)\n")

    a("## 6. End to end: category from OCR output\n")
    a("The categorizers trained on clean text, then fed the merchant and items the OCR pipeline read from the 240 synthetic test photos (seed 0).\n")
    a("| Input | Rules only | ML only | Hybrid |")
    a("|---|---|---|---|")
    for lv, v in e2e.items():
        a(f"| {lv} | " + " | ".join(pct(v[m]['accuracy']) for m in ("Rules only", "ML only", "Hybrid")) + " |")
    a("")

    a("## 7. Learning from corrections\n")
    a(f"Cold start: each model starts from {l['seed_set_size']} training receipts (5 per category). One simulated user then scans "
      f"{l['n_stream']} receipts from {l['n_shops']} shops they visit, some often and some rarely. The user corrects every wrong "
      "suggestion and confirms every flagged one. Mean of 5 seeds.\n")
    a("| Method | First 100 receipts | Last 200 receipts | New shops, 2nd half | Corrections | Confirmations |")
    a("|---|---|---|---|---|---|")
    for m, v in l["summary"].items():
        a(f"| {m} | {pct(v['first100_accuracy'])}% | {pct(v['last200_accuracy'])}% | {pct(v['first_visit_accuracy_second_half'])}% | "
          f"{v['corrections']:.0f} | {v['confirmations']:.0f} |")
    a("\n![](figures/learning_curve.png)\n")
    a("Remembering corrected shops does most of the work, because people go back to the same shops. Retraining the model "
      "is what helps at shops the user hasn't been to before (the 'new shops' column).\n")

    a("## 8. Speed\n")
    a("Mean time per receipt with one CPU core per receipt.\n")
    a("| Test set | Preprocess | OCR | Parse | Total (mean) | Total (95th percentile) |")
    a("|---|---|---|---|---|---|")
    for k, v in f["latency"].items():
        a(f"| {k} | {v['preprocess']['mean_ms']:.0f} ms | {v['ocr']['mean_ms']:.0f} ms | {v['parse']['mean_ms']:.1f} ms | "
          f"{v['total']['mean_ms']:.0f} ms | {v['total']['p95_ms']:.0f} ms |")
    a("\nOn SROIE, preprocessing is the slow part: some scans are over 2,000 px wide and light flattening runs at full "
      "resolution. Shrinking very large images first would cut this.")
    a(f"\nCategorizing one receipt takes {c['predict_ms_per_receipt']:.1f} ms; training the hybrid on 1,500 receipts takes "
      f"{c['train_seconds']:.1f} s.\n")
    a("![](figures/latency.png)\n")

    a("## 9. Tests\n")
    a(f"{cov['summary']}. Line coverage of `src/expense_tracker`: **{cov['coverage']}**. Full report: `results/test_report.txt`.\n")

    a("## Two fixes made after a first full run\n")
    a("Being open about this, since it affects how far the test numbers can be trusted:\n")
    a("1. **Denoising.** The first run filtered every image. On test, clean photos scored lower than medium ones because "
      "the median filter erased decimal points on sharp images. The fix (filter only soft images) and its threshold "
      "(sharpness 1500) were chosen on extra validation images and the SROIE dev set, not on test, and every experiment "
      "was then rerun from scratch.")
    a("2. **Fuzzy shop matching.** Checking end-to-end errors showed a new shop, 'Lanka Enterprises', matched to the "
      "known 'Apna Enterprises' because the generic word dominates the similarity. The same mistake happened on "
      "validation (2 to 12 receipts per seed), so the rule now also requires the first word of the name to match.\n")
    a("## Limitations\n")
    a("- Categorization is measured on synthetic data. The absolute numbers depend on how the generator was built; the "
      "comparisons between methods are the useful part. A set of real labelled Indian receipts is needed to confirm them.")
    a("- SROIE receipts are Malaysian; Indian receipts (GST lines, ₹, Hindi text) were only tested synthetically.")
    a("- Tesseract is run in English only. Bills printed in Hindi would need the `hin` language pack.")
    a("- Timings come from a 2-core cloud machine; a phone or laptop will differ.")
    (RESULTS / "summary.md").write_text("\n".join(s) + "\n")


def main():
    f, c = load_json("fields.json"), load_json("categorization.json")
    e2e, l, dev = load_json("categorization_end_to_end.json"), load_json("learning.json"), load_json("dev_tuning.json")
    cov = load_json("tests.json")
    fig_fields(f)
    fig_ablation(f)
    fig_categorization(c)
    fig_confusion(c)
    fig_tau(c)
    fig_learning(l)
    fig_latency(f)
    write_summary(f, c, e2e, l, dev, cov)
    print("wrote results/summary.md and", len(list(FIGURES.glob("*.png"))), "figures")


if __name__ == "__main__":
    main()
