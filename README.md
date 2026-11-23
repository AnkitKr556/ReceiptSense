# ReceiptSense

**Receipt-driven expense tracker with OCR and ML-based auto categorization.** Runs entirely on your own computer.

![Tests](https://github.com/AnkitKr556/ReceiptSense/actions/workflows/ci.yml/badge.svg)

You take a photo of a bill. The code cleans up the image, reads it with OCR, pulls out the shop name, date, total and items, and puts the expense in a spending category. When it isn't sure about the category it asks you, and every correction you make teaches it something.

This folder has the full algorithm, a test suite, and the experiments behind every number in `results/summary.md`.

## Headline results

Full details, tables and charts: [`results/summary.md`](results/summary.md).

| What was measured | Result |
|---|---|
| Field extraction, 240 synthetic phone photos | merchant, date and total all right on 91% of clean and medium photos, 33% of hard ones |
| Field extraction, 100 real scanned receipts (SROIE) | total 74%, date 76%, shop name 60%; all three right on 42% |
| Category accuracy, rules only / ML only / hybrid (600 test receipts, 5 seeds) | 80.1% / 94.9% / **95.9%** |
| Learning from corrections, starting from 40 training receipts | 64% if it never learns, **99.5%** after about 25 corrections |
| Speed on one CPU core | about 1.1 s per phone photo, 2.8 s per large scan; categorizing takes 2 ms |
| Tests | 86 passing (73 algorithm + 13 web API), 98% line coverage |

The categorization numbers come from synthetic receipts, so treat the comparison between methods as the result, not the exact percentages. The SROIE numbers are the real-world check for field extraction.

## How it works

### 1. Cleaning up the photo (`preprocess.py`)

Tesseract was built for clean, straight, high-resolution scans. Phone photos of receipts are none of those things, so we fix them first:

| Step | What it does | Why |
|---|---|---|
| Grayscale | drops colour | colour adds nothing for printed text |
| Upscale | resizes so the image is at least 1000 px wide | Tesseract reads best when letters are 20–30 px tall |
| Flatten lighting | estimates the paper background with a morphological closing and divides it out | shadows and uneven light made Tesseract skip the darker half of the receipt |
| Deskew | tries rotations from −6° to +6° and keeps the one where the rows of text are sharpest (projection profile) | tilted lines merge into each other |
| Denoise | 3×3 median filter, only on soft images (sharpness below 1500) | cleans compression artifacts on soft photos; on sharp images it erodes decimal points, so it is skipped there |
| Binarize | optional Otsu or adaptive threshold | measured in the ablation: it helps sharp images but ruins soft photos and real scans, so it is off |

### 2. Reading the text (`ocr.py`)

Tesseract 5 (LSTM engine) with page mode 4, "a single column of text of varying sizes", which is what a receipt is. Words come back with positions and confidence scores; we group them into lines and then join lines that sit on the same row, so a label like `TOTAL` stays next to its amount even when Tesseract treats them as separate blocks.

### 3. Finding the fields (`parser.py`)

Plain rules based on how receipts are laid out:

- **Numbers first.** Characters OCR confuses with digits are fixed, but only inside tokens that are already mostly digits: `1O4.5O → 104.50`, while `SOAP` stays `SOAP`.
- **Total.** Lines with a total keyword are candidates. Strong keywords (`Grand Total`, `Net Amount`, `Amount Payable`, `Total (incl. GST)`) beat a plain `Total`. `Sub Total`, `Total Qty` and `Total Tax` are excluded. Among equals we take the last one printed before the payment lines (`Cash`, `Change`), and we stop looking at a `GST SUMMARY` table, which repeats the word TOTAL with the pre-tax amount. If no keyword is found, we take the largest amount that isn't cash handed over or change.
- **Date.** Seven formats (`03/11/2026`, `03.11.26`, `03-Nov-2026`, `2026-11-03`...), day first, falling back to month first when the day-first reading is impossible. Dates on a line that says "Date" win over dates in an address.
- **Merchant.** The first line near the top that is mostly letters and isn't a header, address or phone line. A line naming a kind of business (`Store`, `Medicals`, `Sdn Bhd`, `Restaurant`...) is preferred.
- **Items.** Lines between the shop name and the total that end in a price, with prices and quantities stripped off.

### 4. Choosing the category (`categorizer.py`)

Three strategies with the same `fit` / `predict` interface (the Strategy pattern), so they can be compared fairly:

- **Rules only (baseline).** A dictionary of shops seen before, then hand-written keyword rules on the shop name (`medical → Health & Pharmacy`, `filling station → Fuel & Transport`). If nothing matches it gives up.
- **ML only (baseline).** TF-IDF features on the shop name and item names, then logistic regression. Word n-grams pick up meaningful words (`paneer`, `recharge`); character n-grams (3–5 letters) keep working when OCR garbles a letter (`paracetam0l`) or the shop prints abbreviations (`PNR BTR MSL`).
- **Hybrid (proposed).**

```
function CATEGORIZE(receipt):
    m <- normalize(receipt.merchant)
    if m is in the shop list (exact, or ≥ 90% similar to absorb OCR typos):
        return (shop_list[m], confidence 1.0, RULE)
    (c, p) <- logistic_regression.top_class(tfidf(m, receipt.items))
    if p < tau: return (c, p, NEEDS_REVIEW)        # ask the user
    return (c, p, ML)

function ON_FEEDBACK(receipt, true_category):      # user corrected or confirmed
    shop_list[normalize(receipt.merchant)] <- true_category   # takes effect immediately
    training_set <- training_set + {(receipt, true_category)}
    if feedback since last retrain >= k: retrain(logistic_regression, training_set)
```

τ (the confidence threshold) and the regularization strength `C` are chosen on a validation split, never on test data.

### 5. Putting it together (`pipeline.py`)

`ReceiptPipeline` chains the stages, times each one, and returns the fields, the category and a list of warnings (blurry image, no total found...). Missing files, non-images and tiny images raise `InvalidImageError` instead of crashing.

## Project layout

```
src/expense_tracker/
    preprocess.py     image clean-up
    ocr.py            Tesseract wrapper, line grouping
    parser.py         merchant / date / total / items rules
    categorizer.py    rules, ML and hybrid categorizers, learning loop
    pipeline.py       image -> fields -> category
    synthetic.py      synthetic Indian receipts with known answers, rendered as photos
    evaluate.py       metrics
    cli.py            command line
    api.py            FastAPI backend (web app)
    storage.py        SQLite repository for saved expenses
    web/              dashboard (HTML, CSS, JS)
tests/                86 tests (unit, end to end, API)
experiments/          one script per experiment, run_all.py runs everything
data/download_sroie.py  fetches the real receipts used for testing
results/              summary.md, figures/, raw JSON
```

## Setup

1. Install Tesseract (the OCR program):
   - Ubuntu: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: the installer from https://github.com/UB-Mannheim/tesseract/wiki, then add it to PATH
2. Install the Python package (Python 3.10+): `pip install -e ".[dev]"` (includes the web app's packages)

## The web app

```bash
pip install -e ".[web]"                 # FastAPI, uvicorn (once)
python -m expense_tracker serve          # open http://127.0.0.1:8000
python -m expense_tracker serve --host 0.0.0.0   # also reachable from a phone on the same Wi-Fi
```

What it does:

- **Upload a receipt** (drag and drop, click, or paste an image). It is read in about a second and shown next to the photo so you can check the shop, date, total, items and category before saving. Short on real bills? Use **Try a sample receipt**.
- **Review queue.** Receipts the categorizer isn't sure about (confidence below τ) are flagged *review*. Confirming or correcting the category is sent to `HybridCategorizer.feedback`: the shop is learned at once and the model retrains every 10 corrections. The "Shops learned" tile shows this happening.
- **Dashboard**: spending by category, the last 12 months, and a searchable, filterable list of expenses. Click a bar to filter.
- **Private by design.** Nothing leaves the computer and no CDN is used. The SQLite database (`expenses.db`), the receipt images and *your* copy of the model (`categorizer.pkl`) all live in `data/app/` (or set `EXPENSE_TRACKER_DATA`). Delete that folder to start fresh.

API (interactive docs at http://127.0.0.1:8000/docs):

| Method | Path | What it does |
|---|---|---|
| POST | `/api/receipts` | upload an image (multipart `file`): OCR, parse, categorize, save |
| GET | `/api/expenses?month=&category=&needs_review=&q=` | list expenses |
| POST | `/api/expenses` | add one by hand (category optional: the model predicts it) |
| GET / PATCH / DELETE | `/api/expenses/{id}` | read, edit (a category here counts as user feedback), delete |
| GET | `/api/expenses/{id}/image` | the stored receipt photo |
| GET | `/api/summary?month=` | totals by category and by month, review count, model stats |
| GET | `/api/sample?level=clean\|medium\|hard` | a synthetic receipt photo for demos |
| GET | `/api/categories`, `/api/model`, `/api/health` | metadata |

The backend uses a Repository (`storage.py`) for the database and a small service object around the categorizer, which locks during updates and saves the model after every correction. The dashboard is plain HTML/CSS/JS in `src/expense_tracker/web/`, served by FastAPI, so it needs no Node build step.

## Using it from the command line

```bash
# train the hybrid categorizer (on the synthetic training set, or on your own CSV)
python -m expense_tracker train --out models/categorizer.pkl
python -m expense_tracker train --csv my_expenses.csv --out models/categorizer.pkl   # columns: merchant, items (;-separated), category

# try one of the sample photos in examples/, or make a new one
python -m expense_tracker process examples/receipt_medium.jpg --model models/categorizer.pkl
python -m expense_tracker sample receipt.jpg --level medium
python -m expense_tracker process receipt.jpg --model models/categorizer.pkl
```

From Python:

```python
from expense_tracker import ReceiptPipeline, HybridCategorizer

model = HybridCategorizer.load("models/categorizer.pkl")
result = ReceiptPipeline(categorizer=model).process_file("receipt.jpg")
print(result.to_dict())
if result.prediction.needs_review:
    model.feedback(result.fields.merchant, result.fields.items, "Groceries")  # what the user picked
```

## Reproducing the results

```bash
python data/download_sroie.py      # 150 real receipts, about 70 MB
pytest                             # 86 tests
python experiments/run_all.py      # every experiment, about 40 minutes on 2 cores
```

Everything is seeded, so the numbers come out the same each time (Tesseract versions can shift OCR results slightly).

## What's not in here

User accounts and cloud sync. The app is single-user and local on purpose (privacy). The slides mention a React dashboard; it was built as a plain-JS single page instead, so the project runs with Python alone.
=======
# ReceiptSense
>>>>>>> 9dde14ab7a68df3149ccd1b597f1428142727c79
