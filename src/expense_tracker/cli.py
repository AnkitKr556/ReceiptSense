"""Command line interface.

    python -m expense_tracker train  --out models/categorizer.pkl [--csv data.csv]
    python -m expense_tracker process receipt.jpg --model models/categorizer.pkl
    python -m expense_tracker sample  out.jpg [--level medium]
    python -m expense_tracker serve   [--port 8000]      # web app at http://127.0.0.1:8000

The CSV for `train` needs the columns merchant, items (separated by ';') and category.
Without --csv, the model is trained on the synthetic training set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .categorizer import HybridCategorizer
from .models import Expense
from .pipeline import InvalidImageError, ReceiptPipeline


def read_csv(path: str) -> list[Expense]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = {"merchant", "items", "category"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return [Expense(r["merchant"], [i.strip() for i in r["items"].split(";") if i.strip()], r["category"])
            for r in rows]


def cmd_train(args) -> int:
    if args.csv:
        data = read_csv(args.csv)
    else:
        from .synthetic import build_dataset
        data = [r.to_expense() for r in build_dataset(seed=0).train]
    model = HybridCategorizer(tau=args.tau, retrain_every=args.retrain_every, C=args.C).fit(data)
    model.save(args.out)
    print(f"trained on {len(data)} expenses, {len(model.rules.table)} known shops -> {args.out}")
    return 0


def cmd_process(args) -> int:
    model = HybridCategorizer.load(args.model) if args.model else None
    try:
        result = ReceiptPipeline(categorizer=model, psm=args.psm).process_file(args.image)
    except InvalidImageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = result.to_dict()
    if args.show_text:
        out["ocr_text"] = [l.text for l in result.ocr_lines]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_sample(args) -> int:
    from .synthetic import build_dataset, render_receipt
    rng = np.random.default_rng(args.seed)
    receipt = build_dataset(seed=0).test[args.seed % 600]
    cv2.imwrite(args.out, render_receipt(receipt, rng, args.level))
    print(json.dumps({"written": args.out, "truth": receipt.truth()}, indent=2))
    return 0


def cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        raise SystemExit('the web app needs extra packages: pip install -e ".[web]"')
    from .api import create_app
    app = create_app(data_dir=args.data, model_path=args.model)
    print(f"Expense tracker running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="expense_tracker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train the hybrid categorizer")
    t.add_argument("--csv", help="labelled expenses (merchant, items, category)")
    t.add_argument("--out", default="models/categorizer.pkl")
    t.add_argument("--tau", type=float, default=0.4)
    t.add_argument("--retrain-every", type=int, default=10)
    t.add_argument("--C", type=float, default=10.0)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("process", help="read one receipt image")
    r.add_argument("image")
    r.add_argument("--model", help="trained categorizer (.pkl)")
    r.add_argument("--psm", type=int, default=4)
    r.add_argument("--show-text", action="store_true", help="also print the raw OCR lines")
    r.set_defaults(func=cmd_process)

    s = sub.add_parser("sample", help="write a synthetic receipt image to try the pipeline on")
    s.add_argument("out")
    s.add_argument("--level", choices=["clean", "medium", "hard"], default="clean")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_sample)

    w = sub.add_parser("serve", help="run the web app (API + dashboard)")
    w.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from your phone on the same Wi-Fi")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--data", default=None, help="folder for the database and images (default data/app)")
    w.add_argument("--model", default="models/categorizer.pkl", help="starting categorizer")
    w.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    if args.command in ("train", "sample"):
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
