"""Expense categorizers.

Three strategies share one interface (fit / predict), so they can be swapped and
compared on the same data:

- RuleCategorizer: merchant dictionary + hand-written keyword rules (baseline)
- MLCategorizer: TF-IDF features + logistic regression (baseline)
- HybridCategorizer: the proposed method from the slides

    function CATEGORIZE(r):
        m <- normalize(r.merchant)
        if m in RuleDict: return (RuleDict[m], 1.0, RULE)
        (c, p) <- model.top_class(tfidf(m + r.items))
        if p < tau: return (c, p, NEEDS_REVIEW)
        return (c, p, ML)

    function ON_FEEDBACK(r, c*):
        RuleDict[normalize(r.merchant)] <- c*
        TrainSet <- TrainSet + {(r, c*)}
        if feedback_since_last_train >= k: retrain(model, TrainSet)
"""
from __future__ import annotations

import difflib
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .models import Expense, Prediction

UNKNOWN = "Uncategorized"

_STOPWORDS = {"pvt", "ltd", "private", "limited", "sdn", "bhd", "the", "and", "co", "llp", "ms"}


def normalize_merchant(name: str) -> str:
    """'M/s. Sharma General Store Pvt. Ltd.' -> 'sharma general store'."""
    s = (name or "").lower().replace("&", " and ").replace("m/s", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(t for t in s.split() if t not in _STOPWORDS)


class Categorizer(Protocol):
    def fit(self, data: Iterable[Expense]) -> "Categorizer": ...
    def predict(self, merchant: str, items: list[str]) -> Prediction: ...


# ------------------------------------------------------------------ rules

class MerchantRules:
    """Dictionary from normalized merchant name to category, with fuzzy lookup.

    Fuzzy matching absorbs small OCR errors in the merchant name, e.g.
    'sharma genera1 store'. Generic words make different shops look alike
    ('sri enterprises' vs 'sai enterprises' are 93% similar), so the first word,
    which is usually the distinctive part of the name, must match closely too.
    """

    def __init__(self, fuzzy_cutoff: float = 0.9, first_word_cutoff: float = 0.75):
        self.fuzzy_cutoff = fuzzy_cutoff
        self.first_word_cutoff = first_word_cutoff
        self.table: dict[str, str] = {}

    def fit(self, data: Iterable[Expense]) -> "MerchantRules":
        votes: dict[str, Counter] = defaultdict(Counter)
        for e in data:
            key = normalize_merchant(e.merchant)
            if key:
                votes[key][e.category] += 1
        self.table = {k: v.most_common(1)[0][0] for k, v in votes.items()}
        return self

    def add(self, merchant: str, category: str) -> None:
        key = normalize_merchant(merchant)
        if key:
            self.table[key] = category

    def lookup(self, merchant: str) -> str | None:
        key = normalize_merchant(merchant)
        if not key:
            return None
        if key in self.table:
            return self.table[key]
        if self.fuzzy_cutoff < 1.0:
            first = key.split()[0]
            for cand in difflib.get_close_matches(key, self.table.keys(), n=3, cutoff=self.fuzzy_cutoff):
                if difflib.SequenceMatcher(None, first, cand.split()[0]).ratio() >= self.first_word_cutoff:
                    return self.table[cand]
        return None


# Hand-written keyword rules on the merchant name, checked in order.
KEYWORD_RULES: list[tuple[str, list[str]]] = [
    ("Bills & Utilities", ["telecom", "recharge", "communication", "broadband", "gas agency",
                           "electricity", "cyber"]),
    ("Fuel & Transport", ["filling station", "petrol", "petroleum", "fuel", "fuels", "service station",
                          "travels", "cab", "taxi", "auto centre", "motors"]),
    ("Health & Pharmacy", ["medical", "medicals", "pharmacy", "chemist", "druggist", "clinic",
                           "diagnostic", "diagnostics", "hospital", "health"]),
    ("Entertainment", ["cinema", "cinemas", "multiplex", "talkies", "gaming", "bowling", "fun world",
                       "entertainment"]),
    ("Dining", ["restaurant", "dhaba", "cafe", "bhojnalaya", "kitchen", "biryani", "sweets", "chaat",
                "hotel", "foods", "bakery"]),
    ("Groceries", ["kirana", "general store", "provision", "grocer", "grocers", "supermarket",
                   "daily needs", "fresh"]),
    ("Electronics", ["electronics", "mobile", "computers", "electricals", "digital"]),
    ("Shopping", ["garments", "fashion", "collection", "cloth", "footwear", "saree", "emporium",
                  "boutique"]),
]


class RuleCategorizer:
    """Baseline 1: merchant dictionary, then keyword rules, otherwise give up."""

    def __init__(self, fuzzy_cutoff: float = 0.9, keyword_rules=KEYWORD_RULES):
        self.merchants = MerchantRules(fuzzy_cutoff)
        self.keyword_rules = [(cat, [re.compile(r"\b" + re.escape(k) + r"\b") for k in kws])
                              for cat, kws in keyword_rules]

    def fit(self, data: Iterable[Expense]) -> "RuleCategorizer":
        self.merchants.fit(data)
        return self

    def predict(self, merchant: str, items: list[str]) -> Prediction:
        cat = self.merchants.lookup(merchant)
        if cat:
            return Prediction(cat, 1.0, "rule")
        name = normalize_merchant(merchant)
        for cat, patterns in self.keyword_rules:
            if any(p.search(name) for p in patterns):
                return Prediction(cat, 0.8, "keyword")
        return Prediction(UNKNOWN, 0.0, "none", needs_review=True)


# ------------------------------------------------------------------ ML

def _as_matrix(merchants: list[str], items: list[list[str]]) -> np.ndarray:
    rows = [[normalize_merchant(m), " ; ".join(i).lower()] for m, i in zip(merchants, items)]
    return np.array(rows, dtype=object).reshape(-1, 2)


class MLCategorizer:
    """Baseline 2: TF-IDF on merchant name and item names, then logistic regression.

    Word n-grams catch meaningful words ('medical', 'paneer'); character n-grams
    keep working when OCR garbles a letter or two ('paracetam0l').
    """

    def __init__(self, C: float = 10.0):
        self.C = C
        self.model: Pipeline | None = None

    def _build(self) -> Pipeline:
        features = ColumnTransformer([
            ("merchant_words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), 0),
            ("merchant_chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), 0),
            ("item_words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), 1),
            ("item_chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), 1),
        ])
        clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced")
        return Pipeline([("features", features), ("clf", clf)])

    def fit(self, data: Iterable[Expense]) -> "MLCategorizer":
        data = list(data)
        X = _as_matrix([e.merchant for e in data], [e.items for e in data])
        y = [e.category for e in data]
        self.model = self._build().fit(X, y)
        return self

    @property
    def classes(self) -> list[str]:
        return list(self.model.classes_)

    def predict_proba(self, merchants: list[str], items: list[list[str]]) -> np.ndarray:
        return self.model.predict_proba(_as_matrix(merchants, items))

    def predict(self, merchant: str, items: list[str]) -> Prediction:
        proba = self.predict_proba([merchant], [items])[0]
        k = int(np.argmax(proba))
        return Prediction(self.model.classes_[k], float(proba[k]), "ml")


# ------------------------------------------------------------------ hybrid

class HybridCategorizer:
    """Proposed method: merchant rules first, then ML, with a confidence gate and a learning loop."""

    def __init__(self, tau: float = 0.5, retrain_every: int = 10, fuzzy_cutoff: float = 0.9, C: float = 10.0):
        self.tau = tau
        self.retrain_every = retrain_every
        self.rules = MerchantRules(fuzzy_cutoff)
        self.ml = MLCategorizer(C=C)
        self.train_data: list[Expense] = []
        self.pending = 0
        self.retrain_count = 0

    def fit(self, data: Iterable[Expense]) -> "HybridCategorizer":
        self.train_data = list(data)
        self.rules.fit(self.train_data)
        self.ml.fit(self.train_data)
        self.pending = 0
        return self

    def predict(self, merchant: str, items: list[str]) -> Prediction:
        cat = self.rules.lookup(merchant)
        if cat:
            return Prediction(cat, 1.0, "rule")
        p = self.ml.predict(merchant, items)
        return Prediction(p.category, p.confidence, "ml", needs_review=p.confidence < self.tau)

    def feedback(self, merchant: str, items: list[str], category: str) -> bool:
        """The user confirmed or corrected a category. Returns True if the model was retrained."""
        self.rules.add(merchant, category)
        self.train_data.append(Expense(merchant, list(items), category))
        self.pending += 1
        if self.retrain_every and self.pending >= self.retrain_every:
            self.ml.fit(self.train_data)
            self.pending = 0
            self.retrain_count += 1
            return True
        return False

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "HybridCategorizer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, HybridCategorizer):
            raise TypeError(f"{path} does not contain a HybridCategorizer")
        return obj
