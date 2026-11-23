import pytest

from expense_tracker.categorizer import (UNKNOWN, HybridCategorizer, MerchantRules, MLCategorizer,
                                         RuleCategorizer, normalize_merchant)
from expense_tracker.models import Expense
from expense_tracker.synthetic import build_dataset


@pytest.fixture(scope="module")
def data():
    ds = build_dataset(seed=1, n_train=600, n_val=100, n_test=200)
    return [r.to_expense() for r in ds.train], [r.to_expense() for r in ds.test], ds.test_is_known


def test_normalize_merchant():
    assert normalize_merchant("M/s. Sharma General Store Pvt. Ltd.") == "sharma general store"
    assert normalize_merchant("Gupta & Sons") == "gupta sons"
    assert normalize_merchant("") == ""


def test_merchant_rules_exact_and_fuzzy():
    rules = MerchantRules(fuzzy_cutoff=0.9).fit([Expense("Sharma General Store", [], "Groceries")])
    assert rules.lookup("SHARMA GENERAL STORE") == "Groceries"
    assert rules.lookup("Sharma Genera1 Store") == "Groceries"  # OCR typo
    assert rules.lookup("Verma Medical Hall") is None
    assert rules.lookup("") is None


def test_merchant_rules_do_not_confuse_shops_with_generic_names():
    rules = MerchantRules().fit([Expense("Apna Enterprises", [], "Dining"), Expense("Sai Enterprises", [], "Shopping"),
                                 Expense("Shiv Enterprises", [], "Electronics")])
    assert rules.lookup("Lanka Enterprises") is None
    assert rules.lookup("Sri Enterprises") is None
    assert rules.lookup("5hiv Enterprises") == "Electronics"  # an OCR typo in a longer first word is still fine
    assert rules.lookup("5ai Enterprises") is None  # too short to tell from a different name; the ML model decides


def test_merchant_rules_majority_vote_and_add():
    rules = MerchantRules().fit([Expense("Om Store", [], "Groceries"), Expense("Om Store", [], "Groceries"),
                                 Expense("Om Store", [], "Shopping")])
    assert rules.lookup("Om Store") == "Groceries"
    rules.add("Om Store", "Shopping")
    assert rules.lookup("Om Store") == "Shopping"


def test_rule_categorizer_keyword_and_unknown():
    rc = RuleCategorizer().fit([])
    assert rc.predict("Kashi Filling Station", []).category == "Fuel & Transport"
    assert rc.predict("Lotus Cyber Cafe", []).category == "Bills & Utilities"  # checked before 'cafe'
    p = rc.predict("Gupta Traders", ["Keyboard"])
    assert p.category == UNKNOWN and p.needs_review


def test_ml_categorizer_learns(data):
    train, test, _ = data
    ml = MLCategorizer().fit(train)
    acc = sum(ml.predict(e.merchant, e.items).category == e.category for e in test) / len(test)
    assert acc > 0.85
    assert set(ml.classes) <= {e.category for e in train}


def test_hybrid_uses_rules_for_known_shops(data):
    train, _, _ = data
    h = HybridCategorizer().fit(train)
    e = train[0]
    p = h.predict(e.merchant, ["something new"])
    assert p.source == "rule" and p.confidence == 1.0 and not p.needs_review


def test_hybrid_gate_flags_low_confidence(data):
    train, _, _ = data
    h = HybridCategorizer(tau=1.01).fit(train)  # nothing reaches the gate, so every ML answer is flagged
    p = h.predict("Unseen Name Xyz", ["mystery item"])
    assert p.source == "ml" and p.needs_review
    h.tau = 0.0
    assert not h.predict("Unseen Name Xyz", ["mystery item"]).needs_review


def test_hybrid_feedback_updates_rules_and_retrains(data):
    train, _, _ = data
    h = HybridCategorizer(retrain_every=3).fit(train)
    assert h.feedback("Brand New Shop", ["Paneer Tikka"], "Dining") is False
    assert h.predict("Brand New Shop", []).category == "Dining"  # remembered immediately
    h.feedback("Another Shop", ["Saree"], "Shopping")
    assert h.feedback("Third Shop", ["Petrol"], "Fuel & Transport") is True  # third feedback triggers retrain
    assert h.retrain_count == 1 and h.pending == 0
    assert len(h.train_data) == len(train) + 3


def test_hybrid_no_retrain_when_disabled(data):
    train, _, _ = data
    h = HybridCategorizer(retrain_every=0).fit(train)
    for i in range(5):
        h.feedback(f"Shop {i}", ["x"], "Dining")
    assert h.retrain_count == 0


def test_hybrid_beats_rules_on_new_shops(data):
    train, test, known = data
    rules, hybrid = RuleCategorizer().fit(train), HybridCategorizer().fit(train)
    new = [e for e, k in zip(test, known) if not k]
    acc = lambda m: sum(m.predict(e.merchant, e.items).category == e.category for e in new) / len(new)
    assert acc(hybrid) > acc(rules)


def test_save_and_load(tmp_path, data):
    train, _, _ = data
    h = HybridCategorizer(tau=0.3).fit(train[:200])
    path = tmp_path / "m.pkl"
    h.save(path)
    loaded = HybridCategorizer.load(path)
    assert loaded.tau == 0.3
    e = train[5]
    assert loaded.predict(e.merchant, e.items).category == h.predict(e.merchant, e.items).category


def test_load_rejects_other_objects(tmp_path):
    import pickle
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(TypeError):
        HybridCategorizer.load(path)
