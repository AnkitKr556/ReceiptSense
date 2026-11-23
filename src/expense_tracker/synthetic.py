"""Synthetic Indian shop receipts with known ground truth.

There is no public labelled dataset of Indian receipts with spending categories,
so we generate one. Each merchant gets a fixed name, address, GSTIN and print
style (date format, total label, tax lines...), the way a real shop's billing
software always prints the same layout. Receipts are rendered to images and
degraded at three difficulty levels to imitate phone photos.

The data is made deliberately hard to categorize, the way real bills are:
- about 30% of shops have generic names ("Gupta Traders", "Om Enterprises")
  that say nothing about what they sell
- every third item name in each category never appears in training, only in
  validation and test, so the model has to cope with words it has not seen
- some shops print abbreviated item names, as POS machines do ("PNR BTR MSL")
- about 20% of items on a bill come from other categories (mixed baskets)
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import Expense

CATEGORIES = ["Groceries", "Dining", "Fuel & Transport", "Health & Pharmacy",
              "Shopping", "Electronics", "Entertainment", "Bills & Utilities"]

SURNAMES = ["Sharma", "Gupta", "Agarwal", "Verma", "Singh", "Yadav", "Patel", "Shah", "Mehta", "Jain",
            "Reddy", "Iyer", "Nair", "Rao", "Khan", "Das", "Bose", "Mishra", "Pandey", "Tiwari",
            "Chauhan", "Kapoor", "Malhotra", "Bansal", "Goel", "Saxena", "Joshi", "Kulkarni", "Desai",
            "Pillai", "Menon", "Chopra", "Srivastava", "Dubey", "Maurya", "Jaiswal"]
WORDS = ["Shree", "Sri", "Krishna", "Ganesh", "Balaji", "Laxmi", "Durga", "Om", "Sai", "Annapurna",
         "New", "Royal", "Classic", "Modern", "City", "Metro", "Star", "Galaxy", "Sunrise", "Lotus",
         "Green", "Golden", "Kashi", "Ganga", "Banaras", "Lanka", "Assi", "Sigra", "Vishwanath",
         "Sankat Mochan", "Shiv", "Hanuman", "Jai Mata Di", "Prince", "Lucky", "Apna", "Bharat",
         "National", "Diamond", "Silver"]
AREAS = ["Lanka", "Sigra", "Assi", "Godowlia", "Bhelupur", "Mahmoorganj", "Sarnath", "Cantt",
         "Nadesar", "Lahurabir", "Maldahiya", "Pandeypur", "Chitaipur", "Sundarpur", "Rathyatra"]

# Shop-name templates. {S} = surname, {W} = word. Some templates appear in more
# than one category on purpose.
TEMPLATES = {
    "Groceries": ["{S} General Store", "{W} Kirana Store", "{W} Supermarket", "{W} Fresh Mart",
                  "{S} Provision Store", "{W} Grocers", "{W} Daily Needs", "{S} & Sons", "{W} Mart",
                  "{W} Bazaar"],
    "Dining": ["Hotel {W}", "{W} Restaurant", "{W} Dhaba", "{W} Cafe", "{S} Sweets", "{W} Bhojnalaya",
               "{W} Kitchen", "{W} Biryani House", "{W} Chaat Bhandar", "{W} Foods"],
    "Fuel & Transport": ["{W} Filling Station", "{W} Fuels", "{S} Petroleum", "{W} Service Station",
                         "{W} Auto Centre", "{W} Travels", "{W} Tours & Travels", "{W} Cab Services"],
    "Health & Pharmacy": ["{W} Medical Store", "{S} Medicals", "{W} Pharmacy", "{W} Chemist & Druggist",
                          "{W} Diagnostics", "{W} Clinic", "{W} Medical & General Store", "{W} Health Care"],
    "Shopping": ["{W} Garments", "{W} Fashion", "{W} Collection", "{S} Cloth House", "{W} Footwear",
                 "{W} Saree Centre", "{W} Bazaar", "{W} Mart", "{W} Emporium", "{S} Traders", "{S} & Sons"],
    "Electronics": ["{W} Electronics", "{W} Mobile Point", "{W} Computers", "{S} Electricals",
                    "{W} Digital World", "{W} Mobile Gallery", "{S} Traders", "{W} Enterprises"],
    "Entertainment": ["{W} Cinemas", "{W} Multiplex", "{W} Talkies", "{W} Gaming Zone", "{W} Bowling",
                      "{W} Fun World", "{W} Board Game Cafe", "{W} Enterprises"],
    "Bills & Utilities": ["{W} Telecom", "{W} Recharge Point", "{W} Communication", "{W} Broadband",
                          "{W} Gas Agency", "{W} Cyber Cafe", "{S} Enterprises", "{W} Bill Payment Centre"],
}

# (printed item name, min price, max price, max quantity)
ITEMS = {
    "Groceries": [("Toor Dal 1kg", 140, 180, 2), ("Basmati Rice 5kg", 450, 650, 1), ("Wheat Atta 10kg", 380, 480, 1),
                  ("Sugar 1kg", 42, 50, 3), ("Mustard Oil 1L", 150, 190, 2), ("Refined Oil 1L", 130, 160, 2),
                  ("Butter 500g", 255, 285, 1), ("Milk 1L", 54, 68, 4), ("Paneer 200g", 80, 100, 2),
                  ("Eggs 12pc", 72, 96, 2), ("Tea 250g", 120, 160, 1), ("Salt 1kg", 24, 30, 2),
                  ("Turmeric 100g", 30, 45, 1), ("Onion 1kg", 30, 60, 3), ("Potato 2kg", 40, 70, 2),
                  ("Tomato 1kg", 30, 80, 2), ("Bread", 40, 55, 2), ("Detergent 1kg", 110, 160, 1),
                  ("Bath Soap 4pc", 140, 220, 1), ("Toothpaste 150g", 85, 120, 1), ("Instant Noodles", 56, 96, 2),
                  ("Poha 500g", 35, 50, 1), ("Besan 500g", 55, 75, 1), ("Rajma 1kg", 140, 180, 1),
                  ("Ghee 500ml", 290, 340, 1), ("Curd 400g", 35, 45, 2), ("Biscuits", 10, 40, 4)],
    "Dining": [("Paneer Butter Masala", 220, 320, 1), ("Dal Makhani", 180, 260, 1), ("Butter Naan", 45, 70, 4),
               ("Tandoori Roti", 15, 30, 6), ("Veg Biryani", 160, 240, 2), ("Chicken Biryani", 220, 320, 2),
               ("Masala Dosa", 90, 150, 2), ("Idli Sambhar", 60, 100, 2), ("Chole Bhature", 90, 140, 2),
               ("Samosa", 15, 25, 4), ("Kachori Sabzi", 30, 50, 3), ("Lassi", 50, 90, 2), ("Masala Chai", 15, 30, 4),
               ("Cold Coffee", 90, 160, 2), ("Veg Thali", 150, 250, 2), ("Paneer Tikka", 220, 300, 1),
               ("Gulab Jamun", 40, 70, 2), ("Rasmalai", 60, 90, 2), ("Hakka Noodles", 140, 200, 1),
               ("Pav Bhaji", 90, 140, 2), ("Aloo Paratha", 50, 90, 2), ("Jeera Rice", 120, 170, 1)],
    "Fuel & Transport": [("Engine Oil 1L", 380, 520, 1), ("Bike Service", 350, 700, 1),
                         ("Puncture Repair", 50, 120, 1), ("Car Wash", 200, 450, 1), ("Coolant 1L", 220, 320, 1),
                         ("Air Filter", 180, 350, 1), ("Brake Pad Set", 600, 1200, 1), ("Taxi Fare", 300, 1500, 1),
                         ("Outstation Cab", 1800, 4500, 1), ("Bus Ticket", 250, 900, 3), ("Toll Charges", 50, 250, 1),
                         ("Parking Fee", 20, 60, 1), ("Waiting Charges", 50, 200, 1)],
    "Health & Pharmacy": [("Paracetamol 500mg", 20, 35, 3), ("Cetirizine 10mg", 18, 40, 2),
                          ("Azithromycin 500mg", 70, 120, 1), ("Cough Syrup 100ml", 90, 140, 1),
                          ("Vitamin C Tablets", 60, 110, 2), ("ORS Sachet", 20, 25, 5), ("Antiseptic 100ml", 60, 90, 1),
                          ("Bandage Roll", 30, 60, 2), ("Thermometer", 150, 300, 1), ("Blood Test CBC", 250, 400, 1),
                          ("Thyroid Profile", 450, 800, 1), ("Consultation Fee", 300, 800, 1),
                          ("Pantoprazole 40mg", 90, 150, 1), ("Multivitamin Caps", 250, 450, 1),
                          ("Face Mask 10pc", 50, 120, 1), ("Cotton 100g", 40, 70, 1), ("Pain Relief Gel", 90, 160, 1)],
    "Shopping": [("Cotton Shirt", 599, 1499, 2), ("Denim Jeans", 899, 1999, 1), ("Kurta Set", 799, 2499, 1),
                 ("Saree", 999, 4999, 1), ("T-Shirt", 299, 799, 3), ("Leggings", 249, 499, 2),
                 ("Sports Shoes", 999, 2999, 1), ("Sandals", 399, 999, 1), ("Socks 3 Pair", 149, 299, 2),
                 ("Handbag", 599, 1999, 1), ("Leather Belt", 299, 799, 1), ("Bedsheet Double", 499, 1299, 1),
                 ("Towel Set", 299, 699, 1), ("Dupatta", 199, 699, 1), ("Track Pants", 399, 899, 1),
                 ("Winter Jacket", 1299, 3499, 1), ("Kids Frock", 399, 999, 2), ("Sunglasses", 299, 1299, 1)],
    "Electronics": [("Mobile Charger", 299, 999, 1), ("USB-C Cable", 149, 499, 2), ("Earphones", 299, 1499, 1),
                    ("Bluetooth Speaker", 999, 2999, 1), ("Power Bank 10000", 799, 1799, 1),
                    ("Screen Guard", 99, 399, 1), ("Mobile Cover", 149, 499, 1), ("Extension Board", 299, 699, 1),
                    ("Pen Drive 32GB", 299, 599, 1), ("Wireless Mouse", 399, 999, 1), ("Keyboard", 499, 1299, 1),
                    ("Smartwatch", 1499, 4999, 1), ("Mobile Repair", 300, 1500, 1), ("Laptop Service", 500, 1500, 1),
                    ("HDMI Cable", 199, 499, 1), ("Memory Card 64GB", 499, 899, 1), ("Ceiling Fan", 1499, 3499, 1),
                    ("Electric Kettle", 699, 1499, 1)],
    "Entertainment": [("Movie Ticket Classic", 150, 250, 4), ("Movie Ticket Recliner", 350, 550, 3),
                      ("Popcorn Large", 250, 350, 2), ("Nachos", 180, 260, 1), ("Gaming 1 Hour", 100, 250, 3),
                      ("Bowling Game", 300, 500, 4), ("VR Ride", 200, 400, 2), ("Event Pass", 499, 1499, 2),
                      ("Arcade Tokens", 100, 500, 1), ("Water Park Entry", 600, 1200, 3),
                      ("Board Game Session", 150, 300, 4), ("Concert Ticket", 999, 2999, 2), ("Convenience Fee", 30, 70, 1)],
    "Bills & Utilities": [("Prepaid Recharge", 199, 799, 1), ("Postpaid Bill", 399, 1299, 1),
                          ("DTH Recharge", 250, 600, 1), ("Broadband Bill", 499, 1199, 1),
                          ("Electricity Bill", 600, 4500, 1), ("LPG Cylinder Refill", 850, 1100, 1),
                          ("Water Bill", 200, 600, 1), ("Data Pack", 19, 299, 1), ("Router Rental", 100, 200, 1),
                          ("Printout", 2, 10, 20), ("Photocopy", 1, 3, 40), ("Late Payment Fee", 50, 150, 1),
                          ("New SIM Card", 50, 100, 1)],
}
# Generic shop names that any kind of business might use.
GENERIC_TEMPLATES = ["{S} Enterprises", "{S} Traders", "{W} Enterprises", "{S} & Sons", "{W} Store",
                     "{W} Point", "{W} Centre", "{S} Brothers", "{W} Associates", "{W} Agencies"]
GENERIC_SHARE = 0.3
ITEM_STYLES = ["title", "upper", "trunc", "vowel"]
ITEM_STYLE_P = [0.4, 0.3, 0.15, 0.15]

# Items any kind of shop might sell; added now and then to blur the categories.
SHARED_ITEMS = [("Mineral Water 1L", 20, 25, 2), ("Cold Drink 600ml", 40, 50, 2), ("Chips", 20, 50, 2),
                ("Carry Bag", 5, 10, 1), ("Chocolate", 10, 100, 2)]

GST_RATE = {"Groceries": 0.025, "Dining": 0.025, "Fuel & Transport": 0.09, "Health & Pharmacy": 0.06,
            "Shopping": 0.06, "Electronics": 0.09, "Entertainment": 0.09, "Bills & Utilities": 0.09}

DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%y", "%d-%b-%Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%y"]
DATE_LABELS = ["Date:", "Dt:", "DATE:", "Date :"]
TOTAL_LABELS = ["TOTAL", "GRAND TOTAL", "Grand Total", "Net Amount", "NET PAYABLE", "Amount Payable",
                "Total Amount", "Bill Amount", "Total Rs."]
HEADERS = ["TAX INVOICE", "CASH MEMO", "RETAIL INVOICE", "BILL OF SUPPLY", None, None]
FOOTERS = ["Thank you! Visit again", "Thank You", "Goods once sold will not be taken back",
           "Have a nice day", "** Thank you for shopping **"]
WIDTH = 42  # characters per printed line


@dataclass
class Merchant:
    name: str
    category: str
    address: str
    phone: str
    gstin: str
    style: dict


@dataclass
class Receipt:
    merchant: Merchant
    items: list[tuple[str, int, float, float]]  # name, qty, rate, amount
    date: date
    bill_no: int
    time: str
    subtotal: float
    taxes: list[tuple[str, float]]
    round_off: float
    total: float
    payment: str
    fuel: dict | None = None

    @property
    def item_names(self) -> list[str]:
        return [i[0] for i in self.items]

    def to_expense(self) -> Expense:
        return Expense(self.merchant.name, self.item_names, self.merchant.category)

    def truth(self) -> dict:
        return {"merchant": self.merchant.name, "date": self.date.isoformat(), "total": self.total,
                "items": self.item_names, "category": self.merchant.category}


# ------------------------------------------------------------------ merchants

def _gstin(rng) -> str:
    letters = "".join(rng.choice(list(string.ascii_uppercase), 5))
    digits = "".join(rng.choice(list(string.digits), 4))
    return f"09{letters}{digits}{rng.choice(list(string.ascii_uppercase))}1Z{rng.choice(list(string.digits))}"


def _phone(rng) -> str:
    if rng.random() < 0.3:
        return f"0542-{rng.integers(2200000, 2399999)}"
    return f"9{rng.integers(1000, 9999)} {rng.integers(10000, 99999)}"


def make_merchants(rng: np.random.Generator, per_category: int = 60) -> list[Merchant]:
    used: set[str] = set()
    merchants = []
    for cat in CATEGORIES:
        made = 0
        while made < per_category:
            tpl = rng.choice(GENERIC_TEMPLATES if rng.random() < GENERIC_SHARE else TEMPLATES[cat])
            name = tpl.format(S=rng.choice(SURNAMES), W=rng.choice(WORDS))
            if name in used:
                continue
            used.add(name)
            fuel_style = cat == "Fuel & Transport" and any(k in name for k in ("Filling", "Fuels", "Petroleum")) \
                and rng.random() < 0.8
            style = {
                "date_fmt": str(rng.choice(DATE_FORMATS)),
                "date_label": str(rng.choice(DATE_LABELS)),
                "total_label": str(rng.choice(TOTAL_LABELS)),
                "currency": str(rng.choice(["Rs.", "", "Rs ", "INR "])),
                "tax": bool(rng.random() < 0.7),
                "round_off": bool(rng.random() < 0.35),
                "header": rng.choice(HEADERS),
                "columns": bool(rng.random() < 0.6),
                "comma": bool(rng.random() < 0.5),
                "upper": bool(rng.random() < 0.6),
                "footer": str(rng.choice(FOOTERS)),
                "fuel": bool(fuel_style),
                "item_style": str(rng.choice(ITEM_STYLES, p=ITEM_STYLE_P)),
            }
            address = f"{rng.integers(1, 120)}/{rng.integers(1, 40)}, {rng.choice(AREAS)}, Varanasi - 2210{rng.integers(1, 12):02d}"
            merchants.append(Merchant(name, cat, address, _phone(rng), _gstin(rng), style))
            made += 1
    return merchants


# ------------------------------------------------------------------ receipts

# Every third item of each category is held out of training.
HELD_OUT = {name for cat in ITEMS for i, (name, *_rest) in enumerate(ITEMS[cat]) if i % 3 == 2}


def print_name(name: str, style: str) -> str:
    """How a shop's billing software prints an item name."""
    if style == "upper":
        return name.upper()
    if style == "trunc":
        return " ".join(w[:4] for w in name.upper().split())
    if style == "vowel":
        def squeeze(w):
            if any(c.isdigit() for c in w) or len(w) <= 3:
                return w
            return w[0] + "".join(c for c in w[1:] if c not in "AEIOU")
        return " ".join(squeeze(w) for w in name.upper().split())
    return name


def make_receipt(rng: np.random.Generator, merchant: Merchant, mix: float = 0.2,
                 allow_held_out: bool = True) -> Receipt:
    cat = merchant.category
    start = date(2025, 1, 1)
    d = start + timedelta(days=int(rng.integers(0, 670)))
    t = f"{rng.integers(8, 23):02d}:{rng.integers(0, 60):02d}"
    bill_no = int(rng.integers(100, 99999))

    if merchant.style["fuel"]:
        product = "Petrol" if rng.random() < 0.75 else "Diesel"
        rate = round(float(rng.uniform(94.5, 106.5) if product == "Petrol" else rng.uniform(87, 93)), 2)
        total = float(rng.choice([200, 300, 500, 1000, 1500, 2000, 2500]))
        if rng.random() < 0.3:
            total = round(float(rng.uniform(150, 3000)), 2)
        volume = round(total / rate, 2)
        return Receipt(merchant, [(product, 1, rate, total)], d, bill_no, t, total, [], 0.0, total,
                       "Cash", fuel={"product": product, "rate": rate, "volume": volume,
                                     "nozzle": int(rng.integers(1, 9)),
                                     "vehicle": f"UP65 {''.join(rng.choice(list('ABCDEFGHJK'), 2))} {rng.integers(1000, 9999)}"})

    n = int(rng.integers(1, 7))
    items, seen = [], set()
    while len(items) < n:
        r = rng.random()
        if r < mix / 2:
            pool = SHARED_ITEMS
        elif r < mix:
            pool = ITEMS[CATEGORIES[int(rng.integers(len(CATEGORIES)))]]
        else:
            pool = ITEMS[cat]
        name, lo, hi, qmax = pool[int(rng.integers(len(pool)))]
        if name in seen or (not allow_held_out and name in HELD_OUT):
            continue
        seen.add(name)
        qty = int(rng.integers(1, qmax + 1))
        rate = float(rng.integers(lo, hi + 1))
        if rng.random() < 0.3:
            rate += float(rng.choice([0.5, 0.25, 0.75]))
        items.append((print_name(name, merchant.style["item_style"]), qty, rate, round(qty * rate, 2)))

    subtotal = round(sum(i[3] for i in items), 2)
    taxes = []
    if merchant.style["tax"]:
        half = round(subtotal * GST_RATE[cat], 2)
        rate_pct = GST_RATE[cat] * 100
        taxes = [(f"CGST @{rate_pct:g}%", half), (f"SGST @{rate_pct:g}%", half)]
    gross = round(subtotal + sum(t[1] for t in taxes), 2)
    round_off = 0.0
    if merchant.style["round_off"]:
        round_off = round(round(gross) - gross, 2)
    total = round(gross + round_off, 2)
    payment = str(rng.choice(["Cash", "UPI", "Card", "Paytm"]))
    return Receipt(merchant, items, d, bill_no, t, subtotal, taxes, round_off, total, payment)


def _money(x: float, style: dict) -> str:
    return f"{x:,.2f}" if style["comma"] else f"{x:.2f}"


def _lr(left: str, right: str, width: int = WIDTH) -> str:
    gap = max(1, width - len(left) - len(right))
    return left + " " * gap + right


def receipt_lines(r: Receipt) -> list[tuple[str, str]]:
    """Text lines of the printed receipt, each tagged 'title', 'center' or 'left'."""
    s, m = r.merchant.style, r.merchant
    title = m.name.upper() if s["upper"] else m.name
    out = [(title, "title"), (m.address, "center"), (f"Ph: {m.phone}", "center")]
    if s["tax"] or s["fuel"]:
        out.append((f"GSTIN: {m.gstin}", "center"))
    date_str = r.date.strftime(s["date_fmt"])
    sep = ("-" * WIDTH, "left")

    if r.fuel:
        f = r.fuel
        out += [(f"Receipt No: {r.bill_no}", "left"),
                (_lr(f"{s['date_label']} {date_str}", f"Time: {r.time}"), "left"),
                (f"Nozzle No: {f['nozzle']}", "left"), (f"Product: {f['product']}", "left"),
                (_lr("Rate (Rs/L):", f"{f['rate']:.2f}"), "left"), (_lr("Volume (L):", f"{f['volume']:.2f}"), "left"),
                (_lr("Amount (Rs):", _money(r.total, s)), "left"), (f"Vehicle No: {f['vehicle']}", "left"),
                sep, ("Thank you, drive safe", "center")]
        return out

    if s["header"]:
        out.append((s["header"], "center"))
    out += [(_lr(f"Bill No: {r.bill_no}", f"{s['date_label']} {date_str}"), "left"),
            (_lr(f"Time: {r.time}", f"Pay: {r.payment}"), "left"), sep]
    if s["columns"]:
        out += [("Item".ljust(20) + "Qty".rjust(4) + "Rate".rjust(8) + "Amount".rjust(10), "left"), sep]
        for name, qty, rate, amt in r.items:
            out.append((name[:20].ljust(20) + str(qty).rjust(4) + f"{rate:.2f}".rjust(8)
                        + _money(amt, s).rjust(10), "left"))
    else:
        for name, qty, rate, amt in r.items:
            left = f"{name[:22]} x{qty}" if qty > 1 else name[:26]
            out.append((_lr(left, _money(amt, s)), "left"))
    out.append(sep)
    if r.taxes or r.round_off:
        out.append((_lr("Sub Total", _money(r.subtotal, s)), "left"))
        for label, amt in r.taxes:
            out.append((_lr(label, _money(amt, s)), "left"))
        if r.round_off:
            out.append((_lr("Round Off", f"{r.round_off:+.2f}"), "left"))
        out.append(sep)
    out.append((_lr(s["total_label"], f"{s['currency']}{_money(r.total, s)}"), "left"))
    out.append(sep)
    if r.payment == "Cash" and r.total < 4900:
        tendered = float(np.ceil(r.total / 100.0) * 100 + (100 if r.total % 100 > 80 else 0))
        out += [(_lr("Cash Tendered", _money(tendered, s)), "left"),
                (_lr("Change", _money(tendered - r.total, s)), "left")]
    out.append((s["footer"], "center"))
    return out


# ------------------------------------------------------------------ rendering

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"

PROFILES = {
    #          rotation range,   blur sigma, noise, scale,     jpeg, ink,       light gradient
    "clean":  dict(angle=(0, 0),     blur=0.0, noise=3,  scale=1.0,  jpeg=92, ink=(20, 30),   grad=1.0),
    "medium": dict(angle=(0.5, 2.5), blur=0.8, noise=8,  scale=0.8,  jpeg=70, ink=(45, 65),   grad=0.85),
    "hard":   dict(angle=(2.5, 5.0), blur=1.0, noise=12, scale=0.7,  jpeg=50, ink=(90, 110),  grad=0.7),
}


def render_receipt(r: Receipt, rng: np.random.Generator, difficulty: str = "clean") -> np.ndarray:
    """Draw the receipt and degrade it like a phone photo. Returns a BGR image."""
    p = PROFILES[difficulty]
    size = int(rng.integers(20, 25))
    font = ImageFont.truetype(str(_FONT_DIR / "DejaVuSansMono.ttf"), size)
    bold = ImageFont.truetype(str(_FONT_DIR / "DejaVuSansMono-Bold.ttf"), int(size * 1.2))
    char_w = font.getlength("M")
    margin = int(size * 1.6)
    width = int(char_w * WIDTH) + 2 * margin
    line_h = int(size * 1.45)
    lines = receipt_lines(r)
    height = 2 * margin + line_h * (len(lines) + 1)

    paper = tuple(int(x) for x in rng.integers(236, 252, 3))
    ink_level = int(rng.integers(*p["ink"]))
    ink = (ink_level, ink_level, ink_level)
    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)
    y = margin
    for text, kind in lines:
        if kind == "title":
            w = bold.getlength(text)
            draw.text(((width - w) / 2, y), text, font=bold, fill=ink)
            y += int(line_h * 1.3)
            continue
        x = (width - font.getlength(text)) / 2 if kind == "center" else margin
        draw.text((x, y), text, font=font, fill=ink)
        y += line_h
    arr = np.array(img)[:, :, ::-1].copy()
    return degrade(arr, rng, difficulty, paper)


def degrade(img: np.ndarray, rng: np.random.Generator, difficulty: str, paper=(245, 245, 245)) -> np.ndarray:
    p = PROFILES[difficulty]
    out = img.astype(np.float32)
    if p["grad"] < 1.0:  # uneven light: one corner darker than the other
        h, w = out.shape[:2]
        gy = np.linspace(1.0, p["grad"], h)[:, None]
        gx = np.linspace(1.0, 0.5 + p["grad"] / 2, w)[None, :]
        if rng.random() < 0.5:
            gx = gx[:, ::-1]
        out *= (gy * gx)[:, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    lo, hi = p["angle"]
    if hi > 0:
        angle = float(rng.uniform(lo, hi)) * (1 if rng.random() < 0.5 else -1)
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2 - w / 2
        m[1, 2] += nh / 2 - h / 2
        bg = tuple(int(c * 0.9) for c in paper[::-1])
        out = cv2.warpAffine(out, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=bg)
    if p["scale"] < 1.0:
        out = cv2.resize(out, None, fx=p["scale"], fy=p["scale"], interpolation=cv2.INTER_AREA)
    if p["blur"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    noise = rng.normal(0, p["noise"], out.shape)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ datasets

@dataclass
class Dataset:
    merchants: dict[str, list[Merchant]]  # "known", "val_new", "test_new"
    train: list[Receipt]
    val: list[Receipt]
    test: list[Receipt]
    test_is_known: list[bool] = field(default_factory=list)
    val_is_known: list[bool] = field(default_factory=list)


def build_dataset(seed: int = 0, per_category: int = 60, n_train: int = 1500,
                  n_val: int = 300, n_test: int = 600) -> Dataset:
    """Split merchants 60/20/20 into known / new-for-validation / new-for-test.

    Training receipts come only from known merchants. Validation and test are half
    known merchants (new bills from shops seen in training) and half shops never seen.
    """
    rng = np.random.default_rng(seed)
    merchants = make_merchants(rng, per_category)
    known, val_new, test_new = [], [], []
    for cat in CATEGORIES:
        ms = [m for m in merchants if m.category == cat]
        order = rng.permutation(len(ms))
        a, b = int(0.6 * len(ms)), int(0.8 * len(ms))
        known += [ms[i] for i in order[:a]]
        val_new += [ms[i] for i in order[a:b]]
        test_new += [ms[i] for i in order[b:]]

    def draw(pool, n, held_out=True):
        return [make_receipt(rng, pool[int(rng.integers(len(pool)))], allow_held_out=held_out) for _ in range(n)]

    train = draw(known, n_train, held_out=False)
    val = draw(known, n_val // 2) + draw(val_new, n_val - n_val // 2)
    test = draw(known, n_test // 2) + draw(test_new, n_test - n_test // 2)
    return Dataset({"known": known, "val_new": val_new, "test_new": test_new}, train, val, test,
                   test_is_known=[True] * (n_test // 2) + [False] * (n_test - n_test // 2),
                   val_is_known=[True] * (n_val // 2) + [False] * (n_val - n_val // 2))
