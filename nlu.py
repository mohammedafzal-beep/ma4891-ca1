"""
nlu.py — Natural Language Understanding
=========================================
• spaCy EntityRuler for fast, rule-based entity extraction
  (DINE_MODE, TABLE_NUMBER, MENU_ITEM, PAYMENT_METHOD, CATEGORY)
• Keyword-based intent classification
• RoBERTa Transformer for real-time sentiment analysis
"""
from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.language import Language

# ─── Lazy-loaded globals ──────────────────────────────────────────
_nlp: Optional[Language] = None
_sentiment_pipeline = None


# ═══════════════════════════════════════════════════════════════════
#  spaCy + EntityRuler Setup
# ═══════════════════════════════════════════════════════════════════

# ─── Helper: build a token-based pattern from a phrase ────────────
def _phrase_to_token_pattern(label: str, phrase: str) -> Dict[str, Any]:
    """Convert 'chicken burger' → {"label": "MENU_ITEM", "pattern": [{"LOWER": "chicken"}, {"LOWER": "burger"}]}"""
    tokens = phrase.lower().split()
    if len(tokens) == 1:
        return {"label": label, "pattern": phrase.lower()}
    return {"label": label, "pattern": [{"LOWER": t} for t in tokens]}


# Entity patterns for the EntityRuler
ENTITY_PATTERNS: List[Dict[str, Any]] = []

# DINE_MODE patterns (token-based for multi-word)
_DINE_PHRASES = [
    "dine in", "dine-in", "dinein", "takeaway", "take away",
    "take-away", "pickup", "pick up", "pick-up", "delivery",
]
for phrase in _DINE_PHRASES:
    ENTITY_PATTERNS.append(_phrase_to_token_pattern("DINE_MODE", phrase))

# TABLE_NUMBER patterns (table 1 through table 20)
for n in range(1, 21):
    ENTITY_PATTERNS.append({"label": "TABLE_NUMBER", "pattern": [
        {"LOWER": "table"}, {"LOWER": str(n)}
    ]})
    ENTITY_PATTERNS.append({"label": "TABLE_NUMBER", "pattern": [
        {"LOWER": "table"}, {"LOWER": "number"}, {"LOWER": str(n)}
    ]})
    ENTITY_PATTERNS.append({"label": "TABLE_NUMBER", "pattern": [
        {"LOWER": "table"}, {"LOWER": "no"}, {"LOWER": str(n)}
    ]})

# MENU_ITEM patterns — ALL use token-based matching for robustness
MENU_ITEM_NAMES = [
    "chicken burger", "beef burger", "grilled salmon", "ramen",
    "pad thai", "margherita pizza", "pepperoni pizza", "steak",
    "caesar salad", "fries", "onion rings", "garlic bread",
    "soup of the day", "coke", "sprite", "iced latte", "iced tea",
    "fresh orange juice", "cheesecake", "brownie", "ice cream sundae",
]
for item in MENU_ITEM_NAMES:
    ENTITY_PATTERNS.append(_phrase_to_token_pattern("MENU_ITEM", item))
    # Also add common plural variants (e.g. "chicken burgers", "steaks")
    # so the EntityRuler catches plurals too
    if not item.endswith("s") and not item.endswith("y"):
        ENTITY_PATTERNS.append(_phrase_to_token_pattern("MENU_ITEM", item + "s"))
    elif item.endswith("y"):
        ENTITY_PATTERNS.append(_phrase_to_token_pattern("MENU_ITEM", item[:-1] + "ies"))


# PAYMENT_METHOD patterns (token-based for multi-word)
_PAY_PHRASES = [
    "cash", "card", "credit card", "debit card", "visa",
    "mastercard", "paynow", "pay now", "apple pay",
    "applepay", "google pay", "googlepay",
]
for phrase in _PAY_PHRASES:
    ENTITY_PATTERNS.append(_phrase_to_token_pattern("PAYMENT_METHOD", phrase))

# CATEGORY patterns for menu browsing
for cat in ["mains", "sides", "drinks", "desserts", "main", "dessert",
            "drink", "side", "beverage", "beverages"]:
    ENTITY_PATTERNS.append({"label": "CATEGORY", "pattern": cat})


def _get_nlp() -> Language:
    """Load spaCy model with EntityRuler (singleton)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        # Add EntityRuler BEFORE the NER component so custom rules take precedence
        ruler = _nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(ENTITY_PATTERNS)
    return _nlp


# ═══════════════════════════════════════════════════════════════════
#  Menu Item Normalization (plural → singular)
# ═══════════════════════════════════════════════════════════════════

# Build a lookup: plural → canonical singular
_MENU_ITEM_SET = set(MENU_ITEM_NAMES)
_PLURAL_TO_SINGULAR: Dict[str, str] = {}
for _item in MENU_ITEM_NAMES:
    if not _item.endswith("s") and not _item.endswith("y"):
        _PLURAL_TO_SINGULAR[_item + "s"] = _item
    elif _item.endswith("y"):
        _PLURAL_TO_SINGULAR[_item[:-1] + "ies"] = _item


def _normalize_menu_item(raw: str) -> str:
    """Normalize a menu item to its canonical singular form."""
    if raw in _MENU_ITEM_SET:
        return raw
    if raw in _PLURAL_TO_SINGULAR:
        return _PLURAL_TO_SINGULAR[raw]
    # Try stripping trailing 's'
    if raw.endswith("s") and raw[:-1] in _MENU_ITEM_SET:
        return raw[:-1]
    return raw


# ═══════════════════════════════════════════════════════════════════

def _fallback_menu_items(text: str) -> List[str]:
    """Substring scan for menu items not caught by EntityRuler.
    Also handles plurals (e.g. 'burgers' → 'burger')."""
    t = text.lower()
    # Sort by length descending to match longest items first
    items = sorted(MENU_ITEM_NAMES, key=len, reverse=True)
    hits = []
    for item in items:
        if item in t:
            hits.append(item)
        else:
            # Try plural variants: 'chicken burgers' → 'chicken burger'
            plural = item + "s"
            if plural in t:
                hits.append(item)  # store as canonical singular
            elif item.endswith("y"):
                ies_form = item[:-1] + "ies"
                if ies_form in t:
                    hits.append(item)
    return hits


def _fallback_dine_mode(text: str) -> Optional[str]:
    """Substring scan for dining mode."""
    t = text.lower()
    for phrase in _DINE_PHRASES:
        if phrase in t:
            return _normalize_dine_mode(phrase)
    return None


def _fallback_payment(text: str) -> Optional[str]:
    """Substring scan for payment method."""
    t = text.lower()
    for phrase in _PAY_PHRASES:
        if phrase in t:
            return _normalize_payment(phrase)
    return None


# ═══════════════════════════════════════════════════════════════════
#  Entity Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Run spaCy pipeline and return extracted entities grouped by label.
    Falls back to regex/substring matching for entities the
    EntityRuler might miss due to tokenization differences.

    Returns dict with keys:
        dine_mode:   str | None
        table_number: int | None
        menu_items:  list[{name, qty}]
        payment_method: str | None
        category:    str | None
        person_name: str | None
    """
    nlp = _get_nlp()
    doc = nlp(text)

    result: Dict[str, Any] = {
        "dine_mode": None,
        "table_number": None,
        "menu_items": [],
        "payment_method": None,
        "category": None,
        "person_name": None,
        "raw_entities": [],
    }

    menu_hits: List[str] = []

    for ent in doc.ents:
        result["raw_entities"].append({
            "text": ent.text, "label": ent.label_,
            "start": ent.start_char, "end": ent.end_char,
        })

        if ent.label_ == "DINE_MODE" and result["dine_mode"] is None:
            result["dine_mode"] = _normalize_dine_mode(ent.text)

        elif ent.label_ == "TABLE_NUMBER" and result["table_number"] is None:
            num = _extract_number(ent.text)
            if num is not None:
                result["table_number"] = num

        elif ent.label_ == "MENU_ITEM":
            hit = _normalize_menu_item(ent.text.lower().strip())
            menu_hits.append(hit)

        elif ent.label_ == "PAYMENT_METHOD" and result["payment_method"] is None:
            result["payment_method"] = _normalize_payment(ent.text)

        elif ent.label_ == "CATEGORY" and result["category"] is None:
            result["category"] = _normalize_category(ent.text)

        elif ent.label_ == "PERSON" and result["person_name"] is None:
            result["person_name"] = ent.text.strip()

    # ── Fallback: regex/substring for entities EntityRuler missed ──
    if result["dine_mode"] is None:
        result["dine_mode"] = _fallback_dine_mode(text)

    if result["table_number"] is None:
        m = re.search(r"\btable\s+(\d{1,2})\b", text, re.I)
        if m:
            result["table_number"] = int(m.group(1))

    if not menu_hits:
        menu_hits = _fallback_menu_items(text)

    if result["payment_method"] is None:
        result["payment_method"] = _fallback_payment(text)

    # Build menu_items with quantities
    if menu_hits:
        result["menu_items"] = _attach_quantities(text, menu_hits)

    return result


def _normalize_dine_mode(raw: str) -> str:
    t = raw.lower().strip().replace("-", " ").replace("  ", " ")
    mapping = {
        "dine in": "dine_in", "dinein": "dine_in",
        "takeaway": "takeaway", "take away": "takeaway",
        "pickup": "takeaway", "pick up": "takeaway",
        "delivery": "delivery",
    }
    return mapping.get(t, t.replace(" ", "_"))


def _normalize_payment(raw: str) -> str:
    t = raw.lower().strip()
    mapping = {
        "cash": "cash",
        "card": "card", "credit card": "card", "debit card": "card",
        "visa": "card", "mastercard": "card",
        "paynow": "paynow", "pay now": "paynow",
        "apple pay": "apple_pay", "applepay": "apple_pay",
        "google pay": "google_pay", "googlepay": "google_pay",
    }
    return mapping.get(t, t.replace(" ", "_"))


def _normalize_category(raw: str) -> str:
    t = raw.lower().strip()
    mapping = {
        "main": "mains", "mains": "mains",
        "side": "sides", "sides": "sides",
        "drink": "drinks", "drinks": "drinks",
        "beverage": "drinks", "beverages": "drinks",
        "dessert": "desserts", "desserts": "desserts",
    }
    return mapping.get(t, t)


def _extract_number(text: str) -> Optional[int]:
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def _attach_quantities(full_text: str, item_names: List[str]) -> List[Dict[str, Any]]:
    """
    For each detected menu item, look for a nearby number
    (within 15 chars before or 5 chars after) as the quantity.
    Default qty = 1 if no number found.
    """
    text_lower = full_text.lower()
    items: Dict[str, int] = {}

    for item_name in item_names:
        idx = text_lower.find(item_name)
        if idx == -1:
            items[item_name] = items.get(item_name, 0) + 1
            continue

        # Look for quantity in a window around the item mention
        left = max(0, idx - 15)
        right = min(len(text_lower), idx + len(item_name) + 5)
        window = text_lower[left:right]
        m = re.search(r"(\d+)", window)
        qty = int(m.group(1)) if m else 1
        items[item_name] = items.get(item_name, 0) + qty

    return [{"name": k, "qty": v} for k, v in items.items()]


# ═══════════════════════════════════════════════════════════════════
#  Intent Classification (Rule-Based)
# ═══════════════════════════════════════════════════════════════════

# Intent keywords (checked in priority order)
INTENT_RULES: List[Tuple[str, List[str]]] = [
    ("goodbye",       ["bye", "goodbye", "see you", "goodnight", "good night", "exit", "quit"]),
    ("cancel",        ["cancel", "abort", "never mind", "nevermind", "forget it", "start over"]),
    ("confirm",       ["confirm", "yes place", "place order", "that's all", "thats all",
                       "done ordering", "finalize", "submit", "check out", "checkout"]),
    ("complain",      ["complaint", "complain", "terrible", "horrible", "disgusting",
                       "worst", "unacceptable", "disappointed", "angry", "furious"]),
    ("remove_item",   ["remove", "delete", "take off", "no more", "drop"]),
    ("add_item",      ["add", "order", "want", "give me", "i'd like", "i would like",
                       "can i get", "can i have", "get me", "i'll have", "ill have"]),
    ("ask_price",     ["price", "cost", "how much", "pricing"]),
    ("browse_menu",   ["menu", "browse", "what do you have", "what's available",
                       "show me", "what can i order", "options", "selection"]),
    ("show_order",    ["order status", "my order", "cart", "what did i order",
                       "current order", "show order", "view order", "status"]),
    ("help",          ["help", "commands", "what can you do", "guide", "instructions"]),
    ("hours",         ["hours", "opening", "when are you open", "closing", "timing",
                       "what time", "schedule"]),
    ("location",      ["location", "address", "where are you", "directions",
                       "where is the restaurant"]),
    ("contact",       ["contact", "phone", "email", "call", "number"]),
    ("greet",         ["hi", "hello", "hey", "good morning", "good afternoon",
                       "good evening", "howdy", "sup", "what's up", "yo"]),
    ("select_dine_mode", ["dine in", "dine-in", "takeaway", "take away", "delivery",
                          "pickup", "pick up"]),
    ("select_table",  ["table"]),
    ("pay",           ["pay", "payment", "cash", "card", "paynow", "apple pay",
                       "google pay"]),
]


def classify_intent(text: str) -> str:
    """
    Rule-based intent classification.
    Returns the first matching intent in priority order.
    Falls back to 'unknown'.
    """
    t = text.lower().strip()

    for intent, keywords in INTENT_RULES:
        for kw in keywords:
            if kw in t:
                return intent
    return "unknown"


# ═══════════════════════════════════════════════════════════════════
#  Sentiment Analysis (RoBERTa Transformer)
# ═══════════════════════════════════════════════════════════════════

def _get_sentiment_pipeline():
    """Lazy-load the sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline as hf_pipeline
        _sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
            truncation=True,
            max_length=512,
        )
    return _sentiment_pipeline


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Run RoBERTa sentiment analysis on the user's message.

    Returns:
        {
            "label": "positive" | "neutral" | "negative",
            "score": float (0-1),
            "all_scores": {label: score, ...}
        }
    """
    try:
        pipe = _get_sentiment_pipeline()
        results = pipe(text)[0]  # list of {label, score} dicts

        all_scores = {r["label"]: round(r["score"], 4) for r in results}

        # Find the top label
        top = max(results, key=lambda x: x["score"])

        return {
            "label": top["label"].lower(),
            "score": round(top["score"], 4),
            "all_scores": all_scores,
        }
    except Exception as e:
        # Fallback: if model fails to load, return neutral
        return {"label": "neutral", "score": 0.5, "all_scores": {}, "error": str(e)}
