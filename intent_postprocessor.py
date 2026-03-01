"""
Intent Post-Processor — Production Pipeline Module
===================================================
Sits between the BERT intent model and the intent engine.
Applies 5 optimization layers to raw model predictions:

1. Label Mapping      — normalises model labels to engine labels
2. Confidence Filter  — drops low-confidence single-token spans
3. Slot-Aware Fix     — uses spaCy PhraseMatcher to correct/validate
4. Keyword Fallback   — deterministic rules for intents the model misses
5. Conflict Pruning   — resolves contradictions + caps max intents

This module is used by app.py (real users) AND test_benchmark.py (evaluation).
"""

from typing import List, Dict, Any, Optional
from slots_spacy import (
    extract_order_items_phrase,
    extract_dining_mode_phrase,
    extract_payment_mode_phrase,
)

# ═══════════════════════════════════════════════════════════════════
# LAYER 1: LABEL MAPPING — model labels → engine labels
# ═══════════════════════════════════════════════════════════════════

LABEL_MAP = {
    # model uses short labels; engine uses prefixed ones
    "hours": "ask_hours",
    "location": "ask_location",
    "contact": "ask_contact",
    "payment_methods": "ask_payment_methods",
    "parking": "ask_parking",
    # noise labels from limited training data — not real engine intents
    "dietary_request": None,
    "clarify": None,
    "compliment": None,
    "ask_item_info": None,
    "delivery_status": None,
    "pickup_instructions": None,
    "waitlist_join": None,
}

# ═══════════════════════════════════════════════════════════════════
# LAYER 2: CONFIDENCE FILTERING
# ═══════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.55
MIN_TOKENS_LOW_CONF = 2


def _confidence_filter(intent_spans: List[Dict]) -> List[Dict]:
    """Drop low-confidence single-token spans (noise)."""
    strong = []
    for span in intent_spans:
        conf = span.get("avg_confidence", 1.0)
        tokens = span.get("token_count", 1)
        if conf >= CONFIDENCE_THRESHOLD or tokens >= MIN_TOKENS_LOW_CONF:
            strong.append(span)
    # longer spans first (more reliable)
    strong.sort(key=lambda s: s.get("token_count", 1), reverse=True)
    return strong


def _map_and_dedupe(spans: List[Dict]) -> List[str]:
    """Map model labels to engine labels and deduplicate."""
    result = []
    for span in spans:
        intent = span["intent"]
        if intent in LABEL_MAP:
            mapped = LABEL_MAP[intent]
            if mapped is not None and mapped not in result:
                result.append(mapped)
        elif intent not in result:
            result.append(intent)
    return result if result else ["unknown"]


# ═══════════════════════════════════════════════════════════════════
# LAYER 3: SLOT-AWARE INTENT CORRECTION
# ═══════════════════════════════════════════════════════════════════

def _slot_aware_correction(intents: List[str], user_text: str) -> List[str]:
    """Use spaCy slot results to correct/augment intents."""
    found_items = extract_order_items_phrase(user_text)
    found_dining = extract_dining_mode_phrase(user_text)
    found_payment = extract_payment_mode_phrase(user_text)

    corrected = list(intents)

    # Price keywords — if present, the user is asking about price, not ordering
    price_keywords = ["how much", "price", "cost", "expensive", "cheap"]
    is_asking_price = any(pk in user_text.lower() for pk in price_keywords)

    # menu items found → likely ordering (unless asking price)
    if found_items and "order_create" not in corrected:
        # If user is asking price and model already got ask_price right, skip injection
        if is_asking_price and "ask_price" in corrected:
            pass  # don't inject order_create — user wants price info, not ordering
        else:
            replaceable = {"browse_menu", "unknown"}
            if not is_asking_price:
                replaceable.add("ask_price")
            
            if any(i in replaceable for i in corrected):
                corrected = [i for i in corrected if i not in replaceable]
                corrected.insert(0, "order_create")
            else:
                corrected.append("order_create")

    # dining mode found → ordering context
    if found_dining and "order_create" not in corrected:
        corrected.append("order_create")

    # payment method found
    if found_payment and "ask_payment_methods" not in corrected and "order_create" not in corrected:
        corrected.append("ask_payment_methods")

    return corrected if corrected else ["unknown"]


# ═══════════════════════════════════════════════════════════════════
# LAYER 4: KEYWORD FALLBACK RULES
# ═══════════════════════════════════════════════════════════════════

_KEYWORD_RULES = {
    "complaint": [
        "terrible", "horrible", "awful", "worst", "disgusting",
        "cold food", "was cold", "not happy", "unacceptable", "rude",
        "bad service", "bad food", "poor service", "disappointed",
        "never coming back", "very bad", "not satisfied",
    ],
    "reservation_create": [
        "reserve a table", "book a table", "make a reservation",
        "reservation for", "table for", "reserve for",
        "booking for", "book for tonight", "reserve tonight",
        "i want to make a reservation",
    ],
    "reservation_cancel": [
        "cancel my reservation", "cancel the reservation", "cancel booking",
        "cancel the booking", "don't need the booking", "remove reservation",
    ],
    "order_cancel": [
        "cancel my order", "cancel everything", "cancel the order",
        "don't want anything", "remove everything", "clear my order",
        "forget it", "forget my order",
    ],
    "order_modify": [
        "add more", "add another", "also add", "and also",
        "remove the", "take off", "no more", "change my order",
        "swap the", "replace the", "instead of", "switch to",
    ],
    "browse_menu": [
        "menu", "show menu", "see menu", "see the menu", "what do you have",
        "what's available", "food options", "what food", "what can i order",
        "show me the menu", "what do you serve", "what's on the menu",
        "what are your", "list of food", "food list",
    ],
    "refund_request": [
        "refund", "money back", "get my money", "return my money",
        "want a refund", "give me back",
    ],
    "ask_parking": [
        "parking", "where can i park", "park my car", "car park",
        "parking lot", "parking nearby", "place to park",
    ],
    "ask_hours": [
        "opening hours", "open hours", "what time", "when do you close",
        "when do you open", "are you open", "closing time", "business hours",
        "hours", "open today",
    ],
    "ask_location": [
        "where are you", "your address", "located", "how do i get",
        "directions to", "where is the restaurant", "find you",
        "location", "address",
    ],
    "ask_contact": [
        "phone number", "contact", "email", "call you", "reach you",
    ],
    "ask_payment_methods": [
        "payment method", "accept credit", "take cash", "pay by",
        "paynow", "accept card", "payment options", "how can i pay",
    ],
}

_ORDER_SIGNALS = [
    "i want", "i'd like", "i would like", "give me", "get me",
    "order me", "can i get", "can i have", "let me have",
    "i'll have", "i'll take", "i will have",
    "gimme", "plz gimme",
    "order", "ordering", "i need", "bring me",
]

_GREET_SIGNALS = [
    "hello", "hey", "good morning", "good afternoon",
    "good evening", "what's up", "howdy", "sup",
]

# Bare-word greet signals (exact match only, not substring)
_GREET_EXACT = {"hi", "yo", "hii", "hiii", "helo", "helo", "heyy"}

_CONFIRM_SIGNALS = {"yes", "yeah", "yep", "yea", "sure", "ok", "okay", "alright", "yup", "confirm", "go ahead", "proceed", "that's all", "thats all", "done", "checkout"}


def _keyword_fallback(intents: List[str], user_text: str) -> List[str]:
    """Apply keyword-based fallback rules for missed intents."""
    text_lower = user_text.lower().strip()
    # strip to words for exact matching
    text_words = set(text_lower.split())
    corrected = list(intents)

    # Apply each keyword rule — if keyword fires, REPLACE wrong model predictions
    for intent, signals in _KEYWORD_RULES.items():
        if any(sig in text_lower for sig in signals):
            if intent not in corrected:
                corrected = [i for i in corrected if i == "unknown" and False or i != "unknown"]
                # avoid browse_menu overriding order_create for ordering inputs
                if intent == "browse_menu" and "order_create" in corrected:
                    continue
                corrected = [i for i in corrected if i != "unknown"]
                corrected.append(intent)
    
    # Reservation create keyword override: if 'make a reservation' but model
    # predicted reservation_cancel/refund, replace them entirely
    reserve_kw = ["reserve a table", "book a table", "make a reservation",
                  "reserve for", "table for", "book for"]
    if any(sig in text_lower for sig in reserve_kw) and "cancel" not in text_lower:
        if "reservation_create" in corrected:
            # Remove contradicting intents that the model wrongly predicted
            corrected = [i for i in corrected
                         if i not in {"reservation_cancel", "refund_request", "browse_menu"}]
            if "reservation_create" not in corrected:
                corrected.append("reservation_create")

    # Order create keywords — only if no more specific action intent
    specific_actions = {
        "order_modify", "order_cancel", "reservation_create",
        "reservation_cancel", "refund_request", "complaint",
    }
    if any(sig in text_lower for sig in _ORDER_SIGNALS):
        if not any(i in corrected for i in specific_actions):
            if "order_create" not in corrected:
                corrected = [i for i in corrected if i != "unknown"]
                corrected.append("order_create")

    # Greet keywords — substring match for phrases, exact match for short words
    is_greet = (
        any(sig in text_lower for sig in _GREET_SIGNALS)
        or bool(text_words & _GREET_EXACT)
    )
    if is_greet and "greet" not in corrected:
        # Only add greet if there's no strong ordering intent
        has_order = any(sig in text_lower for sig in _ORDER_SIGNALS)
        if not has_order:
            corrected.append("greet")

    # Remove "unknown" if we found real intents
    real = [i for i in corrected if i != "unknown"]
    return real if real else ["unknown"]


# ═══════════════════════════════════════════════════════════════════
# LAYER 5: CONFLICT RESOLUTION + PRUNING
# ═══════════════════════════════════════════════════════════════════

_OVERRIDES = {
    "ask_parking": ["ask_location", "browse_menu"],
    "refund_request": ["ask_payment_methods"],
    "order_modify": ["order_create", "refund_request"],
    "reservation_cancel": ["reservation_create"],
    "order_cancel": ["order_create", "reservation_cancel"],
    "complaint": ["order_create"],
}

_INFO_INTENTS = {
    "ask_location", "ask_hours", "ask_contact", "ask_parking",
    "ask_payment_methods", "ask_price",
}
_ACTION_INTENTS = {
    "order_create", "order_modify", "order_cancel",
    "reservation_create", "reservation_cancel",
    "complaint", "refund_request",
}

_PRIORITY = [
    "order_create", "order_modify", "order_cancel",
    "reservation_create", "reservation_cancel",
    "complaint", "refund_request",
    "greet", "browse_menu",
    "ask_price", "ask_hours", "ask_location", "ask_contact",
    "ask_parking", "ask_payment_methods",
]

_MENU_SIGNALS = ["menu", "food options", "what do you have", "what's available", "show me", "see your"]


def _conflict_resolve(intents: List[str], user_text: str) -> List[str]:
    """Resolve conflicts, prune noise, and cap at 2 intents max."""
    text_lower = user_text.lower().strip()
    corrected = list(intents)

    # Override rules: specific intent removes generic
    to_remove = set()
    for specific, generics in _OVERRIDES.items():
        if specific in corrected:
            for g in generics:
                if g in corrected:
                    to_remove.add(g)
    if to_remove:
        corrected = [i for i in corrected if i not in to_remove]

    # Greet-only pruning: greet + info (no action, no browse) → just greet
    has_greet = "greet" in corrected
    has_action = any(i in corrected for i in _ACTION_INTENTS)
    has_info = any(i in corrected for i in _INFO_INTENTS)
    has_browse = "browse_menu" in corrected

    if has_greet and has_info and not has_action and not has_browse:
        corrected = ["greet"]

    # Greet + browse: keep browse only if there are menu keywords
    if has_greet and has_browse and not has_action:
        if not any(sig in text_lower for sig in _MENU_SIGNALS):
            corrected = [i for i in corrected if i != "browse_menu"]

    # Reservation pruning: reservation_create should remove browse_menu
    if "reservation_create" in corrected:
        if "browse_menu" in corrected and len(corrected) > 1:
            # Only keep browse_menu if there's an explicit menu keyword
            if not any(sig in text_lower for sig in _MENU_SIGNALS):
                corrected = [i for i in corrected if i != "browse_menu"]

    # Order pruning
    if "order_create" in corrected:
        if "browse_menu" in corrected and len(corrected) > 1:
            corrected = [i for i in corrected if i != "browse_menu"]
        items = extract_order_items_phrase(text_lower)
        if items and "ask_price" in corrected and len(corrected) > 1:
            corrected = [i for i in corrected if i != "ask_price"]
        # Remove spurious greet when no real greeting keywords present
        if "greet" in corrected:
            if not any(text_lower.startswith(gw) or gw in text_lower for gw in _GREET_SIGNALS):
                corrected = [i for i in corrected if i != "greet"]

    # Reservation conflict
    if "reservation_create" in corrected and "reservation_cancel" in corrected:
        if "cancel" in text_lower:
            corrected = [i for i in corrected if i != "reservation_create"]
        else:
            corrected = [i for i in corrected if i != "reservation_cancel"]

    # Refund isolation
    if "refund_request" in corrected:
        if "refund" in text_lower or "money back" in text_lower:
            corrected = [i for i in corrected if i in {"refund_request", "complaint", "greet"}]
            if not corrected:
                corrected = ["refund_request"]

    # Delivery/dining fix
    if "refund_request" in corrected and any(w in text_lower for w in ("delivery", "dine in", "takeaway")):
        if "refund" not in text_lower and "money" not in text_lower:
            corrected = [i for i in corrected if i != "refund_request"]

    # Max 2 intents
    if len(corrected) > 2:
        ranked = sorted(corrected, key=lambda x: _PRIORITY.index(x) if x in _PRIORITY else 99)
        corrected = ranked[:2]

    return corrected if corrected else ["unknown"]


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API — single entry point for the production pipeline
# ═══════════════════════════════════════════════════════════════════

def postprocess_intents(
    raw_intents: List[str],
    user_text: str,
    intent_spans: Optional[List[Dict]] = None,
) -> List[str]:
    """
    Full 5-layer post-processing of raw model predictions.
    
    This is the ONLY function that app.py and chatbot_core.py need to call.
    
    Args:
        raw_intents:  list of intent labels from the BERT model
        user_text:    the original user message
        intent_spans: optional BIO span metadata with confidence scores
    
    Returns:
        list of cleaned, corrected intent labels ready for the intent engine
    """
    # Layer 1+2: confidence filter + label mapping
    if intent_spans:
        strong_spans = _confidence_filter(intent_spans)
        intents = _map_and_dedupe(strong_spans)
    else:
        # No span data — just do label mapping
        intents = []
        for intent in raw_intents:
            if intent in LABEL_MAP:
                mapped = LABEL_MAP[intent]
                if mapped is not None and mapped not in intents:
                    intents.append(mapped)
            elif intent not in intents:
                intents.append(intent)
        intents = intents if intents else ["unknown"]

    # Layer 3: slot-aware correction
    intents = _slot_aware_correction(intents, user_text)

    # Layer 4: keyword fallback
    intents = _keyword_fallback(intents, user_text)

    # Layer 5: conflict resolution + pruning
    intents = _conflict_resolve(intents, user_text)

    return intents
