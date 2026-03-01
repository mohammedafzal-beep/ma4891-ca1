from __future__ import annotations
import re
import spacy
from spacy.matcher import PhraseMatcher
from typing import Dict, List, Tuple, Any

# Load the small english model, might upgrade later if it misses too much
nlp = spacy.load("en_core_web_sm")

# ---- menu config stuff ----
# TODO: move this to a DB eventually
DINING_MODES = ["dine in", "dine-in", "takeaway", "take away", "pickup", "pick up", "delivery"]
PAYMENT_MODES = ["cash", "credit card", "debit card", "visa", "mastercard", "paynow", "apple pay", "google pay"]

MENU_ITEMS = [
    "chicken burger", "beef burger", "fries", "ramen", "pad thai", "margherita pizza",
    "pepperoni pizza", "caesar salad", "tom yum", "sushi platter", "iced latte",
    "iced tea", "coke", "sprite", "cheesecake", "brownie", "tacos", "steak"
]

# ---- partial keyword → full menu item mapping ----
# lets us match "pizza" → "margherita pizza", "burger" → "chicken burger", etc.
def _build_partial_map():
    pmap = {}
    for item in MENU_ITEMS:
        words = item.lower().split()
        for w in words:
            # skip very generic words that are part of many items
            if w in {"and", "the", "a", "with"}:
                continue
            pmap.setdefault(w, []).append(item)
    return pmap

PARTIAL_MENU_MAP = _build_partial_map()

# setup phrase matchers (makes it way faster than regex for static lists)
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("DINING_MODE", [nlp.make_doc(x) for x in DINING_MODES])
matcher.add("PAYMENT_MODE", [nlp.make_doc(x) for x in PAYMENT_MODES])
matcher.add("MENU_ITEM", [nlp.make_doc(x) for x in MENU_ITEMS])

# simple regex for grabbing digits
qty_pattern = re.compile(r"\b(\d+)\b")

def _normalize_mode(raw_text: str) -> str:
    # cleans up the matched dining mode so the backend doesn't freak out
    cleaned = raw_text.lower().strip()
    if cleaned in {"dine-in", "dine in"}:
        return "dine_in"
    if cleaned in {"takeaway", "take away", "pickup", "pick up"}:
        return "pickup"
    if cleaned == "delivery":
        return "delivery"
        
    return cleaned.replace(" ", "_")


# ── Centralized name validator ──
# Used by BOTH extract_name_ner and extract_name_direct to reject non-names
_BLOCKED_NAME_WORDS = {
    # commands and actions (3+ chars to avoid blocking initials)
    "order", "book", "reserve", "cancel", "menu", "help", "hello",
    "hey", "yes", "what", "how", "why", "when", "where", "show",
    "price", "refund", "complaint", "table", "delivery", "pickup",
    "dine", "pay", "cash", "card", "thanks", "bye", "goodbye",
    "give", "get", "want", "need", "can", "please", "make",
    "the", "your", "this", "that",
    "reservation", "checkout", "check", "out", "modify", "change",
    "remove", "delete", "add", "item", "items", "food", "drink",
    # menu items (4+ chars to avoid blocking real names like "Tom")
    "pizza", "burger", "ramen", "fries", "salad", "sushi", "steak",
    "tacos", "brownie", "cheesecake", "coke", "sprite", "latte",
    "chicken", "beef", "margherita", "pepperoni", "caesar",
    "iced", "platter",
}


def _is_valid_name(candidate: str) -> bool:
    """
    Final safety gate: reject anything that looks like a command, menu item,
    partial command (typo), or gibberish. Returns True only for plausible names.
    """
    if not candidate or not candidate.strip():
        return False
    
    c = candidate.strip().lower()
    words = c.split()
    
    # too long → not a name
    if len(words) > 3 or len(c) > 40:
        return False
    
    # check each word against blocked set AND prefix matching
    for w in words:
        if w in _BLOCKED_NAME_WORDS:
            return False
        # also check plural-stripped version (salads→salad, burgers→burger)
        w_singular = w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w
        if w_singular != w and w_singular in _BLOCKED_NAME_WORDS:
            return False
        # prefix match: "orde" → "order", "reser" → "reserve", etc.
        for check_w in {w, w_singular}:
            if len(check_w) >= 3:
                for blocked in _BLOCKED_NAME_WORDS:
                    if blocked.startswith(check_w) and 0 < len(blocked) - len(check_w) <= 3:
                        return False
    
    # must be alphabetic (with hyphens/apostrophes)
    if not all(re.match(r"^[A-Za-z][A-Za-z\-']*$", w) for w in words):
        return False
    
    # reject gibberish: 4+ chars with no vowels
    if len(c.replace(" ", "")) >= 4 and not any(ch in "aeiou" for ch in c):
        return False
    
    return True


def extract_name_ner(chat_text: str) -> str | None:
    # First try NER, but people are weird so fallback to regex for common phrases
    doc = nlp(chat_text)

    # Hardcoded check for "my name is X"
    match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\-']+)\b", chat_text, re.I)
    if match:
        name = match.group(1)
        if _is_valid_name(name):
            return name

    # Check for "i am X" (can be a bit risky but usually works)
    match2 = re.search(r"\bi am\s+([A-Za-z][A-Za-z\-']+)\b", chat_text, re.I)
    if match2:
        name = match2.group(1)
        if _is_valid_name(name):
            return name

    # Let spacy do its thing
    for entity in doc.ents:
        if entity.label_ == "PERSON":
            # Don't grab ridiculously long names, probably a false positive
            found_name = entity.text.strip()
            if 1 <= len(found_name.split()) <= 3 and _is_valid_name(found_name):
                return found_name
                
    return None


def extract_name_direct(chat_text: str) -> str | None:
    """
    Context-aware name extraction — used when the bot has explicitly asked for a name.
    More lenient than extract_name_ner: accepts short inputs.
    Uses centralized _is_valid_name() for all validation.
    """
    cleaned = chat_text.strip()
    
    # strip common prefixes like "my name is", "it's", "i'm", "call me"
    for prefix in ["my name is ", "i'm ", "im ", "it's ", "its ", "call me ", "i am "]:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    
    # use the centralized validator
    if _is_valid_name(cleaned):
        return cleaned
    
    return None

def extract_dining_mode_phrase(text: str) -> str | None:
    # look for dining modes using our phrase matcher
    parsed = nlp(text)
    found_matches = matcher(parsed)
    
    for m_id, start_idx, end_idx in found_matches:
        lbl = nlp.vocab.strings[m_id]
        if lbl == "DINING_MODE":
            raw_span = parsed[start_idx:end_idx].text
            return _normalize_mode(raw_span)
            
    return None

def extract_payment_mode_phrase(text: str) -> str | None:
    # grab the payment type
    parsed = nlp(text)
    hits = matcher(parsed)
    
    for m_id, s_idx, e_idx in hits:
        if nlp.vocab.strings[m_id] == "PAYMENT_MODE":
            p_span = parsed[s_idx:e_idx].text.lower().strip()
            
            # normalize some common ones
            if p_span in {"credit card", "debit card", "visa", "mastercard"}:
                return "card"
            if p_span == "paynow":
                return "paynow"
            if p_span == "apple pay":
                return "apple_pay"
            if p_span == "google pay":
                return "google_pay"
            if p_span == "cash":
                return "cash"
                
            return p_span.replace(" ", "_")
            
    return None

def _normalize_word(word: str) -> str:
    """Strip plurals and common suffixes to normalize a word for matching."""
    w = word.lower().strip()
    # strip trailing 's' for plurals (burgers→burger, salads→salad, fries stays fries)
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        w = w[:-1]
    # strip trailing 'es' (dishes→dish, etc.)
    if len(w) > 4 and w.endswith('es') and w[:-2] + 'es' == word.lower():
        w = w[:-2]
    return w


def _fuzzy_match_key(word: str, keys: set, max_dist: int = 1) -> str | None:
    """
    Find the closest key in the set within edit distance max_dist.
    Simple implementation for short words.
    """
    if len(word) < 3:
        return None
    
    best_key = None
    best_dist = max_dist + 1
    
    for key in keys:
        # quick length check to avoid expensive computation
        if abs(len(key) - len(word)) > max_dist:
            continue
        # compute simple edit distance (Levenshtein)
        dist = _edit_distance(word, key)
        if dist < best_dist:
            best_dist = dist
            best_key = key
    
    return best_key if best_dist <= max_dist else None


def _edit_distance(s1: str, s2: str) -> int:
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost     # replace
            ))
        prev_row = curr_row
    
    return prev_row[-1]


def _partial_match_items(text: str) -> List[Dict[str, Any]]:
    """
    Fallback: match partial menu keywords when exact PhraseMatcher fails.
    Handles: exact partial ("pizza"), plurals ("burgers"→"burger"),
    and typos ("bugers"→"burger") via fuzzy matching.
    
    When a keyword maps to MULTIPLE candidates (e.g. "pizza" → margherita, pepperoni),
    returns an ambiguous result so the bot can ask the user to clarify.
    """
    text_lower = text.lower()
    collected = []
    ambiguous = []  # track items that need clarification
    already_matched = set()
    partial_keys = set(PARTIAL_MENU_MAP.keys())

    for word in text_lower.split():
        # skip very short words and numbers
        if len(word) < 3 or word.isdigit():
            continue
        # strip punctuation
        clean_word = re.sub(r'[^a-z]', '', word)
        
        # Try matching in order: exact → normalized (plural strip) → fuzzy
        matched_key = None
        if clean_word in PARTIAL_MENU_MAP:
            matched_key = clean_word
        else:
            # try plural-stripped version
            normalized = _normalize_word(clean_word)
            if normalized in PARTIAL_MENU_MAP:
                matched_key = normalized
            else:
                # try fuzzy match (edit distance 1) for typos
                # try both original and normalized forms
                fuzzy = _fuzzy_match_key(clean_word, partial_keys, max_dist=1)
                if not fuzzy and normalized != clean_word:
                    fuzzy = _fuzzy_match_key(normalized, partial_keys, max_dist=1)
                if fuzzy:
                    matched_key = fuzzy
        
        if matched_key:
            candidates = PARTIAL_MENU_MAP[matched_key]
            # skip if we already matched one of these candidates
            unmatched = [c for c in candidates if c not in already_matched]
            if not unmatched:
                continue
            
            # look for a quantity near this word
            word_pos = text_lower.find(clean_word)
            left = max(0, word_pos - 10)
            right = min(len(text_lower), word_pos + len(clean_word) + 3)
            window = text_lower[left:right]
            q_match = qty_pattern.search(window)
            qty = int(q_match.group(1)) if q_match else 1
            
            # AMBIGUITY CHECK: if multiple items match this keyword,
            # ask the user to clarify instead of auto-selecting
            if len(unmatched) > 1:
                ambiguous.append({
                    "keyword": matched_key,
                    "options": unmatched,
                    "qty": qty,
                })
                # don't add to collected — wait for clarification
            else:
                chosen = unmatched[0]
                already_matched.add(chosen)
                collected.append({"item": chosen, "qty": qty})

    # attach ambiguity info to the result
    if ambiguous:
        # store it as metadata on the list (Python allows this)
        collected = collected  # keep any non-ambiguous items
        # return with a special marker
        for item in collected:
            item["_ambiguous"] = ambiguous
        if not collected:
            # all items were ambiguous — return a placeholder
            collected.append({"item": "__ambiguous__", "qty": 0, "_ambiguous": ambiguous})

    return collected


def extract_order_items_phrase(text: str) -> List[Dict[str, Any]]:
    # This gets hairy. We need to find the item and then look backwards to find a number.
    # If we don't find a number, just assume 1.
    parsed = nlp(text)
    
    # filter to just menu items
    menu_hits = [(m, s, e) for (m, s, e) in matcher(parsed) if nlp.vocab.strings[m] == "MENU_ITEM"]
    
    if not menu_hits:
        # ── Partial matching fallback ──
        # Try matching partial keywords like "pizza", "burger", "salad"
        partial_results = _partial_match_items(text)
        if partial_results:
            # check if any results contain ambiguity markers
            has_ambiguous = any("_ambiguous" in item for item in partial_results)
            if has_ambiguous:
                # pass through as-is so the intent handler can detect and ask for clarification
                return partial_results
            # normal case: combine items
            combined: Dict[str, int] = {}
            for item_dict in partial_results:
                it = item_dict["item"]
                combined[it] = combined.get(it, 0) + int(item_dict["qty"])
            return [{"item": k, "qty": v} for k, v in combined.items()]
        return []

    collected_items = []
    matched_spans = set()  # track matched character ranges to skip in partial matching
    
    for _, start_word, end_word in menu_hits:
        item_span = parsed[start_word:end_word]
        name_val = item_span.text.lower().strip()
        matched_spans.add(name_val)

        # Hacky context window to look for a quantity. 
        # Look back approx 12 chars, and forward 2 chars just in case.
        left_bound = max(0, item_span.start_char - 12)
        right_bound = min(len(text), item_span.end_char + 2)
        
        search_window = text[left_bound:right_bound]
        q_match = qty_pattern.search(search_window)
        
        quantity = int(q_match.group(1)) if q_match else 1
        collected_items.append({"item": name_val, "qty": quantity})
    
    # ── Also try partial matching on words NOT already matched ──
    # This catches "burger" in "give me burger and fries" (where "fries" was exact-matched)
    remaining_words = []
    for word in text.lower().split():
        clean_w = re.sub(r'[^a-z]', '', word)
        if clean_w and clean_w not in matched_spans and len(clean_w) >= 3:
            remaining_words.append(word)
    if remaining_words:
        remaining_text = " ".join(remaining_words)
        partial_extra = _partial_match_items(remaining_text)
        # only add items we didn't already find via exact match
        existing_items = {it["item"] for it in collected_items}
        for pi in partial_extra:
            if pi["item"] not in existing_items:
                collected_items.append(pi)
        
    # merge duplicate items into a single entry
    combined: Dict[str, int] = {}
    for item_dict in collected_items:
        it = item_dict["item"]
        combined[it] = combined.get(it, 0) + int(item_dict["qty"])
        
    return [{"item": k, "qty": v} for k, v in combined.items()]


# ═══════════════════════════════════════════════════════════════════
# LINGUISTIC ANALYSIS — POS/Dependency Parsing Demo
# ═══════════════════════════════════════════════════════════════════

def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """
    POS tagging + dependency parsing to extract verb/object/subject triples.
    
    This demonstrates spaCy's linguistic pipeline beyond just entity extraction.
    The professor specifically asked for POS/dep analysis in the tutorial feedback.
    
    Returns:
        dict with verbs, nouns, subjects, objects, and parsed triples
    """
    doc = nlp(text)
    
    features = {
        "tokens": [],
        "verbs": [],
        "nouns": [],
        "subjects": [],
        "objects": [],
        "triples": [],
    }
    
    for token in doc:
        features["tokens"].append({
            "text": token.text,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.text,
        })
        
        if token.pos_ == "VERB":
            features["verbs"].append(token.text)
        elif token.pos_ in ("NOUN", "PROPN"):
            features["nouns"].append(token.text)
        
        # Subject detection
        if "subj" in token.dep_:
            features["subjects"].append({
                "text": token.text,
                "verb": token.head.text,
                "dep": token.dep_,
            })
        
        # Object detection
        if "obj" in token.dep_:
            features["objects"].append({
                "text": token.text,
                "verb": token.head.text,
                "dep": token.dep_,
            })
    
    # Build subject-verb-object triples
    for subj in features["subjects"]:
        verb = subj["verb"]
        matching_objs = [o for o in features["objects"] if o["verb"] == verb]
        if matching_objs:
            for obj in matching_objs:
                features["triples"].append(f"{subj['text']} → {verb} → {obj['text']}")
        else:
            features["triples"].append(f"{subj['text']} → {verb}")
    
    return features


# ═══════════════════════════════════════════════════════════════════
# CONTEXT MEMORY — Follow-up Detection
# ═══════════════════════════════════════════════════════════════════

FOLLOWUP_SIGNALS = [
    "also", "and also", "add more", "one more", "another one",
    "same thing", "same again", "that too", "as well",
    "what about", "how about", "anything else",
    "actually", "wait", "change that", "never mind",
    "yes", "yeah", "sure", "ok", "no", "nah",
]


def detect_followup(text: str, history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect if the current message is a follow-up to a previous intent.
    
    Uses conversation history AND pending-slot context to understand:
    - "also add fries" → follow-up to previous order_create
    - "yes" / "no" → response to previous question
    - "actually change that" → follow-up to previous order
    - "mk" after bot asked for name → slot-filling response
    - "2 pizza" after bot asked for order_items → slot-filling response
    
    Args:
        text: current user message
        history: conversation history dict with 'turns' list
    
    Returns:
        dict with is_followup flag, detected context, and suggested intent
    """
    text_lower = text.lower().strip()
    turns = history.get("turns", [])
    
    result = {
        "is_followup": False,
        "followup_type": None,
        "previous_intent": None,
        "suggested_intent": None,
    }
    
    # No history → can't be a follow-up
    if not turns:
        return result
    
    # Get the last turn's intent
    last_turn = turns[-1]
    last_intents = last_turn.get("intents", [])
    last_intent = last_intents[0] if last_intents else None
    
    # ── Pending-slot context: if the bot asked for missing info, the next
    #    reply is almost certainly providing that info ──
    pending = history.get("_pending_slots", [])
    if pending and last_intent:
        # If pending name and user gave a short reply, it's a slot fill
        if "name" in pending:
            direct_name = extract_name_direct(text)
            if direct_name:
                result["is_followup"] = True
                result["followup_type"] = "slot_fill"
                result["previous_intent"] = last_intent
                result["suggested_intent"] = last_intent
                return result
        
        # If pending order_items and user mentions food, it's a slot fill
        if "order_items" in pending:
            items = extract_order_items_phrase(text)
            if items:
                result["is_followup"] = True
                result["followup_type"] = "slot_fill"
                result["previous_intent"] = last_intent
                result["suggested_intent"] = last_intent
                return result
        
        # If pending dining_mode and user mentions a mode
        if "dining_mode" in pending:
            mode = extract_dining_mode_phrase(text)
            if mode:
                result["is_followup"] = True
                result["followup_type"] = "slot_fill"
                result["previous_intent"] = last_intent
                result["suggested_intent"] = last_intent
                return result
        
        # If pending payment_mode and user mentions payment
        if "payment_mode" in pending:
            pay = extract_payment_mode_phrase(text)
            if pay:
                result["is_followup"] = True
                result["followup_type"] = "slot_fill"
                result["previous_intent"] = last_intent
                result["suggested_intent"] = last_intent
                return result
    
    # Check for follow-up signals
    has_followup_signal = any(sig in text_lower for sig in FOLLOWUP_SIGNALS)
    
    if has_followup_signal and last_intent:
        result["is_followup"] = True
        result["previous_intent"] = last_intent
        
        # "yes/yeah/sure" → affirm previous action
        if any(w in text_lower for w in ["yes", "yeah", "sure", "ok", "yep"]):
            result["followup_type"] = "affirm"
            result["suggested_intent"] = last_intent
        
        # "no/nah" → negate previous action
        elif any(w in text_lower for w in ["no", "nah", "nope"]):
            result["followup_type"] = "negate"
            if last_intent == "order_create":
                result["suggested_intent"] = "order_cancel"
            elif last_intent == "reservation_create":
                result["suggested_intent"] = "reservation_cancel"
        
        # "also/add more/one more" → continue previous action
        elif any(w in text_lower for w in ["also", "add more", "one more", "another", "as well"]):
            result["followup_type"] = "continue"
            result["suggested_intent"] = last_intent
        
        # "change/actually/wait" → modify previous action
        elif any(w in text_lower for w in ["actually", "wait", "change"]):
            result["followup_type"] = "modify"
            if last_intent in {"order_create", "order_modify"}:
                result["suggested_intent"] = "order_modify"
    
    # Short messages (< 4 words) after an order are likely follow-ups
    word_count = len(text_lower.split())
    if word_count <= 3 and last_intent == "order_create" and not result["is_followup"]:
        # Check if it contains a menu item
        items = extract_order_items_phrase(text)
        if items:
            result["is_followup"] = True
            result["followup_type"] = "continue"
            result["previous_intent"] = last_intent
            result["suggested_intent"] = "order_create"
    
    return result
