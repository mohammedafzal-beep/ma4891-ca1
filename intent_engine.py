from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import re
import uuid


# =========================
# HARDCODED KNOWLEDGE BASE
# TODO: hook this up to an actual postgres or redis DB later
# =========================
REST_KB = {
    "hours": "Mon–Sun: 10:00am–10:00pm",
    "address": "123 Example Road, Singapore",
    "contact": {"phone": "+65 6123 4567", "email": "hello@example.com"},
    "parking": "Parking available at the mall basement.",
    "payment_modes": ["cash", "card", "paynow", "apple_pay", "google_pay"],
    # full menu — synced with slots_spacy.py MENU_ITEMS so pricing always works
    "menu": {
        "chicken burger": {"price": 6.5, "category": "mains"},
        "beef burger": {"price": 7.5, "category": "mains"},
        "ramen": {"price": 8.0, "category": "mains"},
        "pad thai": {"price": 7.0, "category": "mains"},
        "margherita pizza": {"price": 9.5, "category": "mains"},
        "pepperoni pizza": {"price": 10.0, "category": "mains"},
        "caesar salad": {"price": 6.0, "category": "mains"},
        "steak": {"price": 15.0, "category": "mains"},
        "tom yum": {"price": 7.5, "category": "mains"},
        "sushi platter": {"price": 12.0, "category": "mains"},
        "tacos": {"price": 6.5, "category": "mains"},
        "fries": {"price": 3.0, "category": "sides"},
        "coke": {"price": 2.0, "category": "drinks"},
        "sprite": {"price": 2.0, "category": "drinks"},
        "iced latte": {"price": 4.5, "category": "drinks"},
        "iced tea": {"price": 3.0, "category": "drinks"},
        "cheesecake": {"price": 5.0, "category": "desserts"},
        "brownie": {"price": 4.5, "category": "desserts"},
    },
}

DINING_OPTIONS = {"dine_in", "pickup", "delivery"}
PAYMENT_OPTIONS = set(REST_KB["payment_modes"])

# basic regex for capturing IDS, nothing fancy
ORDER_REGEX = re.compile(r"\b(\d{4,})\b")
RES_REGEX = re.compile(r"\b(R\d{3,})\b", re.I)


# =========================
# Data structures
# =========================
@dataclass
class IntentPayload:
    intent_name: str
    success: bool
    
    # Text blocks we can safely show to the user 
    response_blocks: List[str]         
    
    # Internal structured data for debugging/LLM
    raw_data: Dict[str, Any]
    
    # What did we actually change in the state?
    state_updates: Dict[str, Any]
    
    # List of missing fields we still need to grab
    missing_fields: List[str]


# =========================
# State helpers
# =========================
def setup_history_if_needed(hist: Dict[str, Any]) -> None:
    # make sure the core 4 exist
    hist.setdefault("name", None)
    hist.setdefault("dining_mode", None)
    hist.setdefault("order_items", [])   # canonical order items
    hist.setdefault("payment_mode", None)

    # operational state stuff
    hist.setdefault("cart", [])          
    hist.setdefault("active_order", {"status": "none", "order_id": None})
    hist.setdefault("reservations", [])  
    hist.setdefault("session", {"mode": None})  # "ordering", "reservation", etc.

    # tracking arrays for analytical stuff
    hist.setdefault("complaints", [])
    hist.setdefault("reservation_history", [])
    hist.setdefault("order_modify_history", [])
    hist.setdefault("menu_browse_history", [])


def combine_items(base_list: List[Dict[str, Any]], new_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # merges the dictionaries so we don't have duplicate row entries
    combined = {x["item"].lower().strip(): int(x["qty"]) for x in base_list}
    
    for it in new_list:
        itm = it["item"].lower().strip()
        combined[itm] = combined.get(itm, 0) + int(it["qty"])
        
    return [{"item": k, "qty": v} for k, v in combined.items()]


def subtract_items(base_list: List[Dict[str, Any]], drop_list: List[str]) -> List[Dict[str, Any]]:
    # just filters out the items we want dropped
    drop_set = {x.lower().strip() for x in drop_list}
    return [x for x in base_list if x["item"].lower().strip() not in drop_set]


def calculate_cost(cart_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    sum_total = 0.0
    missing = []
    active_menu = REST_KB["menu"]
    
    for item_dict in cart_items:
        n = item_dict["item"].lower().strip()
        q = int(item_dict["qty"])
        
        if n not in active_menu:
            missing.append(n)
            continue
            
        sum_total += float(active_menu[n]["price"]) * q
        
    return {"ok": len(missing) == 0, "total": round(sum_total, 2), "unknown_items": missing}


def check_missing_core(hist: Dict[str, Any]) -> List[str]:
    # What are we missing before we can checkout?
    miss = []
    if not hist.get("name"):
        miss.append("name")
    if hist.get("dining_mode") not in DINING_OPTIONS:
        miss.append("dining_mode")
    if not hist.get("order_items"):
        miss.append("order_items")
    if hist.get("payment_mode") not in PAYMENT_OPTIONS:
        miss.append("payment_mode")
    return miss


# =========================
# Fallback parsers (in case spaCy completely whiffs it)
# =========================
def parse_items_fallback(raw_val: str) -> List[Dict[str, Any]]:
    lower_val = raw_val.lower()
    # sort by length so "beef burger" matches before "burger"
    sorted_menu = sorted(REST_KB["menu"].keys(), key=len, reverse=True)
    
    results = []
    for m_item in sorted_menu:
        loc = lower_val.find(m_item)
        if loc == -1:
            continue
            
        # look back to see if there's a number
        left_bound = max(0, loc - 6)
        slice_window = lower_val[left_bound:loc]
        number_search = re.search(r"(\d+)\s*$", slice_window)
        found_qty = int(number_search.group(1)) if number_search else 1
        
        results.append({"item": m_item, "qty": found_qty})
    
    # ── Partial keyword matching fallback ──
    # If exact names didn't match, try partial keywords like "pizza" → "margherita pizza"
    # Also handles plurals ("burgers"→"burger") and typos ("bugers"→"burger")
    if not results:
        already_matched = set()
        for word in lower_val.split():
            clean_word = re.sub(r'[^a-z]', '', word)
            if len(clean_word) < 3:
                continue
            
            # generate variants: original, plural-stripped
            variants = [clean_word]
            # strip trailing 's' for plurals
            if len(clean_word) > 3 and clean_word.endswith('s') and not clean_word.endswith('ss'):
                variants.append(clean_word[:-1])
            
            matched = False
            for variant in variants:
                if matched:
                    break
                for m_item in sorted_menu:
                    if variant in m_item and m_item not in already_matched:
                        already_matched.add(m_item)
                        word_pos = lower_val.find(clean_word)
                        left = max(0, word_pos - 8)
                        window = lower_val[left:word_pos]
                        num_match = re.search(r"(\d+)\s*$", window)
                        qty = int(num_match.group(1)) if num_match else 1
                        results.append({"item": m_item, "qty": qty})
                        matched = True
                        break
            
            # fuzzy fallback: try edit distance 1 against menu item words
            if not matched:
                for m_item in sorted_menu:
                    if m_item in already_matched:
                        continue
                    for m_word in m_item.split():
                        if abs(len(m_word) - len(clean_word)) <= 1 and len(clean_word) >= 4:
                            # simple edit distance check
                            dist = sum(c1 != c2 for c1, c2 in zip(clean_word, m_word))
                            dist += abs(len(clean_word) - len(m_word))
                            if dist <= 1:
                                already_matched.add(m_item)
                                word_pos = lower_val.find(clean_word)
                                left = max(0, word_pos - 8)
                                window = lower_val[left:word_pos]
                                num_match = re.search(r"(\d+)\s*$", window)
                                qty = int(num_match.group(1)) if num_match else 1
                                results.append({"item": m_item, "qty": qty})
                                matched = True
                                break
                    if matched:
                        break
        
    return combine_items([], results)


def parse_removals_fallback(raw_val: str) -> List[str]:
    lower_val = raw_val.lower()
    # quick check to see if it even sounds like a removal
    if not any(k in lower_val for k in ["remove", "delete", "no ", "cancel "]):
        return []
        
    found = []
    for m_item in REST_KB["menu"].keys():
        if m_item in lower_val:
            found.append(m_item)
    return found


def extract_order_id(raw_val: str) -> Optional[str]:
    match = ORDER_REGEX.search(raw_val)
    return match.group(1) if match else None


def extract_res_id(raw_val: str) -> Optional[str]:
    match = RES_REGEX.search(raw_val)
    return match.group(1).upper() if match else None


# =========================================================
# INTENT HANDLERS
# These actually do the work of updating the state
# =========================================================

def handle_greet(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    hist["session"]["mode"] = hist["session"]["mode"] or "chat"
    
    # If we already know the user's name, don't re-ask
    existing_name = hist.get("name")
    if existing_name:
        return IntentPayload(
            intent_name="greet",
            success=True,
            response_blocks=[f"Welcome back, {existing_name}! How can I help — order, reservation, or menu questions?"],
            raw_data={},
            state_updates={"session.mode": hist["session"]["mode"]},
            missing_fields=[],
        )
    
    return IntentPayload(
        intent_name="greet",
        success=True,
        response_blocks=["Hi! Welcome to Sakura Grill 🌸. May I have your name?"],
        raw_data={},
        state_updates={"session.mode": hist["session"]["mode"]},
        missing_fields=["name"],
    )

def handle_hours(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    return IntentPayload(
        intent_name="hours",
        success=True,
        response_blocks=[f"Our opening hours: {REST_KB['hours']}"],
        raw_data={"hours": REST_KB["hours"]},
        state_updates={},
        missing_fields=[],
    )

def handle_location(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    return IntentPayload(
        intent_name="location",
        success=True,
        response_blocks=[f"Our address: {REST_KB['address']}"],
        raw_data={"address": REST_KB["address"]},
        state_updates={},
        missing_fields=[],
    )

def handle_contact(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    c_info = REST_KB["contact"]
    return IntentPayload(
        intent_name="contact",
        success=True,
        response_blocks=[f"Contact: {c_info['phone']} | {c_info['email']}"],
        raw_data={"contact": c_info},
        state_updates={},
        missing_fields=[],
    )

def handle_parking(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    return IntentPayload(
        intent_name="parking",
        success=True,
        response_blocks=[REST_KB["parking"]],
        raw_data={"parking": REST_KB["parking"]},
        state_updates={},
        missing_fields=[],
    )

def handle_payment_methods(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    modes_str = ", ".join(REST_KB["payment_modes"])
    return IntentPayload(
        intent_name="payment_methods",
        success=True,
        response_blocks=[f"We accept: {modes_str}."],
        raw_data={"payment_modes": REST_KB["payment_modes"]},
        state_updates={},
        missing_fields=[],
    )

def handle_browse_menu(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    hist["menu_browse_history"].append(text)

    active_menu = REST_KB["menu"]
    cat_filter = slots.get("category")

    if cat_filter:
        found_items = [{"item": k, "price": v["price"]} for k, v in active_menu.items() if v.get("category") == cat_filter]
        
        if not found_items:
            return IntentPayload(
                "browse_menu", False,
                [f"I don't have anything under '{cat_filter}' right now."],
                {"category": cat_filter, "items": []},
                {"menu_browse_history+1": True},
                [],
            )
            
        print_lines = [f"{x['item']} — ${x['price']}" for x in found_items]
        return IntentPayload(
            "browse_menu", True,
            [f"Here's our {cat_filter}:", *print_lines],
            {"category": cat_filter, "items": found_items},
            {"menu_browse_history+1": True},
            [],
        )

    # fallback to just showing all categories and a quick preview list
    category_counts: Dict[str, int] = {}
    for k, v in active_menu.items():
        c_name = v.get("category", "other")
        category_counts[c_name] = category_counts.get(c_name, 0) + 1

    preview_list = list(active_menu.items())[:8]
    preview_str = [f"{k} — ${v['price']} ({v.get('category','other')})" for k, v in preview_list]
    cat_str = ", ".join([f"{c}({n})" for c, n in category_counts.items()])

    return IntentPayload(
        "browse_menu", True,
        [f"Menu categories: {cat_str}", "Preview:", *preview_str, "Tell me a category (like 'drinks') or an item name."],
        {"categories": category_counts, "preview": preview_list},
        {"menu_browse_history+1": True},
        [],
    )


def handle_ask_price(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    # try spacy first, then fallback
    target_items = slots.get("order_items") or parse_items_fallback(text)
    
    if not target_items:
        return IntentPayload(
            "ask_price", False,
            ["Which item's price do you want? (Example: 'price of ramen')"],
            {}, {}, ["order_items"],
        )

    found_prices = {}
    unknowns = []
    
    for it in target_items:
        n = it["item"].lower().strip()
        if n in REST_KB["menu"]:
            found_prices[n] = REST_KB["menu"][n]["price"]
        else:
            unknowns.append(n)

    out_blocks = []
    for name, p in found_prices.items():
        out_blocks.append(f"{name} — ${p}")
        
    if unknowns:
        out_blocks.append(f"I couldn't find: {', '.join(unknowns)}")

    return IntentPayload(
        "ask_price", True,
        out_blocks if out_blocks else ["Tell me the item name so I can look it up."],
        {"prices": found_prices, "unknown_items": unknowns},
        {}, [],
    )


def handle_order_create(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    hist["session"]["mode"] = "ordering"

    target_items = slots.get("order_items") or parse_items_fallback(text)
    if not target_items:
        return IntentPayload(
            "order_create", False,
            ["What would you like to order? Please include item name and quantity (like '2 ramen')."],
            {},
            {"session.mode": "ordering"},
            ["order_items"],
        )

    # ── Handle ambiguous items ──
    # If the partial matcher found ambiguous matches, ask user to clarify
    ambiguous_info = None
    for item in target_items:
        if isinstance(item, dict) and "_ambiguous" in item:
            ambiguous_info = item["_ambiguous"]
            break
    
    if ambiguous_info:
        # Filter out the placeholder items
        real_items = [i for i in target_items if i.get("item") != "__ambiguous__" and "_ambiguous" not in i]
        
        # Build clarification prompts
        clarify_blocks = []
        for amb in ambiguous_info:
            options_str = " or ".join([f"**{opt}**" for opt in amb["options"]])
            clarify_blocks.append(f"Which {amb['keyword']} would you like? We have {options_str}.")
        
        # If we also got some non-ambiguous items, add those to cart
        if real_items:
            hist["cart"] = combine_items(hist["cart"], real_items)
            hist["order_items"] = hist["cart"]
            cost_info = calculate_cost(hist["order_items"])
            out_blocks = ["Added to your cart:"]
            for it in real_items:
                out_blocks.append(f"- {it['qty']} × {it['item']}")
            out_blocks.append(f"Current total: ${cost_info['total']}")
            out_blocks.extend(clarify_blocks)
        else:
            out_blocks = clarify_blocks
        
        return IntentPayload(
            "order_create", True, out_blocks,
            {"clarification_needed": ambiguous_info},
            {"session.mode": "ordering"},
            ["order_items"],
        )

    hist["cart"] = combine_items(hist["cart"], target_items)
    hist["order_items"] = hist["cart"]
    hist["active_order"]["status"] = "building"
    
    cost_info = calculate_cost(hist["order_items"])

    out_blocks = ["Added to your cart:"]
    for it in hist["cart"]:
        out_blocks.append(f"- {it['qty']} × {it['item']}")
        
    if cost_info["unknown_items"]:
        out_blocks.append(f"I couldn't price: {', '.join(cost_info['unknown_items'])}. Check spelling?")
    else:
        out_blocks.append(f"Current total: ${cost_info['total']}")

    return IntentPayload(
        "order_create", True, out_blocks,
        {"cart": hist["cart"], "total": cost_info},
        {"cart": hist["cart"], "session.mode": "ordering", "active_order.status": "building"},
        [],
    )


def handle_order_modify(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    hist["order_modify_history"].append(text)

    # figure out if we are dropping or adding
    drop_list = slots.get("remove_items") or parse_removals_fallback(text)
    add_list = slots.get("order_items") or ([] if drop_list else parse_items_fallback(text))

    did_something = False
    
    if drop_list:
        hist["cart"] = subtract_items(hist["cart"], drop_list)
        did_something = True
        
    if add_list:
        hist["cart"] = combine_items(hist["cart"], add_list)
        did_something = True

    hist["order_items"] = hist["cart"]
    cost_info = calculate_cost(hist["order_items"])

    if not did_something:
        return IntentPayload(
            "order_modify", False,
            ["What do you want to change? Example: 'remove fries' or 'add 1 coke'."],
            {"cart": hist["cart"], "total": cost_info}, {}, ["order_items"],
        )

    out_blocks = ["Updated cart:"]
    if not hist["cart"]:
        out_blocks.append("(cart is empty)")
    else:
        for it in hist["cart"]:
            out_blocks.append(f"- {it['qty']} × {it['item']}")
            
    if cost_info["unknown_items"]:
        out_blocks.append(f"I couldn't price: {', '.join(cost_info['unknown_items'])}.")
    else:
        out_blocks.append(f"Current total: ${cost_info['total']}")

    return IntentPayload(
        "order_modify", True, out_blocks,
        {"removed": drop_list, "added": add_list, "total": cost_info},
        {"cart": hist["cart"]}, [],
    )


def handle_order_cancel(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    found_id = extract_order_id(text)

    # wipe the cart
    hist["cart"] = []
    hist["order_items"] = []
    hist["active_order"]["status"] = "cancel_requested"
    if found_id:
        hist["active_order"]["order_id"] = found_id

    return IntentPayload(
        "order_cancel", True,
        ["Okay — I've cancelled the order request. If you meant a specific order ID, let me know." if not found_id else f"Cancellation requested for order {found_id}."],
        {"order_id": found_id},
        {"cart": [], "active_order.status": "cancel_requested"}, [],
    )


def handle_reservation_create(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    hist["session"]["mode"] = "reservation"

    # just build a random hex ID for now
    r_id = f"R{uuid.uuid4().hex[:6].upper()}"
    res_record = {"reservation_id": r_id, "status": "requested", "raw_text": text}
    
    hist["reservations"].append(res_record)
    hist["reservation_history"].append({"type": "create", "reservation_id": r_id, "text": text})

    missing = []
    if not hist.get("name"):
        missing.append("name")

    out_blocks = [f"Reservation request created (ID: {r_id})."]
    if missing:
        out_blocks.append("I still need: " + ", ".join(missing) + ".")

    return IntentPayload(
        "reservation_create", True, out_blocks,
        {"reservation": res_record},
        {"session.mode": "reservation", "reservation_id": r_id},
        missing,
    )


def handle_reservation_modify(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    r_id = extract_res_id(text)
    
    # if they didn't provide an ID, assume they mean the last one they made
    if not r_id and hist["reservations"]:
        r_id = hist["reservations"][-1]["reservation_id"]

    if not r_id:
        return IntentPayload(
            "reservation_modify", False,
            ["Which reservation should I change? Please give the reservation ID (like R1234)."],
            {}, {}, ["reservation_id"],
        )

    hist["reservation_history"].append({"type": "modify", "reservation_id": r_id, "text": text})
    return IntentPayload(
        "reservation_modify", True,
        [f"Reservation modification requested for {r_id}. Tell me what to change (time/date/people)."],
        {"reservation_id": r_id}, {}, [],
    )


def handle_reservation_cancel(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    r_id = extract_res_id(text)
    
    if not r_id and hist["reservations"]:
        r_id = hist["reservations"][-1]["reservation_id"]

    hist["reservation_history"].append({"type": "cancel", "reservation_id": r_id, "text": text})
    
    msg = f"Cancellation requested{'' if r_id else ' (no ID provided)'}." if not r_id else f"Cancellation requested for reservation {r_id}."
    return IntentPayload(
        "reservation_cancel", True, [msg],
        {"reservation_id": r_id}, {}, [],
    )


def handle_complaint(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    hist["complaints"].append(text)
    
    return IntentPayload(
        "complaint", True,
        ["Sorry about that. I've logged your complaint. If you have an order ID, share it so we can investigate faster."],
        {"complaint": text},
        {"complaints+1": True}, [],
    )


def handle_refund_request(text: str, hist: Dict[str, Any], slots: Dict[str, Any]) -> IntentPayload:
    setup_history_if_needed(hist)
    o_id = extract_order_id(text) or hist["active_order"].get("order_id")
    
    if not o_id:
        return IntentPayload(
            "refund_request", False,
            ["I can help with a refund — what's your order ID?"],
            {}, {}, ["order_id"],
        )
        
    return IntentPayload(
        "refund_request", True,
        [f"Refund request noted for order {o_id}. I'll need a reason and payment method used."],
        {"order_id": o_id}, {}, [],
    )


# =========================
# Dispatch Dictionary
# Maps intent strings to the handler functions
# =========================
INTENT_ROUTER = {
    "greet": handle_greet,
    "hours": handle_hours,
    "location": handle_location,
    "contact": handle_contact,
    "parking": handle_parking,
    "payment_methods": handle_payment_methods,
    "browse_menu": handle_browse_menu,
    "ask_price": handle_ask_price,
    "order_create": handle_order_create,
    "order_modify": handle_order_modify,
    "order_cancel": handle_order_cancel,
    "reservation_create": handle_reservation_create,
    "reservation_modify": handle_reservation_modify,
    "reservation_cancel": handle_reservation_cancel,
    "complaint": handle_complaint,
    "refund_request": handle_refund_request,
}


def run_intents(user_text: str, intents: List[str], history: Dict[str, Any], slots: Dict[str, Any]) -> Dict[str, Any]:
    setup_history_if_needed(history)
    
    processed_outputs: List[IntentPayload] = []
    unknown_hits = []

    # Heuristic: Prevent chaotic multiple outputs if model over-predicts
    critical_intents = [
        "order_create", "order_modify", "order_cancel", 
        "reservation_create", "reservation_modify", "reservation_cancel",
        "refund_request", "complaint"
    ]
    
    # If the model predicts multiple critical state-mutating actions, pick the strongest
    found_critical = [i for i in intents if i in critical_intents]
    if len(found_critical) > 1:
        for p in critical_intents:
            if p in found_critical:
                intents = [i for i in intents if i not in critical_intents or i == p]
                break

    # loop through all predicted intents and fire off their handlers
    for i_name in intents:
        handler_fn = INTENT_ROUTER.get(i_name)
        if not handler_fn:
            unknown_hits.append(i_name)
            continue
            
        processed_outputs.append(handler_fn(user_text, history, slots))

    # smash the response blocks together so the frontend or LLM can just read it out
    visible_blocks: List[str] = []
    needed_fields: List[str] = []
    
    for payload in processed_outputs:
        visible_blocks.extend(payload.response_blocks)
        for req in payload.missing_fields:
            if req not in needed_fields:
                needed_fields.append(req)

    return {
        "intent_outputs": [asdict(p) for p in processed_outputs],
        "unknown_intents": unknown_hits,
        "combined_response_blocks": visible_blocks,
        "need": needed_fields,
        "core_missing": check_missing_core(history),
        
        # quick snapshot of what we know right now
        "history_snapshot": {
            "name": history.get("name"),
            "dining_mode": history.get("dining_mode"),
            "order_items": history.get("order_items"),
            "payment_mode": history.get("payment_mode"),
        },
    }
