from typing import Dict, Any, List, Optional

from slots_spacy import (
    extract_name_ner,
    extract_name_direct,
    extract_dining_mode_phrase,
    extract_order_items_phrase,
    extract_payment_mode_phrase,
    extract_linguistic_features,
    detect_followup,
)
from intent_engine import run_intents
from phi_llm import PhiChat, build_system_instruction, build_user_payload

from spell_correct import correct_input

def init_history() -> Dict[str, Any]:
    # sets up the initial empty state for a new user
    return {
        "name": None,
        "dining_mode": None,
        "order_items": [],
        "payment_mode": None,

        # extra operational stuff
        "cart": [],
        "active_order": {"status": "none", "order_id": None},
        "reservations": [],
        "session": {"mode": None},

        # track what they've been doing
        "complaints": [],
        "reservation_history": [],
        "order_modify_history": [],
        "menu_browse_history": [],

        "turns": []
    }

def merge_order_items(current_items: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # pretty basic logic to combine cart items so we don't have 10 separate entries for 'coke'
    lookup_dict = {i["item"]: int(i["qty"]) for i in current_items}
    
    for it in new_items:
        itm_name = it["item"]
        lookup_dict[itm_name] = lookup_dict.get(itm_name, 0) + int(it["qty"])
        
    return [{"item": k, "qty": v} for k, v in lookup_dict.items()]

def process_turn(
    user_text: str,
    intents: List[str],
    history: Dict[str, Any],
    intent_spans: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[PhiChat] = None,
) -> str:
    try:
        return _process_turn_inner(user_text, intents, history, intent_spans, llm)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [ERROR] process_turn crashed: {e}")
        return "I'm sorry, I had trouble understanding that. Could you try rephrasing? You can order food, make a reservation, or ask about our menu."

def _process_turn_inner(
    user_text: str,
    intents: List[str],
    history: Dict[str, Any],
    intent_spans: Optional[List[Dict[str, Any]]] = None,
    llm: Optional[PhiChat] = None,
) -> str:
    
    # ── Input normalization ──
    import re as _re
    user_text = user_text.strip()
    user_text = _re.sub(r'\s+', ' ', user_text)  # collapse multiple spaces
    user_text = user_text[:500]  # cap length to prevent abuse
    if not user_text:
        return "I didn't catch that. Could you try saying something? 😊"
    
    # ── Global spell correction ──
    # Auto-correct garbled words BEFORE any processing
    # BUT: skip correction when the bot is waiting for a NAME —
    # names are unpredictable and should never be vocabulary-corrected
    # (otherwise "Sarah" → "salad", "Marco" → "margherita", etc.)
    original_text = user_text
    pending = history.get("_pending_slots", [])
    if "name" not in pending:
        user_text = correct_input(user_text)
    
    # ── Correction commands ──
    text_lower = user_text.lower().strip()
    
    # "change my name" / "wrong name" / "fix name"
    if any(p in text_lower for p in ["change my name", "wrong name", "fix my name", "update name", "correct name", "rename"]):
        history["name"] = None
        history["_pending_slots"] = ["name"]
        return "No problem! What's your correct name?"
    
    # "start over" / "reset" / "new order"  
    if any(p in text_lower for p in ["start over", "start again", "reset", "new order", "clear everything", "begin again"]):
        # preserve only model state, reset conversation
        for key in ["name", "dining_mode", "payment_mode"]:
            history[key] = None
        history["order_items"] = []
        history["cart"] = []
        history["active_order"] = {"status": "none", "order_id": None}
        history["session"] = {"mode": None}
        history["_pending_slots"] = []
        history["turns"] = []
        return "All cleared! Let's start fresh. 🔄 What's your name?"
    
    # "undo" / "remove last"
    if any(p in text_lower for p in ["undo", "remove last", "take back", "go back"]):
        cart = history.get("cart", [])
        if cart:
            removed = cart.pop()
            history["order_items"] = list(cart)
            return f"Removed **{removed['qty']}× {removed['item']}** from your cart. {'Your cart is now empty.' if not cart else 'Anything else?'}"
        else:
            return "Nothing to undo — your cart is empty."
    
    found_name = extract_name_ner(user_text)
    
    # ── Context-aware name capture ──
    # If the bot asked for a name last turn and NER missed it,
    # use the more lenient extract_name_direct
    pending = history.get("_pending_slots", [])
    if not found_name and "name" in pending:
        found_name = extract_name_direct(user_text)
    
    if found_name:
        history["name"] = found_name

    found_dining_mode = extract_dining_mode_phrase(user_text)
    if found_dining_mode:
        history["dining_mode"] = found_dining_mode

    # DO NOT mutate the cart here! Just extract the slots.
    # The Intent Engine handles actual cart mutation to prevent double counting.
    found_items = extract_order_items_phrase(user_text)

    found_payment = extract_payment_mode_phrase(user_text)
    if found_payment:
        history["payment_mode"] = found_payment

    # only pass along what we *just* extracted, not the whole history
    current_slots = {
        "name": found_name,
        "dining_mode": found_dining_mode,
        "order_items": found_items,
        "payment_mode": found_payment,
    }

    # ── Context Memory: follow-up detection ──
    followup = detect_followup(user_text, history)
    if followup["is_followup"] and followup["suggested_intent"]:
        # If the model didn't detect the follow-up intent, inject it
        if followup["suggested_intent"] not in intents:
            intents = list(intents)  # don't mutate the original
            intents.append(followup["suggested_intent"])

    # ── Linguistic Analysis: POS/dep parsing ──
    ling = extract_linguistic_features(user_text)

    # ── Terminal logging for demo visibility ──
    print(f"\n{'='*60}")
    print(f"  USER: {user_text}")
    print(f"  INTENTS: {intents}")
    print(f"  SLOTS: name={found_name}, dining={found_dining_mode}, items={found_items}, pay={found_payment}")
    if followup["is_followup"]:
        print(f"  CONTEXT: follow-up ({followup['followup_type']}) -> {followup['suggested_intent']}")
    if ling["triples"]:
        triples_ascii = [t.replace('\u2192', '->') for t in ling['triples']]
        print(f"  LINGUISTIC: {', '.join(triples_ascii)}")

    # 2. Fire up the intent engine to actually do the mutations
    engine_results = run_intents(
        user_text=user_text,
        intents=intents,
        history=history,
        slots=current_slots,
    )

    # 3. Save to log
    history["turns"].append({
        "user": user_text,
        "intents": intents,
        "intent_spans": intent_spans,
        "slots": current_slots,
        "engine_out": engine_results,
    })

    # ── Save pending slots from the engine ──
    explicit_asks = engine_results.get("need", [])

    # 4. Generate the final reply
    if llm is None:
        llm = PhiChat() # just in case it wasn't passed

    sys_prompt = build_system_instruction(history)

    # cram the engine output into the payload so the LLM knows what happened
    payload = build_user_payload(user_text, intents, intent_spans, history)
    payload += "\n\nENGINE_OUT:\n" + str(engine_results)

    final_reply = llm.reply(system=sys_prompt, user=payload)

    # ── Detect what the bot's response ACTUALLY asked for ──
    # This catches cases where the LLM fallback asks for name/items/etc.
    # even if the engine didn't explicitly set 'need'
    reply_lower = final_reply.lower()
    response_asks = list(explicit_asks)  # start with engine needs
    if any(phrase in reply_lower for phrase in ["your name", "telling me your name", "tell me your name", "what is your name", "provide your name"]):
        if "name" not in response_asks:
            response_asks.append("name")
    if any(phrase in reply_lower for phrase in ["what would you like to order", "order_items", "provide your: order"]):
        if "order_items" not in response_asks:
            response_asks.append("order_items")
    if any(phrase in reply_lower for phrase in ["dine in", "takeaway", "delivery", "dining_mode", "dining mode"]):
        if "dining_mode" not in response_asks:
            response_asks.append("dining_mode")
    if any(phrase in reply_lower for phrase in ["how would you like to pay", "payment_mode", "payment method"]):
        if "payment_mode" not in response_asks:
            response_asks.append("payment_mode")
    
    history["_pending_slots"] = response_asks

    # ── Terminal logging: bot reply ──
    safe_reply = final_reply[:150].encode('ascii', 'replace').decode('ascii')
    print(f"  BOT: {safe_reply}{'...' if len(final_reply) > 150 else ''}")
    print(f"{'='*60}")

    return final_reply

