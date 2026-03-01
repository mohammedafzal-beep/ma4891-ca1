import argparse
import json
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


# -----------------------
# Vocab (edit as needed)
# -----------------------
DISHES = [
    "chicken burger", "beef burger", "fries", "ramen", "pad thai", "margherita pizza",
    "pepperoni pizza", "caesar salad", "tom yum", "sushi platter", "iced latte",
    "iced tea", "coke", "sprite", "cheesecake", "brownie", "tacos", "steak"
]
CATEGORIES = ["starters", "mains", "desserts", "drinks", "pizza", "pasta", "noodles", "brunch", "kids meals", "soups"]
DIETS = ["halal", "vegetarian", "vegan", "gluten-free"]
ALLERGENS = ["peanuts", "eggs", "milk", "shrimp", "sesame", "soy", "tree nuts"]
BRANCHES = ["jurong", "tampines", "woodlands", "city", "orchard"]
PAYMENTS = ["cash", "credit card", "debit card", "visa", "mastercard", "apple pay", "google pay", "paynow"]
DATES = ["today", "tonight", "tomorrow", "friday", "saturday", "sunday", "next week"]
TIMES = ["6pm", "6:30pm", "7pm", "7:30pm", "8pm", "8:30pm", "9pm", "9:30pm", "10pm", "18:45"]
NAMES = ["amir", "sara", "ali", "noor", "daniel", "maria", "jun", "wei", "fatimah"]
ORDER_IDS = ["18492", "55002", "7711", "9911", "9988", "8844"]
RES_IDS = ["R1029", "R4431", "R5500", "R7788"]


GREET = ["hi", "hello", "hey", "yo", "good morning", "good evening"]
CONNECTORS = [" and ", ", and ", " also ", "; "]

INTENT_SET = [
    "greet", "hours", "location", "contact", "parking",
    "browse_menu", "ask_item_info", "ask_price",
    "dietary_request",
    "reservation_create", "reservation_modify", "reservation_cancel",
    "waitlist_join", "waitlist_status",
    "order_create", "order_modify", "order_cancel",
    "delivery_status", "pickup_instructions",
    "payment_methods", "refund_request",
    "complaint", "compliment",
    "repeat", "clarify",
]


# -----------------------
# Helpers
# -----------------------
def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def choice(rng: random.Random, xs: List[str]) -> str:
    return xs[rng.randrange(len(xs))]

def maybe(rng: random.Random, p: float) -> bool:
    return rng.random() < p

def entity(t: str, v: str) -> Dict[str, str]:
    return {"type": t, "value": v}

def rand_people(rng: random.Random) -> str:
    n = rng.randint(1, 10)
    return f"{n} pax" if maybe(rng, 0.25) else str(n)

def rand_qty(rng: random.Random) -> str:
    return str(rng.randint(1, 4))


@dataclass
class Clause:
    intent: str
    text: str
    entities: List[Dict[str, str]]


# -----------------------
# Clause templates per intent
# Each clause is self-contained so spans are meaningful.
# -----------------------
def clause_greet(rng: random.Random) -> Clause:
    g = choice(rng, GREET)
    # Keep greet clause short so span is clean
    g = g.rstrip("!?.,")
    return Clause("greet", g, [])

def clause_hours(rng: random.Random) -> Clause:
    t = choice(rng, [
        "are you open now?",
        "what are your opening hours?",
        "what time do you close tonight?",
        "opening hours on sunday?",
        "what time do you close?",
    ])
    ents = []
    if "sunday" in t:
        ents.append(entity("date", "sunday"))
    if "tonight" in t:
        ents.append(entity("date", "tonight"))
    return Clause("hours", t, ents)

def clause_location(rng: random.Random) -> Clause:
    if maybe(rng, 0.4):
        b = choice(rng, BRANCHES)
        t = choice(rng, [
            f"where is your {b} branch?",
            f"how do i get to the {b} branch?",
            f"what's the address for {b}?",
        ])
        return Clause("location", t, [entity("branch", b)])
    t = choice(rng, ["what's your address?", "where are you located?", "send me the location"])
    return Clause("location", t, [])

def clause_contact(rng: random.Random) -> Clause:
    t = choice(rng, ["what's your phone number?", "how can i contact you?", "do you have a contact number?"])
    return Clause("contact", t, [])

def clause_parking(rng: random.Random) -> Clause:
    t = choice(rng, ["do you have parking?", "is there free parking?", "parking available near you?"])
    return Clause("parking", t, [])

def clause_browse_menu(rng: random.Random) -> Clause:
    if maybe(rng, 0.6):
        cat = choice(rng, CATEGORIES)
        t = choice(rng, [f"show me {cat}", f"what {cat} do you have?", f"menu for {cat} please"])
        return Clause("browse_menu", t, [entity("category", cat)])
    t = choice(rng, ["show me the menu", "can i see the full menu?", "what do you have today?"])
    return Clause("browse_menu", t, [])

def clause_ask_item_info(rng: random.Random) -> Clause:
    dish = choice(rng, DISHES)
    t = choice(rng, [f"tell me about the {dish}", f"what's in the {dish}?", f"is the {dish} spicy?"])
    return Clause("ask_item_info", t, [entity("dish", dish)])

def clause_ask_price(rng: random.Random) -> Clause:
    dish = choice(rng, DISHES)
    t = choice(rng, [f"how much is the {dish}?", f"what's the price of {dish}?", f"cost for {dish}?"])
    return Clause("ask_price", t, [entity("dish", dish)])

def clause_dietary_request(rng: random.Random) -> Clause:
    if maybe(rng, 0.5):
        diet = choice(rng, DIETS)
        t = choice(rng, [f"do you have {diet} options?", f"i need {diet} food", f"anything {diet}?"])
        return Clause("dietary_request", t, [entity("diet", diet)])
    allergen = choice(rng, ALLERGENS)
    t = choice(rng, [f"i'm allergic to {allergen}", f"no {allergen} please", f"does anything contain {allergen}?"])
    return Clause("dietary_request", t, [entity("allergen", allergen)])

def clause_reservation_create(rng: random.Random) -> Clause:
    people = rand_people(rng)
    date = choice(rng, DATES)
    time = choice(rng, TIMES)
    if maybe(rng, 0.4):
        name = choice(rng, NAMES)
        t = f"book a table for {people} on {date} at {time} under {name}"
        ents = [entity("people", people), entity("date", date), entity("time", time), entity("name", name)]
    else:
        t = f"reserve a table for {people} on {date} at {time}"
        ents = [entity("people", people), entity("date", date), entity("time", time)]
    return Clause("reservation_create", t, ents)

def clause_reservation_modify(rng: random.Random) -> Clause:
    rid = choice(rng, RES_IDS) if maybe(rng, 0.5) else None
    time = choice(rng, TIMES)
    date = choice(rng, DATES) if maybe(rng, 0.5) else None
    people = rand_people(rng) if maybe(rng, 0.5) else None
    parts = []
    ents = []
    if rid:
        parts.append(f"change reservation {rid}")
        ents.append(entity("reservation_id", rid))
    else:
        parts.append("change my reservation")
    if date:
        parts.append(f"to {date}")
        ents.append(entity("date", date))
    parts.append(f"at {time}")
    ents.append(entity("time", time))
    if people:
        parts.append(f"for {people} people")
        ents.append(entity("people", people))
    t = " ".join(parts)
    return Clause("reservation_modify", t, ents)

def clause_reservation_cancel(rng: random.Random) -> Clause:
    if maybe(rng, 0.5):
        rid = choice(rng, RES_IDS)
        t = f"cancel reservation {rid}"
        return Clause("reservation_cancel", t, [entity("reservation_id", rid)])
    date = choice(rng, ["tonight", "tomorrow", "today"])
    t = f"cancel my reservation for {date}"
    return Clause("reservation_cancel", t, [entity("date", date)])

def clause_waitlist_join(rng: random.Random) -> Clause:
    people = rand_people(rng)
    t = choice(rng, [f"add me to the waitlist for {people}", f"join waitlist, party of {people}", f"waitlist for {people} please"])
    return Clause("waitlist_join", t, [entity("people", people)])

def clause_waitlist_status(rng: random.Random) -> Clause:
    people = rand_people(rng) if maybe(rng, 0.5) else None
    if people:
        t = f"what's the wait time for {people}?"
        return Clause("waitlist_status", t, [entity("people", people)])
    t = "what's the current wait time?"
    return Clause("waitlist_status", t, [])

def clause_order_create(rng: random.Random) -> Clause:
    dish = choice(rng, DISHES)
    qty = rand_qty(rng)
    t = f"order {qty} {dish}"
    return Clause("order_create", t, [entity("quantity", qty), entity("dish", dish)])

def clause_order_modify(rng: random.Random) -> Clause:
    dish = choice(rng, DISHES)
    t = choice(rng, [f"add {dish} to my order", f"remove {dish} from my order", f"change my order, no {dish}"])
    ents = [entity("dish", dish)]
    return Clause("order_modify", t, ents)

def clause_order_cancel(rng: random.Random) -> Clause:
    if maybe(rng, 0.5):
        oid = choice(rng, ORDER_IDS)
        return Clause("order_cancel", f"cancel order {oid}", [entity("order_id", oid)])
    return Clause("order_cancel", "cancel my order", [])

def clause_delivery_status(rng: random.Random) -> Clause:
    if maybe(rng, 0.6):
        oid = choice(rng, ORDER_IDS)
        return Clause("delivery_status", f"track order {oid}", [entity("order_id", oid)])
    return Clause("delivery_status", "where is my delivery?", [])

def clause_pickup_instructions(rng: random.Random) -> Clause:
    t = choice(rng, ["how does pickup work?", "pickup instructions please", "where do i collect my pickup?"])
    return Clause("pickup_instructions", t, [])

def clause_payment_methods(rng: random.Random) -> Clause:
    pay = choice(rng, PAYMENTS) if maybe(rng, 0.6) else None
    if pay:
        t = f"do you accept {pay}?"
        return Clause("payment_methods", t, [entity("payment_method", pay)])
    return Clause("payment_methods", "what payment methods do you accept?", [])

def clause_refund_request(rng: random.Random) -> Clause:
    if maybe(rng, 0.6):
        oid = choice(rng, ORDER_IDS)
        t = f"i want a refund for order {oid}"
        return Clause("refund_request", t, [entity("order_id", oid)])
    return Clause("refund_request", "i want a refund", [])

def clause_complaint(rng: random.Random) -> Clause:
    t = choice(rng, ["the food was cold", "my order is missing items", "this is taking too long", "your staff was rude"])
    return Clause("complaint", t, [])

def clause_compliment(rng: random.Random) -> Clause:
    dish = choice(rng, DISHES) if maybe(rng, 0.6) else None
    if dish:
        return Clause("compliment", f"the {dish} was amazing", [entity("dish", dish)])
    return Clause("compliment", "great service today", [])

def clause_repeat(rng: random.Random) -> Clause:
    t = choice(rng, ["can you repeat that?", "repeat please", "say that again"])
    return Clause("repeat", t, [])

def clause_clarify(rng: random.Random) -> Clause:
    t = choice(rng, ["i'm confused", "i don't understand", "that's not what i meant"])
    return Clause("clarify", t, [])


CLAUSE_GEN: Dict[str, Any] = {
    "greet": clause_greet,
    "hours": clause_hours,
    "location": clause_location,
    "contact": clause_contact,
    "parking": clause_parking,
    "browse_menu": clause_browse_menu,
    "ask_item_info": clause_ask_item_info,
    "ask_price": clause_ask_price,
    "dietary_request": clause_dietary_request,
    "reservation_create": clause_reservation_create,
    "reservation_modify": clause_reservation_modify,
    "reservation_cancel": clause_reservation_cancel,
    "waitlist_join": clause_waitlist_join,
    "waitlist_status": clause_waitlist_status,
    "order_create": clause_order_create,
    "order_modify": clause_order_modify,
    "order_cancel": clause_order_cancel,
    "delivery_status": clause_delivery_status,
    "pickup_instructions": clause_pickup_instructions,
    "payment_methods": clause_payment_methods,
    "refund_request": clause_refund_request,
    "complaint": clause_complaint,
    "compliment": clause_compliment,
    "repeat": clause_repeat,
    "clarify": clause_clarify,
}


# -----------------------
# Multi-intent combo specs (2–4 intents)
# You can add/remove combos here to match your design.
# -----------------------
COMBOS: List[Tuple[str, ...]] = [
    ("greet", "hours", "location"),
    ("browse_menu", "ask_price"),
    ("dietary_request", "ask_item_info", "ask_price"),
    ("dietary_request", "order_create"),
    ("reservation_create", "hours"),
    ("reservation_create", "location", "payment_methods"),
    ("reservation_cancel", "waitlist_join"),
    ("waitlist_join", "waitlist_status"),
    ("order_create", "payment_methods", "pickup_instructions"),
    ("order_modify", "order_cancel", "refund_request"),
    ("delivery_status", "complaint", "refund_request"),
    ("contact", "hours"),
    ("repeat", "clarify", "browse_menu"),
    ("compliment", "browse_menu"),
    ("reservation_modify", "contact"),
    ("order_create", "delivery_status"),
    ("order_create", "ask_price"),
    ("order_modify", "dietary_request"),  # e.g., remove allergen-related item
]


def add_noise(rng: random.Random, text: str, noise: float) -> str:
    """Mild noise without breaking span indices too much.
    IMPORTANT: noise is applied BEFORE span computation if enabled.
    This function preserves length relationships but still changes text,
    so we apply it at clause-level only.
    """
    if noise <= 0:
        return text

    t = text

    if maybe(rng, noise * 0.4):
        t = t.lower()

    if maybe(rng, noise * 0.35):
        t = t.replace("please", "pls").replace("tomorrow", "tmr").replace("tonight", "tnite")

    if maybe(rng, noise * 0.25):
        t = t.replace("address", "adress").replace("reserve", "resreve").replace("open", "opne")

    return norm_spaces(t)


def build_utterance_with_spans(
    clauses: List[Clause],
    rng: random.Random
) -> Tuple[str, List[Dict[str, Any]], List[str], List[Dict[str, str]]]:
    """Join clauses and compute start/end indices for each intent clause."""
    parts: List[str] = []
    intent_spans: List[Dict[str, Any]] = []
    all_intents: List[str] = []
    all_entities: List[Dict[str, str]] = []

    cursor = 0
    for i, c in enumerate(clauses):
        if i > 0:
            sep = choice(rng, CONNECTORS)
            parts.append(sep)
            cursor += len(sep)

        parts.append(c.text)
        start = cursor
        end = cursor + len(c.text)  # end-exclusive

        intent_spans.append({
            "intent": c.intent,
            "start": start,
            "end": end,
            "text": c.text
        })

        cursor = end
        all_intents.append(c.intent)
        all_entities.extend(c.entities)

    text = "".join(parts)
    return text, intent_spans, all_intents, all_entities


def generate_one(rng: random.Random, noise: float) -> Dict[str, Any]:
    combo = choice(rng, COMBOS)
    clause_objs: List[Clause] = []

    for intent in combo:
        c = CLAUSE_GEN[intent](rng)
        # Apply noise at clause-level BEFORE joining (so spans match final text)
        c = Clause(intent=c.intent, text=add_noise(rng, c.text, noise), entities=c.entities)
        clause_objs.append(c)

    text, spans, intents, entities = build_utterance_with_spans(clause_objs, rng)

    # Optional: remove duplicate intent labels while keeping multiple spans if repeated
    # Here we keep both: `intents` as unique labels, `intent_spans` as per-clause spans.
    unique_intents = sorted(set(intents), key=intents.index)

    return {
        "text": text,
        "intents": unique_intents,
        "intent_spans": spans,
        "entities": entities
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="train_spans.jsonl")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--noise", type=float, default=0.15, help="0.0 clean, ~0.2 mild typos/slang")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Rough balancing: cycle combos evenly
    rows: List[Dict[str, Any]] = []
    for i in range(args.n):
        rows.append(generate_one(rng, max(0.0, min(0.5, args.noise))))

    # Write JSONL
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {args.n} samples -> {args.out}")
    print("Example:")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
