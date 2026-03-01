"""
Run all 5 demos through the chatbot pipeline and capture exact bot responses.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot_core import init_history, process_turn
from phi_llm import PhiChat
from intent_postprocessor import postprocess_intents

# We don't load ML models — tests the deterministic pipeline
llm = PhiChat()  # will use CPU fallback

def simulate_intents(text: str) -> list:
    """Without the BERT model, pass empty raw_intents.
    The postprocessor's Layer 3 (slot-aware) and Layer 4 (keyword fallback)
    will detect the correct intents deterministically."""
    return postprocess_intents(
        raw_intents=[],
        user_text=text,
        intent_spans=None,
    )

def run_demo(name: str, inputs: list):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    
    history = init_history()
    results = []
    
    for i, user_input in enumerate(inputs, 1):
        intents = simulate_intents(user_input)
        response = process_turn(user_input, intents, history, llm=llm)
        results.append((user_input, response))
        print(f"\n  Turn {i}")
        print(f"  User:  {user_input}")
        print(f"  Intents: {intents}")
        print(f"  Bot:   {response}")
    
    return results

# ═══════════════════════════════════════════════════════════════
# DEMO 1: Simple Ordering Flow (report Section 5.1)
# ═══════════════════════════════════════════════════════════════
demo1 = run_demo("DEMO 1: Simple Ordering Flow", [
    "Hello!",
    "John",
    "I want to dine in",
    "Table 5",
    "2 chicken burgers and a coke",
    "add a cheesecake",
    "done ordering",
    "Pay by card",
    "confirm",
])

# ═══════════════════════════════════════════════════════════════
# DEMO 2: Cascade Filling (report Section 5.2)
# ═══════════════════════════════════════════════════════════════
demo2 = run_demo("DEMO 2: Complex Multi-Slot Cascade", [
    "Hi there!",
    "Sarah",
    "I'd like to dine in at table 3",
    "Give me a steak, fries, and 2 iced teas",
    "done",
    "Cash please",
    "confirm",
])

# ═══════════════════════════════════════════════════════════════
# DEMO 3: Edge Cases (report Section 5.3)
# ═══════════════════════════════════════════════════════════════
demo3 = run_demo("DEMO 3: Edge Cases & Unknown Items", [
    "Hey",
    "Alex",
    "takeaway",
    "I want a pizza supreme and onion rings",
    "menu",
    "margherita pizza",
    "done",
    "paynow",
    "confirm",
])

# ═══════════════════════════════════════════════════════════════
# DEMO 4: Robustness (NEW)
# ═══════════════════════════════════════════════════════════════
demo4 = run_demo("DEMO 4: Robustness — Typos, Ambiguity, Correction", [
    "hoi",
    "Marco",
    "I want pizza",
    "pepperoni",
    "add bugers",
    "chicken",
    "undo",
    "change my name",
    "Sarah",
    "done",
    "dine in",
    "card",
    "confirm",
])

# ═══════════════════════════════════════════════════════════════
# DEMO 5: Follow-up & Multi-Intent (NEW)
# ═══════════════════════════════════════════════════════════════
demo5 = run_demo("DEMO 5: Follow-Up & Multi-Intent", [
    "Hi, my name is Alex",
    "I want 2 ramen and what are your hours?",
    "also a coke",
    "show menu",
    "add cheesecake",
    "start over",
    "Marco",
    "order steak",
    "done",
    "takeaway",
    "cash",
    "confirm",
])

print("\n\n" + "="*70)
print("  ALL DEMOS COMPLETE")
print("="*70)
