"""
handlers.py — Knowledge Base & Intent Action Handlers
======================================================
Contains the restaurant KB (menu, hours, contact, etc.)
and all intent-specific response generators.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state import StateManager


# ═══════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE — edit with real restaurant data
# ═══════════════════════════════════════════════════════════════════
KB: Dict[str, Any] = {
    "restaurant_name": "Sakura Grill 🌸",
    "hours": "Mon–Sun: 10:00 AM – 10:00 PM",
    "address": "123 Orchard Road, #05-01, Singapore 238858",
    "contact": {
        "phone": "+65 6123 4567",
        "email": "hello@sakuragrill.sg",
    },
    "parking": "Free parking available at basement B1 & B2.",
    "payment_modes": ["cash", "card", "paynow", "apple_pay", "google_pay"],

    "menu": {
        # ── Mains ──
        "chicken burger":    {"price": 12.90, "category": "mains", "emoji": "🍔"},
        "beef burger":       {"price": 14.90, "category": "mains", "emoji": "🍔"},
        "grilled salmon":    {"price": 18.90, "category": "mains", "emoji": "🐟"},
        "ramen":             {"price": 13.50, "category": "mains", "emoji": "🍜"},
        "pad thai":          {"price": 11.90, "category": "mains", "emoji": "🍝"},
        "margherita pizza":  {"price": 15.90, "category": "mains", "emoji": "🍕"},
        "pepperoni pizza":   {"price": 16.90, "category": "mains", "emoji": "🍕"},
        "steak":             {"price": 24.90, "category": "mains", "emoji": "🥩"},
        "caesar salad":      {"price": 10.90, "category": "mains", "emoji": "🥗"},
        # ── Sides ──
        "fries":             {"price": 5.90,  "category": "sides", "emoji": "🍟"},
        "onion rings":       {"price": 6.50,  "category": "sides", "emoji": "🧅"},
        "garlic bread":      {"price": 4.90,  "category": "sides", "emoji": "🍞"},
        "soup of the day":   {"price": 6.90,  "category": "sides", "emoji": "🍲"},
        # ── Drinks ──
        "coke":              {"price": 3.50,  "category": "drinks", "emoji": "🥤"},
        "sprite":            {"price": 3.50,  "category": "drinks", "emoji": "🥤"},
        "iced latte":        {"price": 5.90,  "category": "drinks", "emoji": "☕"},
        "iced tea":          {"price": 4.50,  "category": "drinks", "emoji": "🍵"},
        "fresh orange juice":{"price": 5.50,  "category": "drinks", "emoji": "🍊"},
        # ── Desserts ──
        "cheesecake":        {"price": 8.90,  "category": "desserts", "emoji": "🍰"},
        "brownie":           {"price": 7.50,  "category": "desserts", "emoji": "🍫"},
        "ice cream sundae":  {"price": 9.90,  "category": "desserts", "emoji": "🍨"},
    },
}

CATEGORIES = sorted(set(v["category"] for v in KB["menu"].values()))


# ═══════════════════════════════════════════════════════════════════
#  HANDLER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def handle_greet(sm: "StateManager") -> str:
    name = KB["restaurant_name"]
    return (
        f"Welcome to **{name}**! 🎉\n\n"
        "I'll help you place your order step by step.\n\n"
        "First, how would you like to dine?\n"
        "• **Dine-in** 🪑\n"
        "• **Takeaway** 🥡\n"
        "• **Delivery** 🛵"
    )


def handle_goodbye(sm: "StateManager") -> str:
    return (
        f"Thank you for visiting **{KB['restaurant_name']}**! "
        "Have a wonderful day! 👋😊"
    )


def handle_browse_menu(sm: "StateManager", category: Optional[str] = None) -> str:
    menu = KB["menu"]
    if category and category.lower() in CATEGORIES:
        cat = category.lower()
        items = {k: v for k, v in menu.items() if v["category"] == cat}
        lines = [f"**{cat.title()}** menu:\n"]
        for name, info in items.items():
            lines.append(
                f"  {info['emoji']} {name.title()} — **${info['price']:.2f}**"
            )
        return "\n".join(lines)

    # Show full menu grouped by category
    lines = ["📋 **Our Full Menu:**\n"]
    for cat in CATEGORIES:
        lines.append(f"**{cat.title()}**")
        items = {k: v for k, v in menu.items() if v["category"] == cat}
        for name, info in items.items():
            lines.append(
                f"  {info['emoji']} {name.title()} — **${info['price']:.2f}**"
            )
        lines.append("")
    return "\n".join(lines)


def handle_ask_price(sm: "StateManager", item_names: List[str]) -> str:
    menu = KB["menu"]
    found, unknown = [], []
    for name in item_names:
        key = name.lower().strip()
        if key in menu:
            found.append(f"  {menu[key]['emoji']} {key.title()} — **${menu[key]['price']:.2f}**")
        else:
            unknown.append(key)

    parts = []
    if found:
        parts.append("Here are the prices:\n" + "\n".join(found))
    if unknown:
        parts.append(f"❓ I couldn't find: {', '.join(unknown)}. Please check the menu!")
    return "\n\n".join(parts) if parts else "Please tell me which item you'd like the price for."


def handle_add_item(sm: "StateManager", items: List[Dict[str, Any]]) -> str:
    unknown = sm.add_menu_items(items)
    parts = []

    added = [i for i in items if i.get("name", "").lower().strip() not in unknown]
    if added:
        lines = ["✅ Added to your order:"]
        for it in added:
            lines.append(f"  • {it['qty']}× {it['name'].title()}")
        parts.append("\n".join(lines))

    if unknown:
        parts.append(
            f"❓ These items aren't on our menu: {', '.join(unknown)}.\n"
            "Type **menu** to see what's available!"
        )

    parts.append(f"\n🛒 **Your cart:**\n{sm.get_order_summary()}")
    return "\n\n".join(parts)


def handle_remove_item(sm: "StateManager", item_names: List[str]) -> str:
    removed, not_found = [], []
    for name in item_names:
        if sm.remove_menu_item(name):
            removed.append(name.title())
        else:
            not_found.append(name.title())

    parts = []
    if removed:
        parts.append(f"🗑️ Removed: {', '.join(removed)}")
    if not_found:
        parts.append(f"❓ Not in your cart: {', '.join(not_found)}")
    parts.append(f"\n🛒 **Your cart:**\n{sm.get_order_summary()}")
    return "\n\n".join(parts)


def handle_show_order(sm: "StateManager") -> str:
    summary = sm.get_order_summary()
    stage = sm.get_current_stage().value
    mode = sm.state.dine_mode or "not set"
    table = sm.state.table_number or "not set"
    payment = sm.state.payment_method or "not set"

    return (
        f"📦 **Order Status** (stage: {stage})\n\n"
        f"🍽️ Dine mode: **{mode}**\n"
        f"🪑 Table: **{table}**\n"
        f"💳 Payment: **{payment}**\n\n"
        f"🛒 **Cart:**\n{summary}"
    )


def handle_confirm(sm: "StateManager") -> str:
    sm.confirm_order()
    total = sm.get_total()
    mode = (sm.state.dine_mode or "").replace("_", " ").title()
    table = sm.state.table_number
    payment = (sm.state.payment_method or "").replace("_", " ").title()

    return (
        "🎉 **Order Confirmed!**\n\n"
        f"  🍽️ Mode: **{mode}**\n"
        f"  🪑 Table: **{table}**\n"
        f"  💳 Payment: **{payment}**\n\n"
        f"{sm.get_order_summary()}\n\n"
        f"Your order total is **${total:.2f}**.\n\n"
        "Your food will be prepared shortly. Thank you! 🙏"
    )


def handle_cancel(sm: "StateManager") -> str:
    sm.reset()
    return (
        "❌ Order cancelled. Your cart has been cleared.\n\n"
        "Feel free to start a new order anytime! Just say **hi**. 😊"
    )


def handle_help(sm: "StateManager") -> str:
    return (
        "💡 **Available Commands:**\n\n"
        "• **menu** — Browse the full menu\n"
        "• **order [items]** — Start or add to your order\n"
        "  _e.g. '2 chicken burgers and a coke'_\n"
        "• **remove [item]** — Remove an item from your cart\n"
        "• **status** — View your current order\n"
        "• **confirm** — Confirm and place your order\n"
        "• **cancel** — Cancel your order\n"
        "• **price [item]** — Check item price\n"
        "• **hours** — Our opening hours\n"
        "• **help** — Show this message\n"
    )


def handle_hours(sm: "StateManager") -> str:
    return f"🕐 **Opening Hours:** {KB['hours']}"


def handle_location(sm: "StateManager") -> str:
    return f"📍 **Address:** {KB['address']}"


def handle_contact(sm: "StateManager") -> str:
    c = KB["contact"]
    return f"📞 **Phone:** {c['phone']}\n✉️ **Email:** {c['email']}"


def handle_complain(sm: "StateManager", sentiment_label: str = "negative") -> str:
    return (
        "😔 I'm really sorry to hear that you're having a bad experience.\n\n"
        "Your feedback is important to us. I've noted your concern.\n"
        "Would you like me to connect you with a manager, "
        "or is there something specific I can help fix right now?"
    )


def handle_unknown(sm: "StateManager") -> str:
    return (
        "🤔 I'm not sure I understood that.\n\n"
        "Try saying things like:\n"
        "• _'I want to dine in'_\n"
        "• _'2 chicken burgers and a coke'_\n"
        "• _'Pay by card'_\n"
        "• _'menu'_ to see our menu\n"
        "• _'help'_ for all commands"
    )
