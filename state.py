"""
state.py — Conversation Memory & State Machine
================================================
Manages the slot-filling state machine for the restaurant chatbot.
Slots: dine_mode, table_number, menu_items, payment_method.
Stage transitions are strictly linear to prevent intent looping.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─── Stage Enum ───────────────────────────────────────────────────
class Stage(Enum):
    GREETING    = "greeting"
    DINE_MODE   = "dine_mode"
    TABLE       = "table"
    MENU        = "menu"
    PAYMENT     = "payment"
    CONFIRM     = "confirm"
    DONE        = "done"

# Linear ordering for the slot-filling gate
STAGE_ORDER = [
    Stage.GREETING,
    Stage.DINE_MODE,
    Stage.TABLE,
    Stage.MENU,
    Stage.PAYMENT,
    Stage.CONFIRM,
    Stage.DONE,
]


# ─── Menu Item Dataclass ─────────────────────────────────────────
@dataclass
class MenuItem:
    name: str
    qty: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "qty": self.qty}


# ─── Conversation State ──────────────────────────────────────────
@dataclass
class ConversationState:
    """Holds all slot values and the current stage."""
    current_stage: Stage = Stage.GREETING
    dine_mode: Optional[str] = None          # "dine_in" | "takeaway" | "delivery"
    table_number: Optional[int] = None       # 1-20
    menu_items: List[MenuItem] = field(default_factory=list)
    payment_method: Optional[str] = None     # "cash" | "card" | "paynow" | "apple_pay" | "google_pay"
    order_confirmed: bool = False
    turn_count: int = 0
    last_sentiment: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage.value,
            "dine_mode": self.dine_mode,
            "table_number": self.table_number,
            "menu_items": [m.to_dict() for m in self.menu_items],
            "payment_method": self.payment_method,
            "order_confirmed": self.order_confirmed,
            "turn_count": self.turn_count,
            "last_sentiment": self.last_sentiment,
        }


# ─── State Manager ───────────────────────────────────────────────
class StateManager:
    """
    Controls state transitions and slot updates.
    The slot-filling gate ensures stages only advance
    when the required slot is successfully extracted.
    """

    # Valid values for each slot
    VALID_DINE_MODES = {"dine_in", "takeaway", "delivery"}
    VALID_PAYMENTS   = {"cash", "card", "paynow", "apple_pay", "google_pay"}
    MAX_TABLE        = 20

    def __init__(self) -> None:
        self.state = ConversationState()
        self.history: List[Dict[str, str]] = []   # chat history [{role, content}]

    # ── Stage queries ─────────────────────────────────────────
    def get_current_stage(self) -> Stage:
        return self.state.current_stage

    def is_complete(self) -> bool:
        return self.state.current_stage == Stage.DONE

    def advance_stage(self) -> None:
        """Move to the next stage in the linear order."""
        idx = STAGE_ORDER.index(self.state.current_stage)
        if idx < len(STAGE_ORDER) - 1:
            self.state.current_stage = STAGE_ORDER[idx + 1]

    # ── Slot updates (returns True if the slot was successfully set) ──
    def set_dine_mode(self, mode: str) -> bool:
        mode = mode.lower().strip().replace("-", "_").replace(" ", "_")
        # Normalize common aliases
        aliases = {
            "dine_in": "dine_in", "dinein": "dine_in",
            "takeaway": "takeaway", "take_away": "takeaway",
            "pickup": "takeaway", "pick_up": "takeaway",
            "delivery": "delivery",
        }
        normalized = aliases.get(mode, mode)
        if normalized in self.VALID_DINE_MODES:
            self.state.dine_mode = normalized
            return True
        return False

    def set_table_number(self, num: int) -> bool:
        if 1 <= num <= self.MAX_TABLE:
            self.state.table_number = num
            return True
        return False

    def add_menu_items(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        Adds items to the order. Returns list of item names that
        were NOT found on the menu (unknown items).
        """
        from handlers import KB  # deferred import to avoid circular
        unknown = []
        for item in items:
            name = item.get("name", "").lower().strip()
            qty  = int(item.get("qty", 1))
            if name in KB["menu"]:
                # Check if already in cart — merge quantities
                existing = next((m for m in self.state.menu_items if m.name == name), None)
                if existing:
                    existing.qty += qty
                else:
                    self.state.menu_items.append(MenuItem(name=name, qty=qty))
            else:
                unknown.append(name)
        return unknown

    def remove_menu_item(self, name: str) -> bool:
        name = name.lower().strip()
        before = len(self.state.menu_items)
        self.state.menu_items = [m for m in self.state.menu_items if m.name != name]
        return len(self.state.menu_items) < before

    def set_payment_method(self, method: str) -> bool:
        method = method.lower().strip().replace(" ", "_")
        aliases = {
            "cash": "cash",
            "card": "card", "credit_card": "card", "debit_card": "card",
            "visa": "card", "mastercard": "card",
            "paynow": "paynow", "pay_now": "paynow",
            "apple_pay": "apple_pay", "applepay": "apple_pay",
            "google_pay": "google_pay", "googlepay": "google_pay",
        }
        normalized = aliases.get(method, method)
        if normalized in self.VALID_PAYMENTS:
            self.state.payment_method = normalized
            return True
        return False

    def confirm_order(self) -> None:
        self.state.order_confirmed = True
        self.state.current_stage = Stage.DONE

    # ── Summary helpers ───────────────────────────────────────
    def get_order_summary(self) -> str:
        from handlers import KB
        if not self.state.menu_items:
            return "Your cart is empty."
        lines = []
        total = 0.0
        for m in self.state.menu_items:
            price = KB["menu"].get(m.name, {}).get("price", 0)
            subtotal = price * m.qty
            total += subtotal
            lines.append(f"  • {m.qty}× {m.name.title()} — ${subtotal:.2f}")
        lines.append(f"  **Total: ${total:.2f}**")
        return "\n".join(lines)

    def get_total(self) -> float:
        from handlers import KB
        total = 0.0
        for m in self.state.menu_items:
            price = KB["menu"].get(m.name, {}).get("price", 0)
            total += price * m.qty
        return round(total, 2)

    # ── Reset ─────────────────────────────────────────────────
    def reset(self) -> None:
        self.state = ConversationState()
        self.history = []

    # ── Chat history management ───────────────────────────────
    def add_turn(self, role: str, content: str) -> None:
        self.state.turn_count += 1
        self.history.append({"role": role, "content": content})

    def set_sentiment(self, label: str) -> None:
        self.state.last_sentiment = label
