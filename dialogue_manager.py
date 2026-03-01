"""
dialogue_manager.py — Slot-Filling Gate & Flow Control
=======================================================
Implements the strict slot-filling state machine that prevents
intent looping. Only advances stages when a slot is successfully
extracted. Integrates sentiment analysis for tone adaptation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from state import Stage, StateManager
import nlu
import handlers


# ─── Re-prompt variants (to avoid repetitive responses) ──────────
REPROMPT = {
    Stage.DINE_MODE: [
        "How would you like to dine? **Dine-in**, **Takeaway**, or **Delivery**?",
        "Please choose a dining mode: 🪑 Dine-in, 🥡 Takeaway, or 🛵 Delivery.",
        "I need to know your dining preference first. Dine-in, takeaway, or delivery?",
    ],
    Stage.TABLE: [
        "Great! Which table number? (1–20)",
        "Please pick a table number between 1 and 20. 🪑",
        "What's your table number? We have tables 1 through 20.",
    ],
    Stage.MENU: [
        "What would you like to order? You can say something like _'2 chicken burgers and a coke'_.",
        "Time to order! Tell me what you'd like from the menu. 🍽️",
        "What items can I get for you? Say _'menu'_ to see our options.",
    ],
    Stage.PAYMENT: [
        "How would you like to pay? **Cash**, **Card**, **PayNow**, **Apple Pay**, or **Google Pay**?",
        "Almost done! What's your payment method? 💳",
        "Last step — choose your payment: cash, card, paynow, apple pay, or google pay.",
    ],
    Stage.CONFIRM: [
        "Everything looks good! Say **confirm** to place your order, or **cancel** to start over.",
        "Ready to place your order? Say **confirm** or **cancel**.",
        "Your order is ready. Type **confirm** to finalize! ✅",
    ],
}


def _get_reprompt(stage: Stage, turn: int) -> str:
    """Get a varied re-prompt based on the turn count to avoid repetition."""
    prompts = REPROMPT.get(stage, ["What else can I help with?"])
    return prompts[turn % len(prompts)]


# ═══════════════════════════════════════════════════════════════════
#  Dialogue Manager
# ═══════════════════════════════════════════════════════════════════

class DialogueManager:
    """
    Core dialogue flow controller using a slot-filling gate.

    Pipeline for each message:
        1. Entity extraction (spaCy EntityRuler)
        2. Intent classification (rule-based)
        3. Sentiment analysis (RoBERTa)
        4. Slot-filling gate: try to fill the current required slot
        5. Route to handler → generate response
        6. Update state ONLY on successful slot extraction
    """

    def __init__(self, state_manager: StateManager) -> None:
        self.sm = state_manager

    def process_message(self, user_text: str) -> Dict[str, Any]:
        """
        Main entry point. Processes a user message and returns:
        {
            "response": str,
            "intent": str,
            "entities": dict,
            "sentiment": dict,
            "stage": str,
            "state": dict,
        }
        """
        # ── Step 1: Entity extraction ──
        entities = nlu.extract_entities(user_text)

        # ── Step 2: Intent classification ──
        intent = nlu.classify_intent(user_text)

        # ── Step 3: Sentiment analysis ──
        sentiment = nlu.analyze_sentiment(user_text)
        self.sm.set_sentiment(sentiment["label"])

        # ── Step 4 & 5: Slot-filling gate + handler routing ──
        response = self._route(user_text, intent, entities, sentiment)

        # ── Build debug payload ──
        return {
            "response": response,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "stage": self.sm.get_current_stage().value,
            "state": self.sm.state.to_dict(),
        }

    # ─── Routing Logic ────────────────────────────────────────
    def _route(
        self,
        text: str,
        intent: str,
        entities: Dict[str, Any],
        sentiment: Dict[str, Any],
    ) -> str:
        """
        Routes the message through the slot-filling gate.
        Priority:
            1. Global intents (cancel, help, goodbye, complain) — always handled
            2. Informational intents (menu, price, hours, etc.) — always available
            3. Slot-filling gate — stage-specific processing
        """
        stage = self.sm.get_current_stage()
        sentiment_prefix = self._sentiment_prefix(sentiment)

        # ── Global intents (always available) ──
        if intent == "cancel":
            return sentiment_prefix + handlers.handle_cancel(self.sm)

        if intent == "goodbye":
            return sentiment_prefix + handlers.handle_goodbye(self.sm)

        if intent == "help":
            return sentiment_prefix + handlers.handle_help(self.sm)

        if intent == "complain":
            return sentiment_prefix + handlers.handle_complain(self.sm, sentiment["label"])

        # ── Informational intents (don't affect slot state) ──
        if intent == "browse_menu":
            return sentiment_prefix + handlers.handle_browse_menu(
                self.sm, category=entities.get("category")
            )

        if intent == "ask_price":
            item_names = [it["name"] for it in entities.get("menu_items", [])]
            if not item_names:
                # Try to extract item names from text directly
                item_names = self._fallback_item_names(text)
            return sentiment_prefix + handlers.handle_ask_price(self.sm, item_names)

        if intent == "hours":
            return sentiment_prefix + handlers.handle_hours(self.sm)

        if intent == "location":
            return sentiment_prefix + handlers.handle_location(self.sm)

        if intent == "contact":
            return sentiment_prefix + handlers.handle_contact(self.sm)

        if intent == "show_order":
            return sentiment_prefix + handlers.handle_show_order(self.sm)

        # ── Stage: DONE ──
        if stage == Stage.DONE:
            return sentiment_prefix + (
                "Your order has already been placed! 🎉\n\n"
                "Say **cancel** to start a new order, or **bye** to leave."
            )

        # ── Stage: GREETING ──
        if stage == Stage.GREETING:
            # Always advance past greeting on first interaction
            self.sm.advance_stage()  # → DINE_MODE

            # Check if they already provided dine mode in the same message
            if entities.get("dine_mode"):
                return sentiment_prefix + self._try_fill_dine_mode(entities, text)

            return sentiment_prefix + handlers.handle_greet(self.sm)

        # ── Stage: DINE_MODE ──
        if stage == Stage.DINE_MODE:
            return sentiment_prefix + self._try_fill_dine_mode(entities, text)

        # ── Stage: TABLE ──
        if stage == Stage.TABLE:
            return sentiment_prefix + self._try_fill_table(entities, text)

        # ── Stage: MENU ──
        if stage == Stage.MENU:
            return sentiment_prefix + self._try_fill_menu(entities, text, intent)

        # ── Stage: PAYMENT ──
        if stage == Stage.PAYMENT:
            return sentiment_prefix + self._try_fill_payment(entities, text)

        # ── Stage: CONFIRM ──
        if stage == Stage.CONFIRM:
            return sentiment_prefix + self._try_confirm(intent, text)

        # Fallback
        return sentiment_prefix + handlers.handle_unknown(self.sm)

    # ─── Slot Fillers ─────────────────────────────────────────
    def _try_fill_dine_mode(self, entities: Dict, text: str) -> str:
        mode = entities.get("dine_mode")
        if mode and self.sm.set_dine_mode(mode):
            display = mode.replace("_", " ").title()
            self.sm.advance_stage()  # → TABLE

            # Check if table was also mentioned
            if entities.get("table_number"):
                return self._cascade_fill(entities, text,
                    f"✅ Dining mode set to **{display}**.\n\n")

            return (
                f"✅ Dining mode set to **{display}**.\n\n"
                + _get_reprompt(Stage.TABLE, self.sm.state.turn_count)
            )
        return _get_reprompt(Stage.DINE_MODE, self.sm.state.turn_count)

    def _try_fill_table(self, entities: Dict, text: str) -> str:
        table = entities.get("table_number")
        if table is None:
            # Try regex fallback for just a number
            import re
            m = re.search(r"\b(\d{1,2})\b", text)
            if m:
                table = int(m.group(1))

        if table is not None and self.sm.set_table_number(table):
            self.sm.advance_stage()  # → MENU

            # Check if menu items were also mentioned
            if entities.get("menu_items"):
                return self._cascade_fill(entities, text,
                    f"✅ Table **{table}** selected.\n\n")

            return (
                f"✅ Table **{table}** selected.\n\n"
                + _get_reprompt(Stage.MENU, self.sm.state.turn_count)
            )
        return _get_reprompt(Stage.TABLE, self.sm.state.turn_count)

    def _try_fill_menu(self, entities: Dict, text: str, intent: str) -> str:
        items = entities.get("menu_items", [])

        if intent == "remove_item":
            item_names = [it["name"] for it in items]
            if item_names:
                return handlers.handle_remove_item(self.sm, item_names)
            return "Which item would you like to remove? 🗑️"

        if items:
            response = handlers.handle_add_item(self.sm, items)
            return (
                response + "\n\n"
                "Add more items, say **done ordering** when finished, "
                "or type **menu** to browse."
            )

        if intent == "confirm" or any(k in text.lower() for k in
            ["done", "that's all", "thats all", "nothing else", "finished", "done ordering"]):
            if not self.sm.state.menu_items:
                return "Your cart is empty! Please add at least one item first. 🍽️"
            self.sm.advance_stage()  # → PAYMENT

            # Check if payment was also mentioned
            if entities.get("payment_method"):
                return self._cascade_fill(entities, text,
                    "✅ Great choices!\n\n")

            return (
                "✅ Great choices!\n\n"
                + handlers.handle_show_order(self.sm) + "\n\n"
                + _get_reprompt(Stage.PAYMENT, self.sm.state.turn_count)
            )

        return _get_reprompt(Stage.MENU, self.sm.state.turn_count)

    def _try_fill_payment(self, entities: Dict, text: str) -> str:
        method = entities.get("payment_method")
        if method and self.sm.set_payment_method(method):
            display = method.replace("_", " ").title()
            self.sm.advance_stage()  # → CONFIRM

            return (
                f"✅ Payment method: **{display}**.\n\n"
                + handlers.handle_show_order(self.sm) + "\n\n"
                + _get_reprompt(Stage.CONFIRM, self.sm.state.turn_count)
            )
        return _get_reprompt(Stage.PAYMENT, self.sm.state.turn_count)

    def _try_confirm(self, intent: str, text: str) -> str:
        t = text.lower().strip()
        if intent == "confirm" or t in ("yes", "y", "confirm", "ok", "okay", "sure", "yep"):
            return handlers.handle_confirm(self.sm)
        return _get_reprompt(Stage.CONFIRM, self.sm.state.turn_count)

    # ─── Cascade Fill ─────────────────────────────────────────
    def _cascade_fill(self, entities: Dict, text: str, prefix: str) -> str:
        """
        When a user provides multiple slot values in one message,
        cascade through the stages filling each one.
        """
        stage = self.sm.get_current_stage()

        if stage == Stage.TABLE and entities.get("table_number"):
            table = entities["table_number"]
            if self.sm.set_table_number(table):
                prefix += f"✅ Table **{table}** selected.\n\n"
                self.sm.advance_stage()  # → MENU
                stage = self.sm.get_current_stage()

        if stage == Stage.MENU and entities.get("menu_items"):
            items = entities["menu_items"]
            self.sm.add_menu_items(items)
            lines = [f"  • {it['qty']}× {it['name'].title()}" for it in items]
            prefix += "✅ Added to your order:\n" + "\n".join(lines) + "\n\n"
            # Don't auto-advance from MENU — let user add more items

        if stage == Stage.MENU and not entities.get("menu_items"):
            prefix += _get_reprompt(Stage.MENU, self.sm.state.turn_count)
            return prefix

        if stage == Stage.PAYMENT and entities.get("payment_method"):
            method = entities["payment_method"]
            if self.sm.set_payment_method(method):
                display = method.replace("_", " ").title()
                prefix += f"✅ Payment method: **{display}**.\n\n"
                self.sm.advance_stage()  # → CONFIRM
                prefix += _get_reprompt(Stage.CONFIRM, self.sm.state.turn_count)
                return prefix

        # Prompt for next needed slot
        next_stage = self.sm.get_current_stage()
        if next_stage in REPROMPT:
            prefix += _get_reprompt(next_stage, self.sm.state.turn_count)

        return prefix

    # ─── Sentiment Prefix ─────────────────────────────────────
    def _sentiment_prefix(self, sentiment: Dict[str, Any]) -> str:
        """
        If user sentiment is negative, prepend an empathetic message.
        """
        if sentiment.get("label") == "negative" and sentiment.get("score", 0) > 0.6:
            return (
                "😔 I sense some frustration — I'm sorry about that. "
                "Let me do my best to help.\n\n"
            )
        return ""

    # ─── Helpers ──────────────────────────────────────────────
    def _fallback_item_names(self, text: str) -> List[str]:
        """Extract item names from text using simple substring matching."""
        t = text.lower()
        found = []
        for item_name in nlu.MENU_ITEM_NAMES:
            if item_name in t:
                found.append(item_name)
        return found
