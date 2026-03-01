# System Overview — Sakura Grill Chatbot
### MA4891 CA1 — Restaurant Domain

---

## The Problem We're Solving

A customer walks into a restaurant and wants to:
- Order food ("I want 2 ramen and a coke")
- Ask questions ("What are your hours?")
- Make a reservation ("Book a table for 4 at 7pm")
- Do multiple things at once ("Hi, I'm John, and I'd like to dine in")

A chatbot that handles this needs to:
1. **Understand what the user wants** → Intent Classification
2. **Extract specific details** → Slot Extraction (what item? how many? what name?)
3. **Act on it** → Update the order, look up info, book a table
4. **Respond naturally** → Not sound like a robot

We built a system that does all four — using **three different ML/NLP techniques** working together.

---

## The Pipeline — How a Message Flows

When a user types something, it goes through **4 stages in sequence**.  
Here's the logic, step by step:

```
"Hi, I'm John. Can I get 2 ramen for dine in?"
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
 STAGE 1                         STAGE 2
 WHAT does the user want?        WHAT details did they give?
 (Intent Classification)         (Slot Extraction)
    │                               │
    │ Trained BERT model            │ spaCy PhraseMatcher
    │ tags each word:               │ finds:
    │  "Hi" → B-greet              │  • name: "John"
    │  "get" → B-order_create      │  • item: "ramen" × 2
    │  "ramen" → I-order_create    │  • mode: "dine in"
    │                               │
    │ Result: ["greet",             │ Result: {name, items,
    │  "order_create"]              │  dining_mode}
    └───────────┬───────────────────┘
                ▼
            STAGE 3
            DO something about it
            (Intent Engine)
                │
                │ For each intent, call its handler:
                │  • handle_greet → "Welcome, John!"
                │  • handle_order_create → add 2× ramen to cart
                │
                │ Result: response text + updated cart state
                │
                ▼
            STAGE 4
            SAY it naturally
            (Phi-2 Transformer LLM)
                │
                │ Takes the engine output and generates:
                │ "Hi John! I've added 2 ramen to your order
                │  for dine-in. That's $16.00 so far.
                │  Would you like anything else?"
                │
                ▼
            DISPLAYED IN STREAMLIT UI
```

**Why 4 stages instead of 1?**  
Because each task needs a different tool. A trained classifier is good at understanding intent but can't look up prices. spaCy is fast at matching menu items but can't generate sentences. The LLM writes beautifully but can't be trusted with prices. By separating them, each does what it's best at.

---

## Stage 1: Intent Classification — WHY We Trained Our Own Model

### The decision

| Option | Why we rejected it |
|---|---|
| Keyword matching ("if 'order' in text") | Not ML — doesn't meet assignment requirement. Breaks on "I'd like to get some food" |
| Dialogflow / Rasa API | Just calling an API doesn't demonstrate ML understanding |
| Multi-class BERT classifier | Only picks ONE intent per message — can't handle "order ramen and check hours" |
| **✅ BIO Span Model (what we chose)** | Trained ourselves. Tags each word. Detects MULTIPLE intents per message. |

### How it works

BIO tagging labels every word in the sentence:

| Word | Tag | Meaning |
|---|---|---|
| I | B-order_create | **B**eginning of "order_create" intent |
| want | I-order_create | **I**nside the same intent span |
| 2 | I-order_create | Still inside |
| ramen | I-order_create | Still inside |
| and | O | **O**utside — not part of any intent |
| check | B-ask_hours | **B**eginning of a NEW intent |
| your | I-ask_hours | Inside "ask_hours" |
| hours | I-ask_hours | Inside "ask_hours" |

**Result:** `["order_create", "ask_hours"]` — two intents from one sentence.

### Why this matters

This is the same technique used for Named Entity Recognition (NER) in research papers (Lample et al., 2016), but we applied it to **intent detection** — that's a novel cross-application. Most student chatbots can only detect one intent at a time.

**Files:** `intent_model_adapter.py` (loads model), `intent_span_model/` (265 MB trained weights)

---

## Stage 2: Slot Extraction — WHY spaCy PhraseMatcher

### The decision

| Option | Why we rejected it |
|---|---|
| Simple `if "ramen" in text` | Breaks on "chicken burger" (two words). No quantity extraction. |
| spaCy EntityRuler | Runs on every `nlp()` call — slower. Pipeline-level, not on-demand. |
| Regex only | Fragile. Can't handle case variations or multi-word phrases cleanly. |
| **✅ spaCy PhraseMatcher + NER** | Fast hash lookups. Case-insensitive. Multi-word matching. NER for names. |

### spaCy features we actually use

| Feature | What it does | Why we need it |
|---|---|---|
| `PhraseMatcher(nlp.vocab, attr="LOWER")` | Matches "Chicken Burger", "chicken burger", "CHICKEN BURGER" identically | Customers type in any case |
| `nlp(text).ents` → PERSON | Detects names like "John", "Sarah" | Customer name extraction |
| `parsed[start:end]` | Gets the exact matched text span | Know which words matched |
| `nlp.vocab.strings[match_id]` | Looks up which category matched (MENU_ITEM vs DINING_MODE) | Route to correct slot |
| Regex `\b(\d+)\b` in context window | Scans 12 chars before the match for a number | "2 ramen" → qty=2 |

### What it extracts

| Slot | How | Example |
|---|---|---|
| **Menu items + quantity** | PhraseMatcher + context-window regex | "2 ramen and a coke" → [{ramen, 2}, {coke, 1}] |
| **Customer name** | NER (PERSON) + regex fallback | "I'm John" → "John" |
| **Dining mode** | PhraseMatcher | "dine in" → dine_in |
| **Payment method** | PhraseMatcher | "pay by card" → card |

**File:** `slots_spacy.py` (140 lines)

---

## Stage 3: Intent Engine — WHY a Dispatch Router

### The decision

| Option | Why we rejected it |
|---|---|
| Giant if-else chain | Unreadable. Hard to add new intents. |
| State machine (FSM) | Forces linear conversation — user can't ask "what are your hours?" mid-order |
| **✅ Router pattern** | Maps intent strings to handler functions. Easy to extend. User can do anything at any time. |

### How it works

```python
INTENT_ROUTER = {
    "greet":              handle_greet,
    "order_create":       handle_order_create,
    "browse_menu":        handle_browse_menu,
    "ask_hours":          handle_hours,
    "complaint":          handle_complaint,
    # ... 16 total
}
```

For each predicted intent, the engine calls the corresponding handler. The handler receives the user's text, the conversation state, and the extracted slots — then mutates the state (adds items to cart, sets name, etc.) and returns response blocks.

### All 16 handlers

| Handler | Does what | Example |
|---|---|---|
| `handle_greet` | Welcome + name extraction | "Hi, I'm John" |
| `handle_order_create` | Add items to cart + pricing | "I want 2 ramen" |
| `handle_order_modify` | Change qty or remove items | "Remove the fries" |
| `handle_order_cancel` | Clear entire cart | "Cancel everything" |
| `handle_browse_menu` | Show menu by category | "What's on the menu?" |
| `handle_ask_price` | Price lookup | "How much is steak?" |
| `handle_hours` | Operating hours | "When do you close?" |
| `handle_location` | Address | "Where are you?" |
| `handle_contact` | Phone / email | "What's your number?" |
| `handle_parking` | Parking info | "Is there parking?" |
| `handle_payment_methods` | Accepted payments | "Do you take PayNow?" |
| `handle_reservation_create` | Book a table | "Table for 4 at 7pm" |
| `handle_reservation_modify` | Change booking | "Move it to 8pm" |
| `handle_reservation_cancel` | Cancel booking | "Cancel my reservation" |
| `handle_complaint` | Log + acknowledge | "Food was cold" |
| `handle_refund_request` | Process refund | "I want a refund" |

**File:** `intent_engine.py` (627 lines)

---

## Stage 4: Response Generation — WHY Phi-2, Not ChatGPT

### The decision

| Option | Why we rejected it |
|---|---|
| Template strings | Robotic. Same response every time. |
| ChatGPT / OpenAI API | Costs money. Requires API key. Doesn't demonstrate ML understanding. |
| DialoGPT | Older model (2020). Less capable. |
| **✅ Phi-2 (2.7B params)** | Free. Runs locally. We control the prompt. Demonstrates transformer understanding. |

### How it works — Grounded Generation

The LLM doesn't invent the response from nothing. It reads the **exact engine output** and turns it into natural language:

```
PROMPT TO PHI-2:

System: You are a restaurant assistant. The user's cart has 2× ramen ($16.00).
        Missing: payment_mode.

User: "I want 2 ramen for dine in"

Engine output: { cart: [ramen ×2], response: "Added 2× ramen. Total: $16.00" }

→ Generate a natural, friendly reply based on the above data.
```

**Why this is important:** The LLM never hallucinate prices or menu items because those come from the engine, not from the model's memory. This is the same principle as **Retrieval-Augmented Generation (RAG)** used by Google and OpenAI.

### CPU Fallback — Not a Bug, a Feature

Phi-2 needs a GPU (6GB+ VRAM) for real-time generation. On laptops without a GPU, the system uses a **smart deterministic fallback** that parses the engine's response blocks directly. The bot still works perfectly — it just uses structured text instead of LLM-generated language.

**File:** `phi_llm.py` (123 lines)

---

## The Streamlit UI

The frontend (`app.py`, 329 lines) provides:

| Feature | What it does |
|---|---|
| **Sidebar: Load Models** | One-click to initialise BERT + Phi-2 |
| **Sidebar: Order Tracker** | Live display of name, dining mode, cart with prices |
| **Chat Interface** | Message bubbles with 🧑 user and 🌸 bot avatars |
| **Debug Panels** | Expandable panels showing predicted intents and full history state |
| **Model Status** | Badges showing ✅ loaded or ⏳ not loaded for each model |

---

## Summary — Why This System Is Different

| What most students build | What we built |
|---|---|
| Keyword matching for intents | Trained our own BERT classifier |
| One intent per message | Multi-intent via BIO span tagging |
| `if "ramen" in text` for entities | spaCy PhraseMatcher + NER + regex |
| Template string responses | Phi-2 LLM with grounded generation |
| Terminal chatbot | Production Streamlit UI with live tracking |
| Crashes without GPU | Graceful CPU fallback — always works |
| 3-5 intents | 16 intent handlers covering full restaurant scope |

**The key idea:** Three ML/NLP techniques — none of which is sufficient alone — compose into a system that is reliable, natural, and extensible. Each component does what it's best at, and they hand off to each other cleanly.
