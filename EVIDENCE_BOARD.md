# MA4891 Evidence Board — Sakura Grill Chatbot
### Presentation Package | Hybrid NLP Pipeline

---

> **How to use this document:** Each "page" below corresponds to one slide/screen you will scroll through in your video and evidence board. Screenshots are embedded inline. Follow the Video Run-Sheet at the end to record a tight 5-minute walkthrough.

---

# SECTION A — THE 14-PAGE EVIDENCE BOARD

---

## PAGE 1 — The Problem & Why a Hybrid Pipeline

**Screenshot:** Open `PIPELINE_AT_A_GLANCE.md` in your editor, showing the pipeline diagram (`pipeline_diagram.png`) and the "Pipeline in One Sentence" heading.

**What I Say:**
- "A restaurant chatbot needs to solve four distinct problems: understand intent, extract details, act on them, and respond naturally. No single ML technique does all four."
- "Our solution is a **4-stage hybrid pipeline**: Trained BERT → spaCy PhraseMatcher → 16-handler Intent Engine → Phi-2 LLM."
- "Each stage does what it's best at, and they hand off cleanly — this is the same separation-of-concerns principle used in production NLU systems like Rasa and Google Dialogflow."

**X-Factor:** Unlike typical student chatbots that use one technique (keyword matching OR an API call), we compose three ML/NLP techniques into a grounded, testable pipeline.

**Requirement Satisfied:** Flow chart / system architecture — demonstrates full pipeline understanding.

---

## PAGE 2 — The System Architecture (Flow Chart)

**Screenshot:** Open `SYSTEM_OVERVIEW.md` and scroll to the ASCII pipeline diagram showing the 4-stage message flow.

**What I Say:**
- "Here is the full data flow of a single user message. Stage 1: BERT tags each word with a BIO label. Stage 2: spaCy extracts slots (name, items, dining mode). Stage 3: the Intent Engine dispatches to 16 handlers and mutates the cart. Stage 4: Phi-2 takes the structured output and generates a natural reply."
- "Crucially, Stages 1–3 are fully deterministic — they produce the same output every time. Only Stage 4 introduces variance, and only in phrasing, never in data."
- "This architecture lets us test Stages 1–3 exhaustively (61 automated tests) and still get natural language output."

**X-Factor:** Deterministic core + generative surface — testable AND natural-sounding.

**Requirement Satisfied:** Flow chart + complex dialog management.

---

## PAGE 3 — My Contribution #1: Streamlit Production UI

**Screenshot:**

![Full chatbot UI with sidebar showing models, order state, and chat area](evidence_screenshots/11_full_page_ui.png)

**What I Say:**
- "I built the entire Streamlit frontend — 337 lines in `app.py`. The sidebar has: a model loader with auto-checkpoint discovery, a confidence threshold slider for multi-label mode, live status badges, and a real-time order tracker that displays the cart with line-item pricing."
- "Session state management was critical — Streamlit reruns the entire script on every interaction, so I used `st.session_state` to persist history, messages, debug logs, and loaded models across reruns."

**X-Factor:** Production-grade UI with real-time state management, not a terminal chatbot.

**Requirement Satisfied:** Transformers + spaCy integration in a usable interface.

---

## PAGE 4 — My Contribution #2: The 5-Layer Post-Processing System

**Screenshot:** Open `intent_postprocessor.py` and show the file header (lines 1–14) with the docstring listing all 5 layers.

**What I Say:**
- "This is the module I'm most proud of. The raw BERT model alone achieves roughly 60% accuracy on ordering intents because we had limited training data (~200 examples). My 5-layer post-processing pipeline lifts the system to **96.7% exact-match accuracy**."
- "The five layers are: (1) Label Mapping — normalises model labels to engine labels. (2) Confidence Filter — drops low-confidence single-token noise spans. (3) Slot-Aware Correction — if spaCy PhraseMatcher finds menu items, inject `order_create`. (4) Keyword Fallback — deterministic rules for intents the model consistently misses. (5) Conflict Pruning — resolves contradictions like `reservation_create` + `reservation_cancel` and caps at 2 intents max."
- "The same `postprocess_intents()` function is called in both `app.py` (line 296, live users) and `test_benchmark.py` (line 622, evaluation). One function, two consumers — no divergence."

**X-Factor:** ML + rules hybrid that is measurably better than the model alone — with reproducible evidence.

**Requirement Satisfied:** spaCy integration (Layer 3 uses PhraseMatcher), complex dialog (Layers 4–5 handle multi-intent conflicts), performance optimisation.

---

## PAGE 5 — Layer 1+2 Deep Dive: Label Mapping & Confidence Filtering

**Screenshot:** In `intent_postprocessor.py`, highlight the `LABEL_MAP` dict (lines 27–42) and the `_confidence_filter()` function (lines 52–62).

**What I Say:**
- "Layer 1 solves a label mismatch: the BERT model was trained with short labels like `hours`, `location`, `contact`, but the engine expects `ask_hours`, `ask_location`, `ask_contact`. The map also nullifies noise labels like `dietary_request`, `clarify`, `compliment` — these don't correspond to real engine intents."
- "Layer 2 drops single-token BIO spans with confidence below 0.55. Multi-token spans are kept even at lower confidence because they're more structurally reliable."
- "Together, Layers 1 + 2 eliminate model noise before the downstream layers even see the predictions."

**X-Factor:** The confidence threshold + token-count heuristic is inspired by NER span-filtering in production NLP pipelines.

**Requirement Satisfied:** Transformers (deep understanding of BERT output structure); performance optimisation.

---

## PAGE 6 — Layer 3 Deep Dive: Slot-Aware Intent Correction

**Screenshot:** In `intent_postprocessor.py`, highlight the `_slot_aware_correction()` function. Also have `slots_spacy.py` open side-by-side showing the `extract_order_items_phrase()` function.

**What I Say:**
- "This is the key insight: spaCy's PhraseMatcher is deterministic and near-perfect for known menu items. If PhraseMatcher finds 'ramen' in the text, we know the user is probably ordering. So Layer 3 uses slot extraction results to **correct** the BERT model's intent prediction."
- "This single layer is responsible for fixing the model's main weakness: `order_create` recall was originally ~5.9% because the training data was sparse for ordering. Post Layer 3, `order_create` recall reaches **100%**."

**X-Factor:** Cross-component feedback loop — the slot extractor informs the intent classifier, not just the other way around.

**Requirement Satisfied:** spaCy (PhraseMatcher as a validation signal), complex dialog (handling ambiguity between ordering and price queries).

---

## PAGE 7 — Layer 4+5 Deep Dive: Keyword Fallback & Conflict Pruning

**Screenshot:** In `intent_postprocessor.py`, show `_KEYWORD_RULES` dict and the `_conflict_resolve()` function, particularly the `_OVERRIDES` dict and the max-2-intent cap.

**What I Say:**
- "Layer 4 is a safety net: for intents the model consistently misses — `complaint`, `reservation_create`, `refund_request` — deterministic keyword rules fire if the model's prediction doesn't already contain the correct intent."
- "Layer 5 resolves logical contradictions. The `_OVERRIDES` dict encodes rules like: if `ask_parking` is present, remove `ask_location` (they share vocabulary but mean different things). We also cap at 2 intents max to prevent over-prediction noise."
- "The priority ranking ensures that if we must prune, we keep action intents (`order_create`, `complaint`) over info intents (`ask_hours`) because actions are higher-stakes."

**X-Factor:** Priority-ranked conflict resolution is inspired by dialogue policy managers in production systems (Rasa Core, Microsoft Bot Framework).

**Requirement Satisfied:** Complex dialog management (multi-intent handling), performance optimisation.

---

## PAGE 8 — My Contribution #3: The Evaluation Framework

**Screenshot:** Open `test_benchmark.py` and show the test case list header and a representative sample of test cases.

**What I Say:**
- "I built a comprehensive evaluation framework: 61 test cases across all 16 intent types, covering single-intent, multi-intent, slot extraction, and edge cases (gibberish, empty input, shouting, slang)."
- "The framework runs each test through the **exact same pipeline** as production. It exports results to CSV, JSON metrics, and a Markdown report automatically."
- "This means the accuracy claim of 96.7% is not hand-waved — it's reproducible: run `python test_benchmark.py` and it regenerates everything."

**X-Factor:** Reproducible, automated evaluation with per-intent F1 — not just "the chatbot seems to work."

**Requirement Satisfied:** Performance evaluation + evidence of correctness.

---

## PAGE 9 — The Results: 96.7% Accuracy, Proven

**Screenshot:** Open `TEST_REPORT.md` showing Section 1 (Executive Summary table with 96.7% accuracy, 87.0% slot accuracy) and Section 2 (the full Per-Intent Precision/Recall/F1 table).

**What I Say:**
- "The headline numbers: **96.7% intent exact-match accuracy** (59/61 tests pass), **87.0% slot extraction accuracy** (20/23 slots correct), and **98.1% single-intent accuracy**."
- "The per-intent F1 table shows perfect 1.000 F1 for 12 out of 15 intents. I know exactly why each of the 2 failures occurred — and they are documented."
- "That transparency is itself evidence of rigorous evaluation."

**X-Factor:** Per-intent F1 scores with diagnostic analysis of every failure — not just top-level accuracy.

**Requirement Satisfied:** Performance metrics; quantitative evaluation; evidence of systematic testing.

---

## PAGE 10 — The Results: Detailed Test Log (CSV)

**Screenshot:** Open `test_results.csv` in a spreadsheet viewer.

**What I Say:**
- "Every single test result is logged in `test_results.csv`: input text, expected intents, predicted intents, intent match status, slot accuracy, and pass/fail. This is the raw evidence behind the metrics."
- "I want to highlight the multi-intent test cases (rows 52–56) — these demonstrate BIO span tagging detecting two different intents from a single sentence, which separates our system from standard multi-class classifiers."

**X-Factor:** Full audit trail — every claim is verifiable by opening this CSV.

**Requirement Satisfied:** Performance evaluation; reproducibility.

---

## PAGE 11 — Beyond Lecture: Follow-Up Detection

**Screenshot:** Open `slots_spacy.py` and show the `detect_followup()` function.

**What I Say:**
- "This goes beyond the lecture material. Follow-up detection uses conversation history to understand contextual messages. If the user says 'also add fries', the system detects 'also' as a follow-up signal, checks that the previous turn was `order_create`, and injects `order_create` into the current intents."
- "Four follow-up types are handled: **affirm**, **negate**, **continue**, and **modify**."
- "Short messages (≤3 words) after an ordering turn are treated as implicit follow-ups if they contain a menu item — so 'fries' by itself after ordering ramen becomes 'order_create' for fries."

**X-Factor:** Context Memory — the system understands that 'fries' alone means 'add fries to my order' because of conversation history.

**Requirement Satisfied:** Complex dialog management; beyond-lecture innovation.

---

## PAGE 12 — Beyond Lecture: POS/Dependency Parsing & SVO Triples

**Screenshot:** Open `slots_spacy.py` and show the `extract_linguistic_features()` function.

**What I Say:**
- "As a further demonstration of spaCy's linguistic pipeline, the system performs POS tagging and dependency parsing on every user message. It extracts verbs, nouns, subjects, objects, and constructs **Subject-Verb-Object triples** from the dependency tree."
- "For example, 'I want 2 ramen' produces the triple `I → want → ramen`. This demonstrates understanding of syntactic structure beyond surface-level pattern matching."

**X-Factor:** SVO triple extraction using dependency parsing — a research-level NLP feature most student projects don't include.

**Requirement Satisfied:** spaCy (POS tagging + dependency parsing); beyond-lecture innovation.

---

## PAGE 13 — Reliability & Grounded Generation

**Screenshot:** Show `chatbot_core.py` with the intent engine call, history logging, LLM calling sequence.

**What I Say:**
- "The system is designed for reliability. Stages 1–3 are fully deterministic: same input always produces same intents, same slots, same engine output."
- "The Phi-2 LLM never invents prices or menu items — it reads the engine's structured output and reformulates it into natural language. This is the same principle as RAG: retrieve first, then generate."
- "If no GPU is available, the system doesn't crash — it uses a CPU fallback that parses engine response blocks directly. The bot still works perfectly."

**X-Factor:** GPU or CPU, the chatbot always works — graceful degradation is a production mindset.

**Requirement Satisfied:** Transformers (Phi-2 generation); performance; reliability.

---

## PAGE 14 — Robustness & Foolproofing (NEW!)

This is the new section covering all the robustness improvements made to handle any user input.

### A. Global Spell Correction

**Screenshot:**

![User types 'bugers' with typo — bot fuzzy-matches to burger and asks which variant](evidence_screenshots/08_fuzzy_bugers.png)

**What I Say:**
- "I built a vocabulary-based spell-correction layer in `spell_correct.py` that runs before anything else in the pipeline. It has a vocabulary of ~120 known words (menu items, commands, greetings, slot values) and uses Levenshtein distance to auto-correct garbled input."
- "So 'hoi' becomes 'hi', 'piza' becomes 'pizza', 'oerdr' becomes 'order'. The system handles these transparently before any NLP processing happens."
- "I also had to solve the 'Sarah vs. salad' problem — the edit distance between names and food items can be tiny. So the corrector is context-aware: when the bot is waiting for a name, it physically disables vocabulary correction so names never accidentally become menu items."

### B. Ambiguous Item Handling

**Screenshot:**

![User types 'pizza' — bot asks 'Which pizza — margherita or pepperoni?'](evidence_screenshots/03_ambiguous_pizza.png)

**What I Say:**
- "Previously, typing 'pizza' would auto-select a random variant. Now the system detects when a keyword maps to multiple menu items and asks the user to clarify. 'Which pizza — margherita or pepperoni?'"
- "Same for 'burger' — the system asks 'chicken or beef?' instead of guessing."

### C. Confirmation Flow & Context Awareness

**Screenshot:**

![User types 'done' — bot prompts for next missing field (dine in/takeaway/delivery)](evidence_screenshots/05_done_confirmation.png)

**What I Say:**
- "When the user says 'done', 'yes', or 'ok', the system now detects these as confirmations and guides the user to the next missing field — dining mode, payment, or final checkout."
- "The system always knows what to ask next based on conversation state."

### D. Correction Commands

**Screenshot:**

![User types 'undo' — bot removes last cart item; 'start over' clears everything](evidence_screenshots/10_undo_start_over.png)

**What I Say:**
- "If the user makes a mistake, they now have clear commands: 'change my name' resets their name, 'undo' removes the last cart item, and 'start over' clears everything and starts fresh."
- "The user always has a way to recover from errors — no matter what."

### E. Crash Protection & Graceful Handling

**Screenshot:**

![User types gibberish — bot handles gracefully without crashing](evidence_screenshots/09_gibberish_graceful.png)

**What I Say:**
- "Input normalization strips extra whitespace, caps length at 500 characters, and the entire processing pipeline is wrapped in try-except crash protection."
- "No matter what garbage input you send — random characters, keyboard smashes, empty messages — the bot recovers gracefully and guides you back into the conversation."

---

## PAGE 15 — End-to-End Order Flow Demo

This page consolidates the complete ordering flow through the chatbot, demonstrating all features working together.

### Step 1: Greeting

![Greeting — 'hi' triggers greeting response](evidence_screenshots/01_greeting_hi.png)

### Step 2: Ambiguous Item → Resolution

![Pizza ambiguity resolved — user picks pepperoni](evidence_screenshots/04_pepperoni_resolved.png)

### Step 3: Order Complete

![Order complete with name, dining mode, payment all set](evidence_screenshots/06_cash_order_complete.png)

### Step 4: Menu Browsing

![Full menu display with categories and prices](evidence_screenshots/07_show_menu.png)

---

## PAGE 16 — Limitations, Mitigations & Close

**Screenshot:** Open `TEST_REPORT.md` Section 6 (Known Limitations table).

**What I Say:**
- "Every system has limitations, and ours are documented honestly. (1) `order_create` ↔ `ask_price` confusion on overlapping vocabulary — mitigated by Layer 3's price-keyword check. (2) Informal quantities ('a couple') default to qty=1 because the regex only matches digits. (3) Very short inputs like 'hi' can confuse the BIO tagger — mitigated by keyword fallback in Layer 4."
- "These are known, documented, and we've implemented mitigations for each. That's the engineering mindset."
- "In summary: trained BERT classifier → spaCy PhraseMatcher → 5-layer post-processing → 16-handler engine → spell correction → Phi-2 grounded generation. **96.7% accuracy, 87% slot extraction, reproducible, truly robust, and always works.** Thank you."

**X-Factor:** Honest limitation analysis with concrete mitigations demonstrates engineering maturity.

**Requirement Satisfied:** Performance analysis; critical evaluation.

---

---

# SECTION B — 5-MINUTE VIDEO RUN-SHEET (0:00–5:00)

> **Target narration:** ~700 words. Time codes are guidelines, not rigid.

---

### 0:00–0:20 — Hook & Problem Statement (Page 1)

**Show:** Pipeline diagram from `PIPELINE_AT_A_GLANCE.md`

**Say (~40 words):**
"A restaurant chatbot needs to understand intent, extract details, act on them, and respond naturally. No single ML technique does all four. We built a 4-stage hybrid pipeline — trained BERT, spaCy PhraseMatcher, a 16-handler engine, and Phi-2 LLM — that composes them together."

---

### 0:20–0:50 — Architecture Flow (Page 2)

**Show:** ASCII pipeline in `SYSTEM_OVERVIEW.md` (scroll slowly through the 4 stages)

**Say (~60 words):**
"Here's how a single message flows through the system. The trained BERT model tags each word with BIO labels — detecting multiple intents per message. spaCy extracts the slots: menu items, quantities, names, dining mode. The intent engine dispatches to the correct handler and mutates the cart. Finally, Phi-2 takes the structured output and generates a natural reply."

---

### 0:50–1:30 — My Streamlit UI (Page 3)

**Show:** Live Streamlit app running, sidebar with loaded models, order tracker

**Say (~80 words):**
"I built the entire Streamlit frontend — 337 lines. The sidebar has one-click model loading with auto-checkpoint discovery, live status badges, and a real-time order tracker showing cart items with prices. Session state manages everything across Streamlit's reruns."

---

### 1:30–2:30 — The 5-Layer Post-Processor (Pages 4–7)

**Show:** `intent_postprocessor.py` open, scrolling through each layer

**Say (~140 words):**
"This is the module I'm most proud of. The raw BERT model achieves around 60% accuracy on ordering intents because we had limited training data. My 5-layer post-processing pipeline lifts this to 96.7%.

Layer 1 maps model labels to engine labels and filters noise. Layer 2 drops low-confidence single-token spans using a 0.55 threshold. Layer 3 is the key innovation: if spaCy's PhraseMatcher finds menu items in the text, it corrects the model's intent to `order_create`. Layer 4 applies keyword fallback rules for intents the model consistently misses. Layer 5 resolves contradictions and caps at 2 intents max.

The same function is called in both the live app and the test harness — one pipeline, zero divergence."

---

### 2:30–3:20 — LIVE DEMO (2 turns + robustness tests)

**Show:** Streamlit app, models loaded

**Demo Turn 1: Order with multi-intent**
- Type: `Hi, I want 2 ramen and what are your hours?`
- Point out: sidebar updates with cart, debug panel shows `[order_create, ask_hours]`
- Say: "Notice how the BIO tagger detected two intents. The sidebar updated with the cart."

**Demo Turn 2: Follow-up**
- Type: `Also a coke`
- Point out: follow-up detection fires, sidebar now shows ramen + coke, total updated
- Say: "The system detects 'also' as a follow-up signal and adds the coke. No need to say 'I want to order' again."

**Demo Turn 3: Robustness**
- Type: `bugers` → shows fuzzy match + ambiguity ("chicken or beef?")
- Type: `undo` → removes last item
- Say: "Even with typos and minimal input, the system handles it without breaking."

---

### 3:20–4:00 — Evidence: Test Results (Pages 8–10)

**Show:** `TEST_REPORT.md` Executive Summary table → Per-Intent F1 table

**Say (~80 words):**
"The numbers aren't hand-waved — they're reproducible. Run `python test_benchmark.py` and it regenerates everything. 96.7% exact-match across 61 tests, 87% slot accuracy, 12 out of 15 intents at perfect F1. The two failures are documented and understood. Per-intent F1 uses precision and recall — the standard metric in NLU benchmarking."

---

### 4:00–4:30 — Beyond Lecture: Follow-Up, SVO & Spell Correction (Pages 11–14)

**Show:** `slots_spacy.py` → `detect_followup()` and `spell_correct.py`

**Say (~60 words):**
"Three features beyond the lecture material. Follow-up detection understands 'also fries' as a continuation. POS + dependency parsing extracts SVO triples. And the spell corrector uses Levenshtein distance to handle typos like 'oerdr' to 'order' — with a context-guard that disables correction when expecting a name so 'Sarah' doesn't become 'salad'."

---

### 4:30–5:00 — Reliability, Limitations & Close (Page 16)

**Show:** Limitations table in `TEST_REPORT.md`, then close with pipeline diagram

**Say (~70 words):**
"The pipeline is grounded: the LLM reads verified engine output, never invents data. It works on GPU and CPU — graceful degradation, not a crash. Limitations are real and documented: the model sometimes confuses ordering with price queries, and informal quantities default to 1. But we've implemented mitigations for each. 96.7% accuracy, reproducible, robust, and always-on. Thank you."

---

---

# SECTION C — 2 CONCRETE BEFORE → AFTER CASES

---

## Case 1: Raw Model Wrong → Post-Processor Fixes It

### The Problem
The BERT model was trained on ~200 examples. For ordering messages, it frequently predicts `browse_menu` or `unknown` instead of `order_create`.

### Input
```
"Can I get a chicken burger please"
```

### BERT Raw Prediction (Before Post-Processing)
```
Raw intents:  ["browse_menu"]
Intent spans: [{"intent": "browse_menu", "token_count": 3, "avg_confidence": 0.72}]
```

### After Post-Processing
```
Layer 1 (Label Map):     ["browse_menu"]          — no change
Layer 2 (Confidence):    ["browse_menu"]          — high enough to keep
Layer 3 (Slot-Aware):    ["order_create"]         — PhraseMatcher finds "chicken burger" → 
                                                     replaces browse_menu with order_create ✅
Layer 4 (Keyword):       ["order_create"]         — "can i get" matches order signal
Layer 5 (Conflicts):     ["order_create"]         — clean
```

### Final Result
```
Predicted: ["order_create"]    ← CORRECT ✅
Slots:     {"items": [{"item": "chicken burger", "qty": 1}]}
```

---

## Case 2: Typo + Ambiguity → Spell Correction + Clarification (NEW!)

### The Problem
User types a garbled, misspelled input. Previously this would either crash or return a generic "I don't understand".

### Input
```
"bugers"
```

### What Happens
```
Step 1 — spell_correct.py:
  - "bugers" not in vocabulary
  - Levenshtein distance to "burger" = 1 (closest match)
  - Corrected text: "burger"

Step 2 — slots_spacy.py:
  - PhraseMatcher finds "burger" matches: chicken burger, beef burger
  - Multiple matches → returns ambiguous flag

Step 3 — intent_engine.py:
  - Detects ambiguity
  - Bot asks: "Which burger would you like? We have chicken burger or beef burger."
```

### Final Result
```
Bot: "Which burger would you like? We have chicken burger or beef burger."
← User guided to correct choice — no crash, no wrong guess ✅
```

### Screenshot Evidence

![Fuzzy matching + ambiguity handling for 'bugers'](evidence_screenshots/08_fuzzy_bugers.png)

---

---

# SECTION D — 6 TOUGH PROFESSOR QUESTIONS + ANSWERS

---

### Q1: "Why post-processing instead of just retraining the model with more data?"

**Strong Answer:**
"Three reasons. First, **data cost**: collecting and labelling high-quality training data for 16 intents is time-intensive. We had ~200 examples — enough for the model to learn feature representations, but not enough for high recall on every intent. Post-processing gives us 96.7% accuracy today, without waiting for more data.

Second, **debuggability**: when a prediction is wrong, I can trace it through 5 discrete layers and identify exactly where it failed. With a retrained model, the fix is opaque — you just hope more data helps.

Third, **speed and resource cost**: retraining BERT takes GPU hours and risks accuracy regression. Post-processing rules can be deployed in minutes and tested immediately against the 61-test benchmark."

---

### Q2: "Why PhraseMatcher + NER instead of pure regex or pure NER?"

**Strong Answer:**
"Pure regex is fragile for multi-word entities. Matching 'chicken burger' in 'Can I get a chicken burger please' requires accounting for case, partial matches, and word boundaries.

Pure NER (spaCy's built-in entity detection) is trained for general entities — PERSON, ORG, GPE — not for menu items like 'pad thai' or 'sushi platter'. To use NER for menu items, I'd need to train a custom NER model.

PhraseMatcher gives us the best of both: it's **hash-based** (O(1) lookup), **case-insensitive** via `attr=LOWER`, and handles multi-word patterns natively. I use NER only for customer names (where spaCy's PERSON model works well) and PhraseMatcher for everything domain-specific."

---

### Q3: "How do you prevent the Phi-2 LLM from hallucinating prices or menu items?"

**Strong Answer:**
"By architecture, not by prompt engineering. The LLM never has access to the raw menu database directly. The intent engine looks up prices from a verified knowledge base, constructs response blocks with exact prices, and passes them to the LLM. The system instruction tells the LLM to reformulate the engine output, not generate new data.

So the LLM's job is **paraphrasing verified data**, not generating new data. This is the same principle as RAG used by systems like Copilot.

On CPU where Phi-2 is too slow, the fallback mode skips the LLM entirely and returns the engine's response blocks directly — no hallucination risk at all."

---

### Q4: "How does the system handle multi-intent messages like 'order ramen and check hours'?"

**Strong Answer:**
"We use BIO sequence labelling. Each word gets a tag: 'I want 2 ramen' → `B-order_create I-order_create I-order_create I-order_create`. 'check your hours' → `B-ask_hours I-ask_hours I-ask_hours`. The model identifies two separate B-tags, which means two separate intent spans. Both handlers fire sequentially and results get fed to Phi-2 for a unified response.

Our test suite includes 5 multi-intent test cases. The one failure is Test #55 where `browse_menu` was pruned by Layer 5 — a known tradeoff to prevent over-prediction."

---

### Q5: "How do you handle typos and misspelled user input?"

**Strong Answer:**
"I built a custom spell-correction layer in `spell_correct.py`. It maintains a vocabulary of ~120 known words — menu items, command words, greetings, slot values. For each unknown word in user input, it computes Levenshtein edit distances against the vocabulary and picks the closest match.

The tricky part was handling names. The edit distance between 'Sarah' and 'salad' is tiny, so the corrector was accidentally turning names into food items. I added a context-guard: when the system is waiting for a name input, it physically disables the spell corrector. So names are protected and everything else is corrected.

This works alongside the fuzzy matching in `slots_spacy.py` which handles plural stripping ('burgers' → 'burger') and ambiguity detection ('burger' → 'which variant?')."

---

### Q6: "What are the system's main limitations and how would you address them?"

**Strong Answer:**
"Three limitations I've identified through the 61-test benchmark:

**1. Order ↔ Price confusion.** 'How much is the ramen' and 'I want ramen' share the word 'ramen'. Layer 3 mitigates this with a price-keyword check, but edge cases remain. **Next step:** More contrastive training examples.

**2. Informal quantities.** 'A couple of ramen' — the regex only catches digits, so these default to qty=1. **Next step:** Pattern dictionary mapping 'a couple' → 2, 'a few' → 3.

**3. Out-of-vocabulary items.** If a user orders 'avocado toast' (not on our menu), PhraseMatcher returns nothing. **Next step:** Add a `menu_not_found` sub-handler using cosine similarity with spaCy word vectors.

No chatbot handles 100% of inputs perfectly. But the 5-layer post-processing gives us a modular place to add each fix — I don't need to retrain the model for any of these improvements."

---

---

# APPENDIX — FILE REFERENCE MAP

| File | Lines | Contribution | Pages |
|------|-------|-------------|-------|
| `app.py` | 337 | ✅ Streamlit UI, sidebar, session state | 3, demo |
| `intent_postprocessor.py` | 420+ | ✅ 5-layer pipeline + keyword rules | 4–7 |
| `test_benchmark.py` | 865 | ✅ 61-test evaluation framework | 8–10 |
| `spell_correct.py` | 170+ | ✅ **NEW** — vocabulary spell correction | 14 |
| `chatbot_core.py` | 160+ | ✅ Orchestration + correction commands + crash protection | 13, 14 |
| `slots_spacy.py` | 400+ | ✅ Follow-up detection + POS/SVO + fuzzy matching + ambiguity | 6, 11, 12, 14 |
| `phi_llm.py` | 170+ | ✅ Smart contextual fallback + confirmation detection | 13 |
| `TEST_REPORT.md` | 150 | ✅ Generated by test framework | 9, 16 |
| `test_metrics.json` | 91 | ✅ Generated by test framework | 9 |
| `test_results.csv` | 63 | ✅ Generated by test framework | 10 |
| `PIPELINE_AT_A_GLANCE.md` | 62 | Shared doc | 1 |
| `SYSTEM_OVERVIEW.md` | 262 | Shared doc | 2, 13 |

> **Buddy scope (DO NOT CLAIM):** core intent engine handlers + KB (`intent_engine.py`), base BERT training scripts/checkpoint creation (`intent_span_model/`), initial slot PhraseMatcher skeleton (lines 1–140 of `slots_spacy.py`), base Phi wrapper skeleton (`phi_llm.py`).
