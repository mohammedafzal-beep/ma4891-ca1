"""
app.py
Streamlit frontend for the restaurant chatbot.
Hooks into the trained BIO intent classifier + spaCy slots + Phi-2 LLM.
Most of the heavy lifting happens in chatbot_core.py, this is just the UI layer.
"""
import os
import sys
import streamlit as st

# quick hack so python can find our local modules without a proper package setup
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from chatbot_core import init_history, process_turn
from phi_llm import PhiChat
from intent_model_adapter import (
    load_span_intent_model, predict_span_intents,
    load_multilabel_intent_model, predict_multilabel_intents,
)
from intent_postprocessor import postprocess_intents
from intent_engine import REST_KB
from spell_correct import correct_input

# ---- page config (streamlit boilerplate) ----
st.set_page_config(
    page_title="Sakura Grill 🌸 — Restaurant Chatbot",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- custom CSS to make it look less like a default streamlit app ----
# spent way too long on this tbh
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main header */
.main-header {
    text-align: center;
    padding: 1.2rem 0 0.8rem;
    background: linear-gradient(135deg, #ff6b9d 0%, #c44569 50%, #e74c3c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.sub-header {
    text-align: center;
    color: #888;
    font-size: 0.95rem;
    margin-top: -0.8rem;
    margin-bottom: 1.5rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0e0 !important;
}

/* Status badges */
.stage-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stage-active {
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: white;
}

.stage-done {
    background: linear-gradient(135deg, #6c5ce7, #a29bfe);
    color: white;
}

.stage-pending {
    background: #333;
    color: #888;
}

/* Sentiment indicator */
.sentiment-box {
    padding: 10px 15px;
    border-radius: 12px;
    text-align: center;
    font-weight: 600;
    margin: 8px 0;
}

.sentiment-positive { background: #00b89422; border: 1px solid #00b894; color: #00b894; }
.sentiment-neutral  { background: #fdcb6e22; border: 1px solid #fdcb6e; color: #fdcb6e; }
.sentiment-negative { background: #e74c3c22; border: 1px solid #e74c3c; color: #e74c3c; }

/* Model status badge */
.model-loaded {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: linear-gradient(135deg, #00b894, #00cec9);
    color: white;
    margin: 4px 0;
}

.model-pending {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    background: #e74c3c33;
    color: #e74c3c;
    border: 1px solid #e74c3c;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ---- session state setup (streamlit reruns the whole script each time, so we need this) ----
if "history" not in st.session_state:
    st.session_state["history"] = init_history()
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "debug_log" not in st.session_state:
    st.session_state["debug_log"] = []
if "last_intents" not in st.session_state:
    st.session_state["last_intents"] = []
if "last_spans" not in st.session_state:
    st.session_state["last_spans"] = None


# ---- sidebar: model controls + live order tracker ----
with st.sidebar:
    st.markdown("## 🌸 Sakura Grill")
    st.markdown("---")

    # model loading section — user clicks a button and we spin up the heavy stuff
    st.markdown("### 🤖 ML Models")

    # try to find the checkpoint automatically so the user doesn't have to type it
    default_ckpt = os.path.join(current_dir, "intent_span_model", "checkpoint-1013")
    if not os.path.isdir(default_ckpt):
        # fallback: scan for any checkpoint folder we can find
        span_dir = os.path.join(current_dir, "intent_span_model")
        if os.path.isdir(span_dir):
            subs = [d for d in os.listdir(span_dir) if d.startswith("checkpoint")]
            if subs:
                default_ckpt = os.path.join(span_dir, subs[0])

    intent_mode = st.selectbox("Intent classifier", ["Span BIO Model", "Multi-Label Classifier"])
    # Show relative path in UI to keep it clean
    display_path = os.path.relpath(default_ckpt, current_dir) if os.path.isdir(default_ckpt) else default_ckpt
    intent_model_dir = st.text_input("Model checkpoint", value=display_path)
    phi_model_name = st.text_input("Phi model name", value="microsoft/phi-2")
    threshold = st.slider("Confidence threshold (multi-label)", 0.0, 1.0, 0.5, 0.05)

    if st.button("⚡ Load Models", use_container_width=True):
        with st.spinner("Loading models... this may take a minute ☕"):
            try:
                # Resolve relative path back to absolute for model loading
                abs_model_dir = os.path.join(current_dir, intent_model_dir) if not os.path.isabs(intent_model_dir) else intent_model_dir
                if intent_mode == "Multi-Label Classifier":
                    tokenizer, model = load_multilabel_intent_model(abs_model_dir)
                    st.session_state["intent_tok"] = tokenizer
                    st.session_state["intent_mdl"] = model
                    st.session_state["intent_mode"] = "multi"
                else:
                    tokenizer, model = load_span_intent_model(abs_model_dir)
                    st.session_state["intent_tok"] = tokenizer
                    st.session_state["intent_mdl"] = model
                    st.session_state["intent_mode"] = "span"

                st.session_state["phi"] = PhiChat(model_name=phi_model_name)
                st.success("Models loaded successfully! 🎉")
            except Exception as e:
                st.error(f"Failed to load models: {str(e)}")

    # little status badges so you can see at a glance what's loaded
    if "intent_tok" in st.session_state:
        mode_label = st.session_state.get("intent_mode", "span").upper()
        st.markdown(f'<span class="model-loaded">✅ Intent: {mode_label}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="model-pending">⏳ Intent: Not loaded</span>', unsafe_allow_html=True)

    if "phi" in st.session_state:
        phi_obj = st.session_state["phi"]
        gpu_tag = "GPU" if phi_obj.available else "CPU fallback"
        st.markdown(f'<span class="model-loaded">✅ Phi-2 ({gpu_tag})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="model-pending">⏳ Phi-2: Not loaded</span>', unsafe_allow_html=True)

    st.markdown("---")

    # live order tracker — pulls straight from the history dict
    st.markdown("### 🛒 Current Order")
    hist = st.session_state["history"]
    st.markdown(f"**Name:** {hist.get('name') or '—'}")
    st.markdown(f"**Dine Mode:** {(hist.get('dining_mode') or '—').replace('_', ' ').title()}")
    st.markdown(f"**Payment:** {(hist.get('payment_mode') or '—').replace('_', ' ').title()}")

    cart = hist.get("cart", [])
    if cart:
        st.markdown("**Cart:**")
        menu_prices = REST_KB.get("menu", {})
        running_total = 0.0
        for item in cart:
            name = item["item"]
            qty = item["qty"]
            price = menu_prices.get(name, {}).get("price", 0)
            line_total = price * qty
            running_total += line_total
            st.markdown(f"  • {qty}× {name.title()} (${line_total:.2f})")
        st.markdown(f"**Total: ${running_total:.2f}**")
    else:
        st.markdown("_Cart is empty_")

    st.markdown("---")

    # nuke everything and start fresh
    if st.button("🔄 New Order", use_container_width=True):
        st.session_state["history"] = init_history()
        st.session_state["messages"] = []
        st.session_state["debug_log"] = []
        st.session_state["last_intents"] = []
        st.session_state["last_spans"] = None
        st.rerun()


# ---- main chat area ----
st.markdown('<div class="main-header">🌸 Sakura Grill Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Trained Intent Classifier • spaCy PhraseMatcher • Phi-2 Transformer</div>',
    unsafe_allow_html=True,
)

# re-render previous messages (streamlit wipes the page on every rerun)
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🌸"):
        st.markdown(msg["content"])

# gentle nudge if they haven't loaded the models yet
if "intent_tok" not in st.session_state:
    st.info("👈 Click **Load Models** in the sidebar to get started!")

# the actual chat input box at the bottom
user_text = st.chat_input("Type your message here... 💬")

if user_text:
    # slap the user message on screen
    st.session_state["messages"].append({"role": "user", "content": user_text})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_text)

    predicted_intents = []
    found_spans = None

    # run intent prediction if models are loaded, otherwise skip gracefully
    if "intent_tok" in st.session_state and "intent_mdl" in st.session_state:
        try:
            tok = st.session_state["intent_tok"]
            mdl = st.session_state["intent_mdl"]

            if st.session_state.get("intent_mode") == "multi":
                predicted_intents = predict_multilabel_intents(tok, mdl, user_text, threshold=threshold)
                found_spans = None
            else:
                out_dict = predict_span_intents(tok, mdl, user_text)
                predicted_intents = out_dict["intents"]
                found_spans = out_dict["intent_spans"]
        except Exception as e:
            st.error(f"Intent prediction error: {e}")
            predicted_intents = []

    # ── Spell-correct the input BEFORE intent classification ──
    # Skip when the bot is waiting for name input to avoid correcting names
    pending_slots = st.session_state["history"].get("_pending_slots", [])
    if "name" not in pending_slots:
        corrected_text = correct_input(user_text)
    else:
        corrected_text = user_text

    # ── ALWAYS run the 5-layer post-processing ──
    # This is critical: even without a model, the keyword fallback (Layer 4)
    # and slot-aware correction (Layer 3) can identify intents from user text
    predicted_intents = postprocess_intents(predicted_intents, corrected_text, found_spans)

    st.session_state["last_intents"] = predicted_intents
    st.session_state["last_spans"] = found_spans

    # fire the main pipeline: spacy slots -> intent engine -> phi LLM
    active_llm = st.session_state.get("phi", None)

    try:
        with st.spinner("Bot is thinking... 🤔"):
            final_response = process_turn(
                user_text=user_text,
                intents=predicted_intents,
                history=st.session_state["history"],
                intent_spans=found_spans,
                llm=active_llm,
            )

        # show the bot's reply
        st.session_state["messages"].append({"role": "assistant", "content": final_response})
        with st.chat_message("assistant", avatar="🌸"):
            st.markdown(final_response)

    except Exception as e:
        error_msg = f"⚠️ Processing error: {e}"
        st.session_state["messages"].append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant", avatar="🌸"):
            st.error(error_msg)
