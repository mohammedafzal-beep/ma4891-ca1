from typing import Dict, Any, List, Optional
import json

# wrap this in a try-except so the whole app doesn't crash if torch isn't fully set up yet
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception as e:
    print(f"Warning: Failed to import torch/transformers: {e}")
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

class PhiChat:
    def __init__(self, model_name: str = "microsoft/phi-2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.available = False

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            print("HF libraries missing, PhiChat will run in fallback mode.")
            return

        try:
            # CPU SAFETY CHECK: loading a 5GB model on CPU will hang standard laptops and Streamlit
            if torch is None or not torch.cuda.is_available():
                print("No GPU detected! Automatically switching to the smart CPU fallback mode to prevent system freezing.")
                self.available = False
                return

            print(f"Attempting to load {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Use fp16 if we got a GPU
            dt = torch.float16 if torch and torch.cuda.is_available() else None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dt,
            )
            self.model.eval()
            self.available = True
            print("Model loaded ok!")
        except Exception as e:
            # CPU might not handle it or out of memory; we fall back.
            print(f"Error loading model: {e}. Falling back to hardcoded responses.")
            self.available = False

    def reply(self, system: str, user: str, max_new_tokens: int = 160) -> str:
        if not self.available:
            # fallback: deterministic, but smart enough to read the execution blocks!
            try:
                # Find the ENGINE_OUT part in the payload we sent
                engine_part = user.split("ENGINE_OUT:\n")[-1]
                import ast
                # convert the stringified dict back into a real python dict
                engine_dict = ast.literal_eval(engine_part)
                
                blocks = engine_dict.get("combined_response_blocks", [])
                missing = engine_dict.get("need", [])
                core_missing = engine_dict.get("core_missing", [])
                snapshot = engine_dict.get("history_snapshot", {})
                
                reply_text = " ".join(blocks)
                
                # if there's anything missing, tack it onto the end politely
                if missing:
                    # format the missing fields nicely (not raw field names)
                    field_labels = {
                        "order_items": "what you'd like to order",
                        "name": "your name",
                        "dining_mode": "how you'd like to dine (dine in, takeaway, or delivery)",
                        "payment_mode": "how you'd like to pay",
                    }
                    nice_missing = [field_labels.get(f, f) for f in missing]
                    reply_text += f"\n\nBefore we proceed, could you please let me know {', '.join(nice_missing)}?"
                    
                if not reply_text.strip():
                    # ── Context-aware fallback ──
                    # Instead of "I didn't quite catch that", give a helpful 
                    # contextual reply based on what we know about the conversation
                    
                    has_name = snapshot.get("name")
                    has_items = snapshot.get("order_items")
                    has_dining = snapshot.get("dining_mode")
                    has_payment = snapshot.get("payment_mode")
                    
                    # Detect confirmation inputs (yes/ok/sure) and guide to next step
                    user_text_lower = ""
                    try:
                        import json
                        payload_dict = json.loads(user.split("ENGINE_OUT:")[0].strip())
                        user_text_lower = payload_dict.get("user_text", "").lower().strip()
                    except Exception:
                        pass
                    
                    _confirms = {"yes", "yeah", "yep", "yea", "sure", "ok", "okay",
                                 "alright", "yup", "confirm", "done", "checkout",
                                 "that's all", "thats all", "go ahead", "proceed"}
                    is_confirm = user_text_lower in _confirms or any(c == user_text_lower for c in _confirms)
                    
                    # Build a SMART response based on conversation state
                    if core_missing:
                        if has_items and "name" in core_missing and not has_name:
                            reply_text = "Great choices! Could you tell me your name so I can complete the order?"
                        elif has_items and has_name and "dining_mode" in core_missing and not has_dining:
                            reply_text = "Would you like to **dine in**, **takeaway**, or **delivery**? 🪑"
                        elif has_items and has_name and has_dining and "payment_mode" in core_missing:
                            reply_text = "How would you like to pay? We accept **cash**, **card**, **PayNow**, **Apple Pay**, or **Google Pay**. 💳"
                        elif "order_items" in core_missing and has_name:
                            reply_text = "What would you like to order? We have burgers, pizza, ramen, salads, and more! 🍕"
                        elif "name" in core_missing:
                            reply_text = "I'd love to help! Could you start by telling me your name?"
                        else:
                            missing_labels = {
                                "order_items": "what you want to order",
                                "name": "your name",
                                "dining_mode": "dine in/takeaway/delivery",
                                "payment_mode": "payment method",
                            }
                            nice = [missing_labels.get(m, m) for m in core_missing]
                            reply_text = f"I still need: {', '.join(nice)}. Could you help me with those?"
                    elif has_items and has_name and has_dining and has_payment:
                        # ALL fields filled — order is ready!
                        if is_confirm:
                            reply_text = "Your order is confirmed! 🎉 We'll get it ready for you right away. Thank you!"
                        else:
                            reply_text = "Your order looks complete! Would you like to **confirm** your order, or make any changes?"
                    elif is_confirm and has_items:
                        # User confirms but some fields still missing — guide them
                        if not has_dining:
                            reply_text = "Almost there! Would you like to **dine in**, **takeaway**, or **delivery**? 🪑"
                        elif not has_payment:
                            reply_text = "Just one more thing — how would you like to pay? **Cash**, **card**, or **PayNow**? 💳"
                        else:
                            reply_text = "Your order looks good! Anything else you'd like to add?"
                    else:
                        reply_text = ("I'm here to help! You can:\n"
                                     "• **Order food** — just tell me what you'd like (e.g. '2 margherita pizza')\n"
                                     "• **Book a table** — say 'reserve a table'\n"
                                     "• **See the menu** — say 'show menu'\n"
                                     "• **Ask about hours, location, or parking**")
                    
                return reply_text
            except Exception as e:
                return "I can help! Just tell me what you want to order or if you need a reservation."

        # Basic prompt template that works reasonably well for Phi
        prompt_str = f"### System\n{system}\n\n### User\n{user}\n\n### Assistant\n"
        input_ids = self.tokenizer(prompt_str, return_tensors="pt")

        with torch.no_grad():
            output_tokens = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,     # a little bit of randomness makes it feel less robotic
                temperature=0.7,
                top_p=0.9,
            )
            
        decoded_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        # hacky way to extract just the assistant part, but it works
        parts = decoded_text.split("### Assistant\n")
        return parts[-1].strip() if len(parts) > 1 else decoded_text.strip()


def build_system_instruction(history: Dict[str, Any]) -> str:
    # Must instruct the LLM to ask for these exact missing fields
    # 1) name 2) dining_mode 3) order_items+qty 4) payment_mode
    return (
        "You are a restaurant chatbot.\n"
        "You must respond naturally and helpfully.\n"
        "Before confirming an order or reservation, ensure these 4 fields are mapped in history:\n"
        "1) name, 2) dining_mode (dine_in/pickup/delivery), 3) food item(s) with quantity, 4) payment_mode.\n"
        "If one or more of these 4 fields are missing, ask the user for the missing info in natural language.\n"
        "Use conversation history to avoid asking repeated questions.\n"
        "Be concise.\n"
    )

def build_user_payload(
    user_text: str,
    intents: List[str],
    intent_spans: Optional[List[Dict[str, Any]]],
    history: Dict[str, Any],
) -> str:
    # dump everything into a json block so the LLM can parse the exact state
    payload_data = {
        "user_text": user_text,
        "predicted_intents": intents,
        "intent_spans": intent_spans,
        "history": history,
    }
    return json.dumps(payload_data, ensure_ascii=False, indent=2)
