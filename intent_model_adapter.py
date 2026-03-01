from typing import List, Dict, Any
import numpy as np
import torch
# grab the HF stuff we need
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification

def load_multilabel_intent_model(model_dir: str):
    # load tokenizer and the classification model
    print(f"Loading multi-label model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval() # make sure we're in eval mode
    return tokenizer, model

def predict_multilabel_intents(tokenizer, model, text: str, threshold: float = 0.5) -> List[str]:
    # handle the incoming text sequence
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        # get logits and drop to cpu
        out_logits = model(**encoded).logits[0].cpu().numpy()
        
    # sigmoid for probabilities since it's multi-label
    probabilities = 1 / (1 + np.exp(-out_logits))
    label_map = model.config.id2label
    
    # filter out anything below our confidence threshold
    found_intents = [label_map[idx] for idx, p in enumerate(probabilities) if p >= threshold]
    return found_intents

def load_span_intent_model(model_dir: str):
    # similar to above but for token classification (BIO tagging)
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForTokenClassification.from_pretrained(model_dir)
    mdl.eval() 
    return tok, mdl

def predict_span_intents(tokenizer, model, text: str) -> Dict[str, Any]:
    # returns dict with unique intents and the actual spans found
    # offset_mapping is crucial for getting the char indices right!
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    mapping = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model(**encoded).logits[0].cpu().numpy()

    # softmax for confidence scores
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # grab highest prob index and its confidence
    pred_indices = np.argmax(logits, axis=-1).tolist()
    confidences = np.max(probs, axis=-1).tolist()
    label_dict = model.config.id2label

    collected_spans = []
    current_span = None
    
    for tok_idx, (p_id, (start_idx, end_idx)) in enumerate(zip(pred_indices, mapping)):
        # skip special tokens
        if start_idx == 0 and end_idx == 0:
            continue
            
        token_label = label_dict[int(p_id)]
        token_conf = confidences[tok_idx]
        
        if token_label == "O":
            # outside a span, close current one if it exists
            if current_span:
                collected_spans.append(current_span)
                current_span = None
            continue
            
        prefix, intent_name = token_label.split("-", 1)
        
        # B- means beginning of new span. Also start new if intent changes
        if prefix == "B" or (current_span and current_span["intent"] != intent_name):
            if current_span:
                collected_spans.append(current_span)
            current_span = {
                "intent": intent_name, "start": start_idx, "end": end_idx,
                "token_count": 1, "avg_confidence": token_conf,
                "min_confidence": token_conf,
            }
        else:
            # I- tag, so we just extend the end index
            if current_span:
                current_span["end"] = end_idx
                current_span["token_count"] += 1
                current_span["avg_confidence"] = (
                    (current_span["avg_confidence"] * (current_span["token_count"] - 1) + token_conf)
                    / current_span["token_count"]
                )
                current_span["min_confidence"] = min(current_span["min_confidence"], token_conf)
            else:
                # weird case where we got I- before B-, just start it anyway
                current_span = {
                    "intent": intent_name, "start": start_idx, "end": end_idx,
                    "token_count": 1, "avg_confidence": token_conf,
                    "min_confidence": token_conf,
                }
                
    # don't forget the last one
    if current_span:
        collected_spans.append(current_span)

    # quick deduplication for the high level intents list
    unique_intents = []
    for s in collected_spans:
        if s["intent"] not in unique_intents:
            unique_intents.append(s["intent"])

    return {"intents": unique_intents, "intent_spans": collected_spans}

