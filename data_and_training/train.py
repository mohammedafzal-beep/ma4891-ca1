import os
import json
import random
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "distilbert-base-uncased"   # CPU-friendly-ish; can swap to bert-base-uncased
TRAIN_PATH = "train_spans.jsonl"        # generated with your spans script
OUTPUT_DIR = "./intent_span_model"
SEED = 42

MAX_LENGTH = 192
TEST_SIZE = 0.1
VAL_SIZE = 0.1

# If your spans never overlap, this method is correct.
# If spans can overlap, you need a different model (multi-head / span classification).


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_intent_set(jsonl_path: str) -> List[str]:
    intents = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for sp in obj.get("intent_spans", []):
                intents.add(sp["intent"])
    return sorted(intents)


def build_bio_label_maps(intents: List[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    labels = ["O"]
    for it in intents:
        labels.append(f"B-{it}")
        labels.append(f"I-{it}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def span_for_token(token_start: int, token_end: int, spans: List[Dict[str, Any]]) -> str:
    """
    Return intent name if token overlaps exactly one intent span.
    We treat overlap as: token range intersects [span.start, span.end)
    """
    hit = None
    for sp in spans:
        s, e = int(sp["start"]), int(sp["end"])
        if token_end <= s or token_start >= e:
            continue
        # overlap
        if hit is not None:
            # overlap with multiple spans (shouldn't happen for clause spans)
            return "__OVERLAP__"
        hit = sp["intent"]
    return hit if hit is not None else "O"


def make_token_labels(
    offsets: List[Tuple[int, int]],
    spans: List[Dict[str, Any]],
    label2id: Dict[str, int],
) -> List[int]:
    """
    Build BIO labels per token.
    Special tokens have offset (0,0) and will be set to -100 (ignored).
    """
    labels = []
    prev_intent = "O"

    for (ts, te) in offsets:
        # Special tokens or padding: ignore
        if ts == 0 and te == 0:
            labels.append(-100)
            prev_intent = "O"
            continue

        intent = span_for_token(ts, te, spans)
        if intent == "__OVERLAP__":
            # If your data ever overlaps, this approach breaks.
            # Ignore token to avoid crashing.
            labels.append(-100)
            prev_intent = "O"
            continue

        if intent == "O":
            labels.append(label2id["O"])
            prev_intent = "O"
            continue

        # BIO decision
        if prev_intent != intent:
            labels.append(label2id[f"B-{intent}"])
        else:
            labels.append(label2id[f"I-{intent}"])
        prev_intent = intent

    return labels


def preprocess(example: Dict[str, Any], tokenizer, label2id):
    text = example["text"]
    spans = example.get("intent_spans", [])

    tok = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )

    offsets = tok["offset_mapping"]
    tok["labels"] = make_token_labels(offsets, spans, label2id)

    # Remove offsets from training inputs (keep only for inference later)
    tok.pop("offset_mapping")
    return tok


def compute_metrics(eval_pred):
    """
    Token-level micro F1 over non-ignored tokens.
    This is simple and fast. (Not seqeval.)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true = []
    y_pred = []

    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l == -100:
                continue
            y_true.append(l)
            y_pred.append(p)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # micro precision/recall/f1
    tp = np.sum((y_pred == y_true) & (y_true != 0))  # not-O correct
    fp = np.sum((y_pred != y_true) & (y_pred != 0))
    fn = np.sum((y_pred != y_true) & (y_true != 0))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # overall token accuracy
    acc = float(np.mean(y_pred == y_true))

    return {"tok_f1": float(f1), "tok_precision": float(precision), "tok_recall": float(recall), "tok_acc": acc}


def main():
    set_seed(SEED)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Missing {TRAIN_PATH}")

    intents = build_intent_set(TRAIN_PATH)
    labels, label2id, id2label = build_bio_label_maps(intents)

    print(f"Intents ({len(intents)}): {intents}")
    print(f"BIO labels ({len(labels)}): first 10 -> {labels[:10]}")

    ds = load_dataset("json", data_files={"data": TRAIN_PATH})["data"]

    ds = ds.train_test_split(test_size=TEST_SIZE, seed=SEED)
    test_ds = ds["test"]
    train_val = ds["train"].train_test_split(test_size=VAL_SIZE, seed=SEED)
    train_ds, val_ds = train_val["train"], train_val["test"]
    dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized = dataset.map(
        lambda ex: preprocess(ex, tokenizer, label2id),
        remove_columns=dataset["train"].column_names
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        seed=SEED,

        # CPU settings
        fp16=False,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        dataloader_num_workers=0,

        # Training control
        num_train_epochs=8,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="tok_f1",
        greater_is_better=True,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[DONE] Training interrupted by user. Model saved.")

    print("\nValidation:")
    print(trainer.evaluate(tokenized["validation"]))

    print("\nTest:")
    print(trainer.evaluate(tokenized["test"]))

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save intent list and label list for decoding
    with open(os.path.join(OUTPUT_DIR, "intents.json"), "w", encoding="utf-8") as f:
        json.dump({"intents": intents, "labels": labels}, f, indent=2)

    print(f"\nSaved -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
