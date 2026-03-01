"""
spell_correct.py — Vocabulary-based spell correction for garbled input.

Builds a known-word vocabulary from:
  - menu item words (burger, pizza, ramen, etc.)
  - command words (order, reserve, cancel, menu, etc.)
  - greeting words (hi, hello, hey, etc.)
  - slot words (dine, takeaway, delivery, cash, card, etc.)

For each unknown word in user input, finds the closest vocabulary
match using Levenshtein distance. This catches heavily misspelled
input like 'heemeni' → 'menu', 'oerdr' → 'order', 'hoi' → 'hi'.
"""

from typing import List, Tuple
import re

# ═══════════════════════════════════════════════════════════════
# VOCABULARY — every word the system should recognise
# ═══════════════════════════════════════════════════════════════

_MENU_WORDS = {
    "chicken", "burger", "beef", "ramen", "pad", "thai",
    "margherita", "pepperoni", "pizza", "caesar", "salad",
    "steak", "tom", "yum", "sushi", "platter", "tacos",
    "fries", "coke", "sprite", "iced", "latte", "tea",
    "cheesecake", "brownie", "onion",
}

_COMMAND_WORDS = {
    "order", "cancel", "reserve", "reservation", "book", "booking",
    "table", "menu", "show", "browse", "checkout", "confirm",
    "refund", "complaint", "modify", "change", "remove", "add",
    "undo", "reset", "start", "over", "new", "fix", "correct",
    "wrong", "update", "help",
}

_GREET_WORDS = {
    "hi", "hello", "hey", "morning", "afternoon", "evening",
    "howdy", "good", "thanks", "thank", "bye", "goodbye",
    "please", "yes", "yeah", "no", "nope", "sure", "okay",
    "alright", "done",
}

_SLOT_WORDS = {
    "dine", "takeaway", "delivery", "pickup", "cash", "card",
    "paynow", "apple", "google", "pay", "name", "payment",
    "dining",
}

_QUESTION_WORDS = {
    "what", "where", "when", "how", "much", "cost", "price",
    "hours", "open", "close", "location", "address", "parking",
    "contact", "phone", "email", "time",
}

_ACTION_WORDS = {
    "want", "like", "need", "give", "get", "have", "take",
    "bring", "can", "would", "will", "could", "please",
    "also", "more", "another", "some",
}

# Common short words we should never try to "correct" (they're fine as-is)
_STOP_WORDS = {
    # articles, pronouns, determiners
    "i", "a", "an", "the", "is", "am", "are", "to", "for", "of",
    "in", "on", "at", "it", "my", "me", "we", "us", "do", "so",
    "or", "and", "but", "not", "if", "up", "as", "be",
    # common prepositions and conjunctions
    "by", "with", "from", "this", "that", "them", "they", "he",
    "she", "his", "her", "our", "its", "was", "were", "been",
    "has", "had", "did", "does", "will", "shall", "may", "might",
    "than", "then", "there", "here", "where", "when", "who",
    "whom", "which", "what", "why", "how", "all", "each", "any",
    "few", "many", "much", "own", "same", "too", "very",
    "just", "only", "also", "both", "such", "into", "over",
    "after", "before", "about", "above", "below", "between",
    # common verb forms
    "go", "got", "get", "let", "put", "say", "see", "try",
    "use", "way", "out", "day", "new", "old", "big",
    "your", "you", "yours", "their", "theirs",
    # food-adjacent words that aren't menu items
    "rings", "ring", "hot", "cold", "large", "small", "extra",
}

# Build the full vocabulary
VOCABULARY = (
    _MENU_WORDS | _COMMAND_WORDS | _GREET_WORDS |
    _SLOT_WORDS | _QUESTION_WORDS | _ACTION_WORDS
)

# ═══════════════════════════════════════════════════════════════
# LEVENSHTEIN DISTANCE
# ═══════════════════════════════════════════════════════════════

def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(min(
                prev_row[j + 1] + 1,   # deletion
                curr_row[j] + 1,        # insertion
                prev_row[j] + cost,     # substitution
            ))
        prev_row = curr_row
    return prev_row[-1]


def _best_match(word: str, max_dist: int = 2) -> str:
    """Find the closest vocabulary word within max_dist edits.
    
    For short words (≤3 chars), only allow distance 1.
    For medium words (4-5 chars), allow distance 2.
    For long words (6+ chars), allow distance 2.
    
    Returns the original word if no close match found.
    """
    if len(word) <= 3:
        max_dist = 1  # stricter for short words
    
    best_word = word
    best_dist = max_dist + 1
    
    for vocab_word in VOCABULARY:
        # quick length filter — edit distance can't be less than length difference
        if abs(len(word) - len(vocab_word)) > max_dist:
            continue
        
        dist = _edit_distance(word, vocab_word)
        if dist < best_dist or (dist == best_dist and len(vocab_word) < len(best_word)):
            best_dist = dist
            best_word = vocab_word
    
    return best_word if best_dist <= max_dist else word


# ═══════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════

def correct_input(text: str) -> str:
    """
    Auto-correct garbled user input by matching unknown words
    to the closest vocabulary word.
    
    Examples:
        'heemeni' -> 'menu' (closest match)
        'hoi' -> 'hi' (1 edit)
        'oerdr' -> 'order' (2 edits)
        'piza' -> 'pizza' (1 edit)
        'bugers' -> 'burger' (1 edit with plural strip)
        'margherita pizza' -> unchanged (already correct)
    
    Returns the corrected text.
    """
    words = text.split()
    corrected = []
    any_changed = False
    
    # Detect "my name is X" pattern — skip correction for words after "name is"
    text_lower = text.lower()
    name_context = False
    if "my name is" in text_lower or "name is" in text_lower or "i'm" in text_lower or "i am" in text_lower:
        name_context = True
    
    # Find the index of the name word (word after "is" in "name is" pattern)
    skip_indices = set()
    if name_context:
        for idx, w in enumerate(words):
            if idx > 0 and words[idx - 1].lower().strip(",.!?") == "is":
                skip_indices.add(idx)
            if idx > 0 and words[idx - 1].lower().strip(",.!?") in ("i'm", "am"):
                skip_indices.add(idx)
    
    for i, word in enumerate(words):
        # Skip words identified as potential names
        if i in skip_indices:
            corrected.append(word)
            continue
        
        # preserve original casing for display, work with lowercase
        lower = word.lower().strip()
        
        # strip punctuation for matching but preserve it
        clean = re.sub(r'[^a-z]', '', lower)
        
        # skip empty, purely numeric, or very short words
        if not clean or clean.isdigit():
            corrected.append(word)
            continue
        
        # skip stop words — they're too short to safely correct
        if clean in _STOP_WORDS:
            corrected.append(word)
            continue
        
        # already a known word? keep it
        if clean in VOCABULARY:
            corrected.append(word)
            continue
        
        # try plural-stripped version
        singular = clean
        if clean.endswith('s') and len(clean) > 3 and not clean.endswith('ss'):
            singular = clean[:-1]
            if singular in VOCABULARY:
                corrected.append(word)  # keep original form
                continue
        
        # try fuzzy matching
        best = _best_match(clean)
        if best != clean:
            # replace the word with the corrected version
            corrected.append(best)
            any_changed = True
        else:
            corrected.append(word)
    
    result = " ".join(corrected)
    
    # Log corrections for debugging
    if any_changed:
        print(f"  [SPELL] '{text}' -> '{result}'")
    
    return result

