"""History compression, with several selectable strengths (light / medium /
aggressive) so callers can trade off context-window size against factual
retention instead of a single "compressed or not" toggle.

NOTE ON PROVENANCE: this is our own operationalization of the CAVEMAN
description given for this project (strip function words and redundancy,
keep entities/relations/hard facts) — it is NOT a verified reproduction of
a specific published algorithm; no paper/spec was available when this was
written. If you have the actual CAVEMAN source, replace `compress_log`
with a faithful port and update this note accordingly.

Base method: split into sentences, drop near-duplicate sentences (the kind
of redundancy a long campaign log accumulates - "the door creaked"
mentioned three times), then within each remaining sentence drop function
words (articles, prepositions, conjunctions, auxiliary/copula verbs, common
pronouns and discourse fillers) while always keeping anything that looks
like a "hard fact" — numbers and capitalized tokens (proper nouns, names,
places), since those are the words a closed factual question is actually
going to depend on.

Levels dial two things: how aggressively near-duplicate sentences are
merged (a lower dedup_threshold treats more sentences as "close enough" to
drop), and whether whole sentences carrying no hard data get dropped
outright once function words are stripped (not just shortened).

Shared verbatim with experiments/caveman.py, which is where the
compression-vs-retention trade-off documented in
experiments/paper/noob_paper_en.tex was measured (Medium: ~36% tokens
saved, ~0.52 retention; Aggressive: ~70% saved, ~0.34 retention) before
this module was wired into this live server.
"""

import difflib
import re
from typing import List

FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "for",
    "of", "in", "on", "at", "by", "to", "from", "with", "without", "into",
    "onto", "over", "under", "above", "below", "between", "among", "through",
    "during", "before", "after", "since", "until", "while", "as", "than",
    "that", "which", "who", "whom", "whose", "this", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "ours", "theirs",
    "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
    "there", "here", "then", "than", "too", "also", "very", "quite", "just",
    "not", "no", "only", "own", "same", "such", "again", "further", "once",
    "up", "down", "out", "off", "if", "because", "about", "against", "further",
    "some", "any", "each", "few", "more", "most", "other", "some", "such",
    "all", "both", "each", "either", "neither", "one", "s", "t",
}

# LEVELS[level] -> (dedup_threshold, strip_function_words, drop_generic_sentences)
# Lower dedup_threshold = more sentences treated as duplicates = more merged
# away. "light" only removes near-exact repeats and otherwise leaves prose
# intact (a real, milder point on the curve — not just a synonym for
# "medium"). "aggressive" additionally drops whole sentences that have zero
# hard-data tokens once stripped (pure flavor text with no named entity or
# number in it), on top of the medium-level word stripping.
LEVELS = {
    "light": {"dedup_threshold": 0.95, "strip_function_words": False, "drop_generic_sentences": False},
    "medium": {"dedup_threshold": 0.85, "strip_function_words": True, "drop_generic_sentences": False},
    "aggressive": {"dedup_threshold": 0.75, "strip_function_words": True, "drop_generic_sentences": True},
}


def _is_hard_data(token: str) -> bool:
    stripped = token.strip(".,!?;:\"'()")
    if not stripped:
        return False
    if any(ch.isdigit() for ch in stripped):
        return True
    # Capitalized and not merely the first word of a sentence being a common
    # word -> heuristic proxy for a proper noun / named entity.
    if stripped[0].isupper() and stripped.lower() not in FUNCTION_WORDS:
        return True
    return False


def _has_hard_data(sentence: str) -> bool:
    # Skip leading "Player:"/"GM:" speaker-label tokens — they're capitalized
    # structural markers from the log format, not content, and would
    # otherwise make every sentence trivially "have hard data".
    tokens = [t for t in sentence.split() if not t.endswith(":")]
    return any(_is_hard_data(token) for token in tokens)


def _compress_sentence(sentence: str) -> str:
    kept = []
    for token in sentence.split():
        bare = token.strip(".,!?;:\"'()").lower()
        if _is_hard_data(token) or (bare and bare not in FUNCTION_WORDS):
            kept.append(token)
    return " ".join(kept)


def _dedupe_near_duplicates(sentences: List[str], threshold: float) -> List[str]:
    kept: List[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(
            difflib.SequenceMatcher(None, lowered, k.lower()).ratio() > threshold for k in kept
        ):
            kept.append(sentence)
    return kept


def compress_log(log_text: str, level: str = "medium") -> str:
    if level not in LEVELS:
        raise ValueError(f"Unknown compression level {level!r}; expected one of {list(LEVELS)}")
    params = LEVELS[level]

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", log_text.strip()) if s]
    deduped = _dedupe_near_duplicates(sentences, threshold=params["dedup_threshold"])

    if params["strip_function_words"]:
        processed = [_compress_sentence(s) for s in deduped]
    else:
        processed = deduped

    if params["drop_generic_sentences"]:
        # Check hard-data presence on the *stripped* sentence content, so
        # this only fires alongside strip_function_words.
        processed = [s for s in processed if s and _has_hard_data(s)]

    return " ".join(s for s in processed if s)
