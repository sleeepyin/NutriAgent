"""Match a VLM food name to an annotated category. Matching never sees the weight."""
from __future__ import annotations

import re

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = str(text).lower().replace("_", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _singularize(text: str, vocab: set[str]) -> str:
    parts = []
    for tok in text.split():
        if tok in vocab:
            parts.append(tok)
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and tok[:-1] in vocab:
            parts.append(tok[:-1])
        else:
            parts.append(tok)
    return " ".join(parts)


def match_name(name: str, labels: list[str]) -> tuple[str | None, str]:
    """Return (annotated label, exact|synonym|none). Longest contained phrase wins."""
    labels = list(dict.fromkeys(labels))
    vocab: set[str] = set()
    for lab in labels:
        vocab.update(normalize(lab).split())
    text = normalize(name)
    if not text:
        return None, "none"
    hits = [lab for lab in labels if normalize(lab) == text]
    if hits:
        hits.sort(key=len)
        return hits[0], "exact"
    singular = _singularize(text, vocab)
    queries = [text] if singular == text else [text, singular]
    for query in queries:
        contained = [
            lab for lab in labels if normalize(lab) and f" {normalize(lab)} " in f" {query} "
        ]
        if contained:
            contained.sort(key=lambda lab: len(normalize(lab)), reverse=True)
            kind = "exact" if query == text and normalize(contained[0]) == text else "synonym"
            return contained[0], kind
        tokens = set(query.split())
        subset = [
            lab
            for lab in labels
            if set(normalize(lab).split()) and set(normalize(lab).split()) <= tokens
        ]
        if subset:
            subset.sort(key=lambda lab: len(normalize(lab).split()), reverse=True)
            return subset[0], "synonym"
    return None, "none"


def self_check() -> None:
    labels = ["apple", "pineapple", "Almond(bowl)", "Egg"]
    assert match_name("apple", labels) == ("apple", "exact")
    assert match_name("green apple", labels)[0] == "apple"
    assert match_name("pineapple slice", labels)[0] == "pineapple"
    assert match_name("almonds in a bowl", labels)[0] == "Almond(bowl)"
    assert match_name("boiled eggs", labels)[0] == "Egg"
    assert match_name("pizza", labels)[0] is None
