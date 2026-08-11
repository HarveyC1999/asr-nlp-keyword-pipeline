"""Simple keyword extraction for transcript text."""

from __future__ import annotations

import re
from collections import OrderedDict


class KeywordExtractor:
    """Rule-based extraction that remains deterministic and testable."""

    def __init__(self) -> None:
        self.patterns = [
            ("cancel policy", "cancel policy"),
            ("high premium", "high premium"),
            ("premium is too high", "high premium"),
            ("too high", "too high"),
            ("policy", "policy"),
            ("premium", "premium"),
        ]

    def extract(self, text: str) -> list[str]:
        if not text:
            return []

        cleaned = text.lower().strip()
        found: OrderedDict[str, None] = OrderedDict()

        for phrase, label in self.patterns:
            if phrase in cleaned:
                found[label] = None

        # Add phrase-style fallback for common domain terms.
        bonus_phrases = re.findall(
            r"\b(?:cancel|renew|upgrade|premium|policy|claim|coverage|high)\b(?:\s+\b(?:policy|premium|claim|coverage|plan|service|high)\b){0,2}",
            cleaned,
        )
        for phrase in bonus_phrases:
            phrase = " ".join(phrase.split())
            if phrase and phrase not in found:
                found[phrase] = None

        return list(found.keys())
