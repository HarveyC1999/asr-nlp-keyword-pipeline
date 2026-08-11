"""Text cleanup and normalization helpers."""

from __future__ import annotations

import re


class TextNormalizer:
    """Normalize transcript text to a more consistent form for downstream keyword extraction."""

    @staticmethod
    def normalize(text: str) -> str:
        if text is None:
            return ""

        normalized = str(text)
        normalized = normalized.replace("\r", " ").replace("\n", " ")
        normalized = normalized.replace("，", ",").replace("。", ".").replace("；", ";")
        normalized = normalized.replace("：", ":").replace("（", "(").replace("）", ")")
        normalized = re.sub(r"[\t\u00A0]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"(?<!\w)[!?.;,]+", "", normalized)
        normalized = re.sub(r"\s+([,.!?;:])", r"\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def __call__(self, text: str) -> str:
        return self.normalize(text)
