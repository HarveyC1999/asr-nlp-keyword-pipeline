"""Core orchestrator for ASR + normalization + keyword extraction."""

from __future__ import annotations

from typing import Any

from .extraction.keyword_extractor import KeywordExtractor
from .preprocessing.text_normalizer import TextNormalizer


class Pipeline:
    def __init__(
        self,
        transcriber: Any | None = None,
        normalizer: TextNormalizer | None = None,
        extractor: KeywordExtractor | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.normalizer = normalizer or TextNormalizer()
        self.extractor = extractor or KeywordExtractor()

    def run(self, audio_path: str) -> dict[str, Any]:
        if self.transcriber is None:
            raise ValueError("A transcriber instance is required to run the pipeline.")

        transcript = self.transcriber.transcribe(audio_path)
        normalized_text = self.normalizer.normalize(transcript)
        keywords = self.extractor.extract(normalized_text)

        return {
            "transcript": transcript,
            "normalized_text": normalized_text,
            "keywords": keywords,
        }
