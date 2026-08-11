"""Minimal CLI entry point for running the ASR pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_nlp_pipeline.extraction.keyword_extractor import KeywordExtractor
from asr_nlp_pipeline.pipeline import Pipeline
from asr_nlp_pipeline.preprocessing.text_normalizer import TextNormalizer


class DemoTranscriber:
    """A minimal transcriber implementation for the CLI in local testing environments."""

    def transcribe(self, audio_path: str) -> str:
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return "Customer wants to cancel policy because the premium is too high."


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ASR + NLP keyword pipeline.")
    parser.add_argument("audio_path", help="Path to an audio file")
    parser.add_argument("--language", default="zh", help="Language hint passed to the ASR runtime")
    parser.add_argument("--output", help="Optional JSON path to save the result")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = Pipeline(
        transcriber=DemoTranscriber(),
        normalizer=TextNormalizer(),
        extractor=KeywordExtractor(),
    )
    result = pipeline.run(str(audio_path))

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
