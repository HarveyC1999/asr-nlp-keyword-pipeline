"""Minimal CLI entry point for running the ASR pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from asr_nlp_pipeline.extraction.keyword_extractor import KeywordExtractor
from asr_nlp_pipeline.pipeline import Pipeline
from asr_nlp_pipeline.preprocessing.text_normalizer import TextNormalizer
from asr_nlp_pipeline.transcription.transcriber import WhisperTranscriber


class DemoTranscriber:
    """Small deterministic fallback for demo behavior when no real Whisper runtime is desired."""

    def __init__(self, language: str = "zh") -> None:
        self.language = language

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        return "Customer wants to cancel policy because the premium is too high."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ASR + NLP keyword pipeline.")
    parser.add_argument("audio_path", help="Path to an audio file")
    parser.add_argument("--language", default="zh", help="Language hint passed to the ASR runtime")
    parser.add_argument("--model", default="tiny", help="Whisper model name or local model path")
    parser.add_argument("--device", default="cpu", help="Torch/Whisper device, e.g. cpu or cuda")
    parser.add_argument("--output", help="Optional JSON path to save the result")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a deterministic demo transcriber instead of a real Whisper model.",
    )
    args = parser.parse_args(argv)

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcriber = DemoTranscriber(language=args.language) if args.demo else WhisperTranscriber(
        model_size=args.model,
        device=args.device,
        language=args.language,
    )

    pipeline = Pipeline(
        transcriber=transcriber,
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
