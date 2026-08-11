"""Transcription interfaces and an optional Whisper-backed implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WhisperTranscriber:
    """Small wrapper around a local Whisper runtime, if available."""

    def __init__(self, model_size: str = "tiny", device: str | None = None, language: str = "zh") -> None:
        self.model_size = model_size
        self.device = device
        self.language = language
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
            raise RuntimeError(
                "faster-whisper is required for local transcription. "
                "Install a compatible runtime or inject a custom transcriber."
            ) from exc

        model_path = None
        if model_path is None:
            # Keep the interface compatible with local runtime use without hard-coding paths.
            self._model = WhisperModel(self.model_size, device=self.device or "cpu", compute_type="int8")
            return self._model

        self._model = WhisperModel(model_path, device=self.device or "cpu", compute_type="int8")
        return self._model

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            from faster_whisper import WhisperModel  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "A local Whisper runtime is not available. For a real transcription run, "
                "install the ASR runtime dependencies and provide a model path."
            )

        model = self._ensure_model()
        segments, _ = model.transcribe(
            str(path),
            language=language or self.language,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text for segment in segments)
