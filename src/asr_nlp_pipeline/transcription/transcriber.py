"""Whisper-backed transcription implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WhisperTranscriber:
    """Wrap a local Whisper model without binding to project-specific paths."""

    def __init__(
        self,
        model_size: str = "tiny",
        model_path: str | Path | None = None,
        device: str | None = None,
        language: str = "zh",
    ) -> None:
        self.model_size = model_size
        self.model_path = str(model_path) if model_path is not None else None
        self.device = device or "cpu"
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
                "Install the project with `pip install -e '.[dev]'` or install faster-whisper manually."
            ) from exc

        if self.model_path:
            self._model = WhisperModel(self.model_path, device=self.device, compute_type="int8")
            return self._model

        self._model = WhisperModel(self.model_size, device=self.device, compute_type="int8")
        return self._model

    def transcribe(self, audio_path: str | Path, language: str | None = None) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        model = self._ensure_model()
        segments, _ = model.transcribe(
            str(path),
            language=language or self.language,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text for segment in segments)
