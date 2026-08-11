import json
from unittest.mock import patch

from scripts.run_pipeline import DemoTranscriber, main


def test_demo_cli_uses_demo_transcriber_and_keeps_output_structured(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio-data")
    output_path = tmp_path / "result.json"

    exit_code = main([str(audio_path), "--demo", "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["transcript"].startswith("Customer wants to cancel policy")
    assert "cancel policy" in payload["keywords"]


def test_demo_transcriber_uses_mock_content():
    transcriber = DemoTranscriber(language="zh")
    result = transcriber.transcribe("sample.wav")
    assert "cancel policy" in result.lower()


def test_cli_uses_whisper_transcriber_with_runtime_args(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio-data")
    captured = {}

    class FakeWhisperTranscriber:
        def __init__(self, model_size: str, device: str, language: str):
            captured["model"] = model_size
            captured["device"] = device
            captured["language"] = language

        def transcribe(self, audio_path: str, language: str | None = None) -> str:
            return "Customer wants to cancel policy because the premium is too high."

    with (
        patch("scripts.run_pipeline.WhisperTranscriber", FakeWhisperTranscriber),
        patch("scripts.run_pipeline.Pipeline") as pipeline_mock,
    ):
        pipeline_mock.return_value.run.return_value = {
            "transcript": "Customer wants to cancel policy because the premium is too high.",
            "normalized_text": "Customer wants to cancel policy because the premium is too high.",
            "keywords": ["cancel policy", "high premium"],
        }

        exit_code = main([str(audio_path), "--model", "medium", "--language", "en", "--device", "cuda"])

    assert exit_code == 0
    assert captured == {"model": "medium", "device": "cuda", "language": "en"}
    pipeline_mock.return_value.run.assert_called_once_with(str(audio_path))
