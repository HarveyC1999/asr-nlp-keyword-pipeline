import json

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
