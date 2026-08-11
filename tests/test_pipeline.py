from asr_nlp_pipeline.pipeline import Pipeline


class FakeTranscriber:
    def transcribe(self, audio_path):
        return "Customer wants to cancel policy because the premium is too high."


def test_pipeline_orchestrates_transcription_normalization_and_keywords():
    pipeline = Pipeline(transcriber=FakeTranscriber())

    result = pipeline.run("dummy.wav")

    assert result["transcript"] == "Customer wants to cancel policy because the premium is too high."
    assert "customer wants to cancel policy" in result["normalized_text"].lower()
    assert "cancel policy" in result["keywords"]
    assert "high premium" in result["keywords"]
