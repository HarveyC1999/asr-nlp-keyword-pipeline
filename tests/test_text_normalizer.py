from asr_nlp_pipeline.preprocessing.text_normalizer import TextNormalizer


def test_normalize_text_removes_extra_spaces_and_noise():
    normalizer = TextNormalizer()
    text = "  hello!!! customer wants to cancel   policy due to high premium.  "

    result = normalizer.normalize(text)

    assert result.startswith("hello")
    assert "customer wants to cancel policy due to high premium" in result.lower()
    assert "  " not in result
