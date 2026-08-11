from asr_nlp_pipeline.extraction.keyword_extractor import KeywordExtractor


def test_keyword_extractor_finds_policy_and_premium_phrases():
    text = "Customer wants to cancel policy because the premium is too high."
    extractor = KeywordExtractor()

    keywords = extractor.extract(text)

    assert "cancel policy" in keywords
    assert "high premium" in keywords
