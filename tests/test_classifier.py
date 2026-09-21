from classifier import advice_for, display_term, extract_cues, find_highlights


def test_url_placeholder_is_readable():
    assert display_term("urlplaceholder") == "link / URL"


def test_scam_advice_mentions_links():
    text = advice_for("Scam", 0.9)
    assert "link" in text.lower() or "scam" in text.lower()


def test_extract_cues_finds_link_and_urgency():
    text = "URGENT: verify now at http://evil.example/login"
    ids = {cue["id"] for cue in extract_cues(text)}
    assert "url" in ids
    assert "urgency" in ids


def test_highlights_cover_url():
    text = "Pay the fee at http://fee.example now"
    spans = find_highlights(text, [{"term": "link / URL", "raw": "urlplaceholder"}])
    assert spans
    chunk = text[spans[0]["start"]:spans[0]["end"]]
    assert chunk.startswith("http")
