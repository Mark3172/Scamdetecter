from classifier import advice_for, display_term


def test_url_placeholder_is_readable():
    assert display_term("urlplaceholder") == "link / URL"


def test_scam_advice_mentions_links():
    text = advice_for("Scam", 0.9)
    assert "link" in text.lower() or "scam" in text.lower()
