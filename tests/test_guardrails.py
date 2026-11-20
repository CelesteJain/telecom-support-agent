from agent.guardrails import check_answer, mask, unmask


def test_email_is_masked():
    masked, _ = mask("Please write to john.smith@example.com about my bill")
    assert "john.smith@example.com" not in masked
    assert "<EMAIL_ADDRESS_1>" in masked


def test_uk_mobile_is_masked_in_both_formats():
    for number in ["+44 7700 900123", "07700 900854"]:
        masked, _ = mask(f"Call me on {number} please")
        assert number not in masked


def test_person_name_is_masked():
    masked, _ = mask("My name is John Smith and my line is down")
    assert "John Smith" not in masked


def test_customer_id_and_support_text_are_kept():
    text = "Customer C002 has had no signal for 2 days. Restart the router at 192.168.1.1."
    masked, mapping = mask(text)
    assert masked == text
    assert mapping == {}


def test_unmask_restores_original():
    text = "I am Priya Smith, call 07700 900854 or priya.smith@example.com"
    masked, mapping = mask(text)
    assert masked != text
    assert unmask(masked, mapping) == text


def test_answer_without_citation_is_flagged():
    assert "no_citation" in check_answer("Restart your router.", ["wifi-keeps-dropping.md"])


def test_answer_with_citation_passes():
    answer = "Restart your router [wifi-keeps-dropping.md]"
    assert check_answer(answer, ["wifi-keeps-dropping.md"]) == []


def test_pii_in_answer_is_flagged():
    problems = check_answer("Call John Smith on 07700 900123 [a.md]", ["a.md"])
    assert any(p.startswith("pii_in_answer") for p in problems)

def test_made_up_citation_is_flagged():
    problems = check_answer("Restart it [made-up.md]", ["wifi-keeps-dropping.md"])
    assert any(p.startswith("unknown_citation") for p in problems)