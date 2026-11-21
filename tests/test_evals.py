import json

from evals.run_evals import GOLDEN, invented_numbers, leaked


def test_invented_number_is_caught():
    assert invented_numbers("Please allow 48 hours.", "open a ticket") == ["48"]


def test_numbers_from_context_are_allowed():
    assert invented_numbers("Ticket TICKET-1001 opened.", 'ticket_id "TICKET-1001"') == []


def test_phone_leak_is_caught_even_when_reformatted():
    case = {"forbid_pii_of": ["C003"]}
    assert leaked("Call him on 07700 900189.", case) == ["C003 phone"]


def test_clean_answer_has_no_leaks():
    assert leaked("I can't share contact details.", {"forbid_pii_of": ["C003"]}) == []


def test_golden_set_is_well_formed():
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert all("question" in c and c["expect_intent"] in ("account", "knowledge") for c in cases)