import json

from agent.graph import action_reply


def item(data):
    return {"source": "tool:open_ticket", "text": json.dumps(data)}


def test_created_ticket_reply_is_built_from_the_result():
    reply = action_reply([item({"ticket_id": "TICKET-1001", "customer_id": "C002"})])
    assert "TICKET-1001" in reply and "C002" in reply and "[tool:open_ticket]" in reply


def test_rejected_ticket_reply_says_nothing_was_created():
    reply = action_reply([item({"status": "rejected"})])
    assert "No ticket was created" in reply


def test_failed_ticket_reply_does_not_leak_the_raw_error():
    reply = action_reply([item({"error": "No customer with id C999"})])
    assert "couldn't open the ticket" in reply and "C999" not in reply


def test_no_action_means_no_template():
    assert action_reply([{"source": "tool:lookup_customer", "text": "{}"}]) is None