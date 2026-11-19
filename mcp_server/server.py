import itertools
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Build the path from this file's location, so it works from any folder
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "customers.json"
CUSTOMERS = {c["id"]: c for c in json.loads(DATA_FILE.read_text(encoding="utf-8"))}

mcp = FastMCP("telecom-tools")
_ticket_numbers = itertools.count(1001)


@mcp.tool()
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer's plan and any open incident by customer ID,
    for example C001. Does not return personal contact details."""
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        return {"error": f"No customer with id {customer_id}"}
    return {
        "id": customer["id"],
        "plan": customer["plan"],
        "open_incident": customer["open_incident"],
    }


@mcp.tool()
def check_line_status(customer_id: str) -> dict:
    """Check the current status of a customer's line. Possible values:
    active, suspended_unpaid, no_signal, roaming_blocked, sim_not_activated."""
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        return {"error": f"No customer with id {customer_id}"}
    return {"customer_id": customer_id, "line_status": customer["line_status"]}


@mcp.tool()
def open_ticket(customer_id: str, summary: str) -> dict:
    """Create a support ticket for a customer and return the ticket ID.
    This changes data, so use it only after a person has approved it."""
    if customer_id not in CUSTOMERS:
        return {"error": f"No customer with id {customer_id}"}
    return {
        "ticket_id": f"TICKET-{next(_ticket_numbers)}",
        "customer_id": customer_id,
        "summary": summary,
    }


if __name__ == "__main__":
    mcp.run()  # talks to the client through standard input and output