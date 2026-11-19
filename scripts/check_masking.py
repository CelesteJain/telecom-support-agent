import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.guardrails import mask  # noqa: E402

customers = json.loads((ROOT / "data" / "customers.json").read_text(encoding="utf-8"))

names = phones = emails = ids_kept = 0
for c in customers:
    text = (f"Hi, my name is {c['name']}, ID {c['id']}. "
            f"Call me on {c['phone']} or email {c['email']}. My line is not working.")
    masked, _ = mask(text)
    names += c["name"] not in masked
    phones += c["phone"] not in masked
    emails += c["email"] not in masked
    ids_kept += c["id"] in masked
    if c["name"] in masked:
        print("Name NOT masked:", c["name"])

n = len(customers)
print(f"names masked: {names}/{n}  phones: {phones}/{n}  "
      f"emails: {emails}/{n}  customer IDs kept: {ids_kept}/{n}")

ordinary_text = [
    "Restart the router, wait 30 seconds, then check the lights. The router IP is 192.168.1.1.",
    "I was charged 45 pounds on 12 March for the Unlimited 5G plan. Reference TICKET-1001.",
    "Step 1: Go to Settings. Step 2: Choose Network, then Wi-Fi.",
]
wrongly_changed = sum(mask(t)[0] != t for t in ordinary_text)
print(f"ordinary support sentences wrongly changed: {wrongly_changed}/{len(ordinary_text)}")