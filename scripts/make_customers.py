import json
import random
from pathlib import Path

random.seed(42)  # same fake data every time you run it

FIRST = ["Aarav", "Priya", "Liam", "Sofia", "Noah", "Meera", "Oliver", "Zara",
         "Ethan", "Anika", "Lucas", "Isha", "Mason", "Leela", "Daniel", "Kavya",
         "Henry", "Nina", "Arjun", "Emma"]
LAST = ["Sharma", "Patel", "Smith", "Rossi", "Brown", "Iyer", "Taylor", "Khan",
        "Miller", "Nair"]
PLANS = ["Basic 5G", "Plus 5G", "Unlimited 5G",
         "Home Broadband 100", "Home Broadband 500"]
STATUSES = ["active", "active", "active", "suspended_unpaid",
            "no_signal", "roaming_blocked", "sim_not_activated"]
INCIDENTS = [None, None, None, "Network outage in area",
             "Router firmware fault", "Billing dispute open",
             "SIM replacement pending"]

customers = []
for i in range(1, 21):
    first = FIRST[i - 1]
    last = random.choice(LAST)
    customers.append({
        "id": f"C{i:03d}",
        "name": f"{first} {last}",
        "email": f"{first.lower()}.{last.lower()}@example.com",
        "phone": f"+44 7700 900{random.randint(100, 999)}",
        "plan": random.choice(PLANS),
        "line_status": random.choice(STATUSES),
        "open_incident": random.choice(INCIDENTS),
    })

Path("data").mkdir(exist_ok=True)
with open("data/customers.json", "w", encoding="utf-8") as f:
    json.dump(customers, f, indent=2)

print(f"Wrote {len(customers)} customers to data/customers.json")