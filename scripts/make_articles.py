import ollama
from pathlib import Path

MODEL = "qwen2.5:7b"

TOPICS = [
    "Router has no internet connection",
    "Broadband is slower than expected",
    "Wifi keeps dropping",
    "How to restart your router",
    "Router lights explained",
    "Setting up a new router",
    "Changing your wifi password",
    "No mobile signal at home",
    "SIM card not working",
    "Activating a new SIM",
    "Replacing a lost or damaged SIM",
    "Mobile data not working",
    "Roaming abroad how to enable",
    "Roaming charges explained",
    "Understanding your bill",
    "Disputing a charge on your bill",
    "Late payment and service suspension",
    "Changing your plan",
    "Upgrading to 5G",
    "Cancelling your contract",
    "Moving home and transferring service",
    "Reporting a network outage",
    "Checking for outages in your area",
    "Voicemail setup and problems",
    "Cannot make or receive calls",
    "Text messages not sending",
    "Setting up parental controls",
    "Transferring your number to another provider",
    "How to check your internet speed",
    "Contacting support and ticket status",
]

out = Path("data/articles")
out.mkdir(parents=True, exist_ok=True)

for i, topic in enumerate(TOPICS, start=1):
    filename = topic.lower().replace(" ", "-") + ".md"
    path = out / filename
    if path.exists():
        print(f"[{i}/{len(TOPICS)}] skipping {filename} (already exists)")
        continue

    prompt = (
        f"Write a short help article (150 to 250 words) for a fictional "
        f"telecom company's knowledge base, titled '{topic}'. Use markdown: "
        f"a # title, a one-line summary, numbered troubleshooting steps, and "
        f"a short 'When to contact support' section. Do not mention real "
        f"company names. Do not invent phone numbers."
    )
    reply = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 500},
    )
    path.write_text(reply["message"]["content"], encoding="utf-8")
    print(f"[{i}/{len(TOPICS)}] wrote {filename}")