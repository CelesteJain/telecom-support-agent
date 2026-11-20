from functools import lru_cache
import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer

# The kinds of personal data we mask. Customer IDs like C001 are NOT personal
# data here: the agent needs them to call the tools.
ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE"]


@lru_cache(maxsize=1)
def get_analyzer():
    # Loading the language model takes a few seconds, so do it once, on first use
    analyzer = AnalyzerEngine()

    # The built-in phone detector ignores numbers it considers invalid, such as
    # the UK numbers reserved for fiction that our fake customers use.
    # A custom pattern catches UK mobile numbers in both common formats.
    uk_mobile = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[Pattern(name="uk_mobile",
                          regex=r"(?:\+44\s?|\b0)7\d{3}\s?\d{6}\b",
                          score=0.6)],
    )
    analyzer.registry.add_recognizer(uk_mobile)
    return analyzer


def mask(text):
    """Replace personal data with numbered placeholders like <PERSON_1>.

    Returns (masked_text, mapping). The mapping lets us restore the original
    values later, in the reply that goes back to the same customer.
    """
    found = get_analyzer().analyze(text=text, language="en", entities=ENTITIES)
    found = sorted(found, key=lambda r: (r.start, -r.score))

    # Two detectors can flag overlapping text; keep only the first of each overlap
    kept, last_end = [], 0
    for r in found:
        if r.start >= last_end:
            kept.append(r)
            last_end = r.end

    mapping, counters, pieces, pos = {}, {}, [], 0
    for r in kept:
        original = text[r.start:r.end]
        placeholder = next((p for p, v in mapping.items() if v == original), None)
        if placeholder is None:
            counters[r.entity_type] = counters.get(r.entity_type, 0) + 1
            placeholder = f"<{r.entity_type}_{counters[r.entity_type]}>"
            mapping[placeholder] = original
        pieces.append(text[pos:r.start])
        pieces.append(placeholder)
        pos = r.end
    pieces.append(text[pos:])
    return "".join(pieces), mapping


def unmask(text, mapping):
    """Put the original values back in place of the placeholders."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


def check_answer(answer, sources, needs_citation=True):
    """Check a model's answer before it is shown. Returns a list of problems;
    an empty list means the answer passed."""
    problems = []
    if needs_citation and not any(name in answer for name in sources):
        problems.append("no_citation")

    # Every [bracketed] citation must be a source we actually gave the model
    cited = set(re.findall(r"\[([^\]]+)\]", answer))
    unknown = sorted(c for c in cited if c not in sources)
    if unknown:
        problems.append("unknown_citation: " + ", ".join(unknown))

    leaked = get_analyzer().analyze(text=answer, language="en", entities=ENTITIES)
    if leaked:
        kinds = sorted({r.entity_type for r in leaked})
        problems.append("pii_in_answer: " + ", ".join(kinds))
    return problems