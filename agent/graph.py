from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from agent.guardrails import mask, unmask

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are a helpful telecom customer support assistant. "
    "Keep answers short and friendly."
)

CLASSIFY_PROMPT = (
    "Classify the customer's message. Reply with exactly one word.\n"
    "account = the customer asks about their own line, plan, bill or ticket, "
    "or gives a customer ID such as C001.\n"
    "knowledge = a general how-to or troubleshooting question.\n\n"
    "Message: {message}"
)


class State(TypedDict):
    question: str          # what the customer typed
    masked_question: str   # the same text with personal data hidden
    mapping: dict          # placeholder -> real value (never sent to the model)
    intent: str            # "knowledge" or "account"          (new in 6b)
    context: list          # material found for the answer     (new in 6b)
    answer: str            # the model's reply (still contains placeholders)
    final_answer: str      # the reply with real values restored


def mask_node(state: State) -> dict:
    masked, mapping = mask(state["question"])
    return {"masked_question": masked, "mapping": mapping}


def parse_intent(text: str) -> str:
    """Models do not always follow 'exactly one word', so read the reply
    defensively and fall back to the safer choice."""
    return "account" if "account" in text.lower() else "knowledge"


def classify_node(state: State) -> dict:
    reply = ollama.chat(
        model=MODEL,
        messages=[{"role": "user",
                   "content": CLASSIFY_PROMPT.format(message=state["masked_question"])}],
        options={"temperature": 0, "num_predict": 5},
    )
    return {"intent": parse_intent(reply["message"]["content"])}


def route_by_intent(state: State) -> str:
    return state["intent"]


def retrieve_node(state: State) -> dict:
    return {"context": ["(help-article search is added in 6c)"]}


def diagnose_node(state: State) -> dict:
    return {"context": ["(account tools are added in 6d)"]}


def answer_node(state: State) -> dict:
    reply = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state["masked_question"]},
        ],
    )
    return {"answer": reply["message"]["content"]}


def unmask_node(state: State) -> dict:
    return {"final_answer": unmask(state["answer"], state["mapping"])}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("mask", mask_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("answer", answer_node)
    graph.add_node("unmask", unmask_node)

    graph.add_edge(START, "mask")
    graph.add_edge("mask", "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {"knowledge": "retrieve", "account": "diagnose"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("diagnose", "answer")
    graph.add_edge("answer", "unmask")
    graph.add_edge("unmask", END)
    return graph.compile()


if __name__ == "__main__":
    # Part 1: test only the classifier on a few questions (fast, one short call each)
    tests = [
        ("My router has no internet since yesterday", "knowledge"),
        ("How do I change my wifi password?", "knowledge"),
        ("What does a blinking red light on the router mean?", "knowledge"),
        ("Can you check the line status for C002?", "account"),
        ("I was charged twice this month, my ID is C005", "account"),
    ]
    correct = 0
    for question, expected in tests:
        got = classify_node({"masked_question": question})["intent"]
        correct += got == expected
        print("OK  " if got == expected else "MISS", got.ljust(9), "|", question)
    print(f"Classifier: {correct}/{len(tests)} correct\n")

    # Part 2: run the whole graph once and see which branch it took
    app = build_graph()
    result = app.invoke({"question": "Hi, I'm John Smith. Can you check the line for C002?"})
    print("INTENT       :", result["intent"])
    print("CONTEXT      :", result["context"])
    print("CUSTOMER GETS:", result["final_answer"])