from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from agent.guardrails import mask, unmask

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are a helpful telecom customer support assistant. "
    "Keep answers short and friendly."
)


class State(TypedDict):
    question: str          # what the customer typed
    masked_question: str   # the same text with personal data hidden
    mapping: dict          # placeholder -> real value (never sent to the model)
    answer: str            # the model's reply (still contains placeholders)
    final_answer: str      # the reply with real values restored


def mask_node(state: State) -> dict:
    masked, mapping = mask(state["question"])
    return {"masked_question": masked, "mapping": mapping}


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
    graph.add_node("answer", answer_node)
    graph.add_node("unmask", unmask_node)
    graph.add_edge(START, "mask")
    graph.add_edge("mask", "answer")
    graph.add_edge("answer", "unmask")
    graph.add_edge("unmask", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "question": "Hi, I'm John Smith, call me on 07700 900123. My router has no internet."
    })
    print("MODEL SAW    :", result["masked_question"])
    print("MODEL SAID   :", result["answer"])
    print("CUSTOMER GETS:", result["final_answer"])