from typing import TypedDict

import ollama
from langgraph.graph import END, START, StateGraph

from agent.guardrails import mask, unmask
from agent.tools import run_tools

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are a customer support assistant for a telecom company.\n"
    "Rules:\n"
    "1. Answer ONLY from the context provided. Do not use outside knowledge.\n"
    "2. After each fact, cite its source in square brackets, like [file-name.md].\n"
    "3. If the context does not contain the answer, say you do not have that "
    "information and suggest contacting support.\n"
    "4. Never say you have done something or will do something. "
    "You can only give information.\n"
    "5. Keep answers short. Use numbered steps for instructions."
)

CLASSIFY_PROMPT = (
    "Classify the customer's message. Reply with exactly one word.\n"
    "account = the customer asks about their own line, plan, bill or ticket, "
    "or gives a customer ID such as C001.\n"
    "knowledge = a general how-to or troubleshooting question.\n\n"
    "Message: {message}"
)

# Used when there is nothing to ground an answer on. The model is not called at all.
NO_CONTEXT_REPLY = {
    "knowledge": "I couldn't find this in our help articles. "
                 "Please contact support and they will help you.",
    "account": "I couldn't find account details. Please share your customer ID "
               "(for example C001), or contact support.",
}


class State(TypedDict):
    question: str          # what the customer typed
    masked_question: str   # the same text with personal data hidden
    mapping: dict          # placeholder -> real value (never sent to the model)
    intent: str            # "knowledge" or "account"
    context: list          # items like {"source": "file.md", "text": "..."}   (changed in 6c)
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
    # Imported here so the search models only load when a question needs them
    from rag.search import search

    # We search with the masked question, so personal data never reaches the search either
    return {"context": search(state["masked_question"])}


def diagnose_node(state: State) -> dict:
    # The model decides which read-only MCP tools to call; our code runs them
    return {"context": run_tools(state["masked_question"], MODEL)}


def format_context(context: list) -> str:
    return "\n\n".join(f"[{item['source']}]\n{item['text']}" for item in context)


def answer_node(state: State) -> dict:
    # Nothing found: do not ask the model, which would only improvise
    if not state["context"]:
        return {"answer": NO_CONTEXT_REPLY[state["intent"]]}

    user_message = (
        f"Context:\n{format_context(state['context'])}\n\n"
        f"Customer question: {state['masked_question']}"
    )
    reply = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
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
    app = build_graph()
    questions = [
        "Hi, I'm John Smith. Can you check the line for C002?",
        "What plan is customer C001 on?",
        "Can you check the line for C999?",
        "My bill looks wrong",
    ]
    for question in questions:
        result = app.invoke({"question": question})
        print("QUESTION:", question)
        print("INTENT  :", result["intent"])
        print("CONTEXT :", result["context"])
        print("ANSWER  :", result["final_answer"])
        print()