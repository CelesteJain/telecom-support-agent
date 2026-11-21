import re
from typing import TypedDict

import json
from typing import TypedDict

import ollama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.guardrails import check_answer, mask, unmask
from agent.tools import run_approved_tool, run_tools

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = (
    "You are a customer support assistant for a telecom company.\n"
    "Rules:\n"
    "1. Answer ONLY from the context provided. Do not use outside knowledge.\n"
    "2. After each fact, cite its source in square brackets, like [file-name.md].\n"
    "3. If the context does not contain the answer, say you do not have that "
    "information and suggest contacting support.\n"
    "4. Never say you have done something or will do something, unless the "
    "context shows it was done (for example a ticket ID). "
    "Otherwise you can only give information.\n"
    "5. Keep answers short. Use numbered steps for instructions."
    "6. Do not add details, times, promises or steps that are not in the context.\n"
)

CLASSIFY_PROMPT = (
    "Classify the customer's message. Reply with exactly one word: "
    "account or knowledge.\n"
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

# Shown instead of an answer that failed the checks
SAFE_REPLY = ("Sorry, I can't give a reliable answer to that. "
              "Please contact support and they will help you.")


class State(TypedDict):
    question: str          # what the customer typed
    masked_question: str   # the same text with personal data hidden
    mapping: dict          # placeholder -> real value (never sent to the model)
    intent: str            # "knowledge" or "account"
    context: list          # items like {"source": "file.md", "text": "..."}   (changed in 6c)
    answer: str            # the model's reply (still contains placeholders)
    final_answer: str      # the reply with real values restored
    pending: dict          # a write action waiting for approval, or None   (new in 6e)
    problems: list         # what the answer check found; empty = passed     (new in 6e)


def mask_node(state: State) -> dict:
    masked, mapping = mask(state["question"])
    return {"masked_question": masked, "mapping": mapping}


def parse_intent(text: str) -> str:
    """Read the model's reply defensively, but make surprises visible."""
    lowered = text.lower()
    if "account" in lowered:
        return "account"
    if "knowledge" not in lowered:
        print(f"WARNING: unexpected classifier reply {text!r}, defaulting to knowledge")
    return "knowledge"


CUSTOMER_ID = re.compile(r"\bC\d{3}\b")

def classify_node(state: State) -> dict:
    # Hard rule first: a customer ID always means an account question
    if CUSTOMER_ID.search(state["masked_question"]):
        return {"intent": "account"}
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
    # The model decides which MCP tools to call; our code runs the read-only ones
    context, pending = run_tools(state["masked_question"], MODEL)
    return {"context": context, "pending": pending}


def route_after_diagnose(state: State) -> str:
    return "approval" if state.get("pending") else "answer"


def approval_node(state: State) -> dict:
    pending = state["pending"]
    # interrupt() pauses the whole graph here and hands this dict to the caller.
    # IMPORTANT: when we resume, this node runs again from the top, so nothing
    # above this line may have side effects.
    decision = interrupt({
        "action": pending["tool"],
        "arguments": pending["arguments"],
        "question": "Approve this action?",
    })

    if decision == "approve":
        arguments = dict(pending["arguments"])
        # The model only saw placeholders; restore real values before writing data
        if "summary" in arguments:
            arguments["summary"] = unmask(arguments["summary"], state["mapping"])
        result = run_approved_tool(pending["tool"], arguments)
        item = {"source": f"tool:{pending['tool']}", "text": json.dumps(result)}
    else:
        item = {"source": f"tool:{pending['tool']}",
                "text": json.dumps({"status": "rejected"})}
    return {"context": state["context"] + [item], "pending": None}


def format_context(context: list) -> str:
    return "\n\n".join(f"[{item['source']}]\n{item['text']}" for item in context)

def action_reply(context: list):
    """Build the reply for a write action in CODE, so the model cannot invent
    promises or steps. Returns None when there is no action outcome to report."""
    for item in context:
        if item["source"] != "tool:open_ticket":
            continue
        try:
            data = json.loads(item["text"])
        except ValueError:
            return None
        tag = " [tool:open_ticket]"  # keeps the citation check happy and the source visible
        if data.get("status") == "rejected":
            return "No ticket was created, because the request was not approved." + tag
        if "ticket_id" in data:
            return (f"I've opened ticket {data['ticket_id']} "
                    f"for customer {data['customer_id']}." + tag)
        return ("I couldn't open the ticket. Please check the customer ID "
                "or contact support." + tag)
    return None

def answer_node(state: State) -> dict:
    # Nothing found: do not ask the model, which would only improvise
    if not state["context"]:
        return {"answer": NO_CONTEXT_REPLY[state["intent"]]}

    # An action outcome (ticket opened, rejected, failed) is reported by code, not by the model
    reply_text = action_reply(state["context"])
    if reply_text:
        return {"answer": reply_text}
    
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

def check_node(state: State) -> dict:
    """Hard control: check the answer in code before the customer sees it."""
    sources = [item["source"] for item in state["context"]]
    problems = check_answer(state["answer"], sources,
                            needs_citation=bool(state["context"]))
    if problems:
        return {"problems": problems, "answer": SAFE_REPLY}
    return {"problems": []}

def unmask_node(state: State) -> dict:
    return {"final_answer": unmask(state["answer"], state["mapping"])}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("mask", mask_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("approval", approval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("check", check_node)
    graph.add_node("unmask", unmask_node)

    graph.add_edge(START, "mask")
    graph.add_edge("mask", "classify")
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {"knowledge": "retrieve", "account": "diagnose"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_conditional_edges(
        "diagnose",
        route_after_diagnose,
        {"approval": "approval", "answer": "answer"},
    )
    graph.add_edge("approval", "answer")
    graph.add_edge("answer", "check")
    graph.add_edge("check", "unmask")
    graph.add_edge("unmask", END)
    # The checkpointer saves the state at every step, which is what makes pausing possible
    return graph.compile(checkpointer=InMemorySaver())


def ask(app, question: str, thread_id: str, decide=None) -> dict:
    """Run one question. If the graph pauses for approval, ask the human and resume."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        print("PAUSED  :", request)
        answer = (decide or input)("Approve? (y/n) ").strip().lower()
        decision = "approve" if answer == "y" else "reject"
        result = app.invoke(Command(resume=decision), config)
    return result


if __name__ == "__main__":
    app = build_graph()
    questions = [
        "How do I fix my wifi that keeps dropping?",
        "Hi, I'm John Smith. Customer C002 has had no roaming for days. Please open a ticket.",
    ]
    for number, question in enumerate(questions, start=1):
        result = ask(app, question, thread_id=f"demo-{number}")
        print("QUESTION:", question)
        print("INTENT  :", result["intent"])
        print("CONTEXT :", result["context"])
        print("PROBLEMS:", result["problems"])
        print("ANSWER  :", result["final_answer"])
        print()