import uuid

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.graph import build_graph

app = FastAPI(title="Telecom Support Agent")
graph = build_graph()  # built once, when the server starts


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ApprovalRequest(BaseModel):
    thread_id: str
    approve: bool


def to_response(thread_id: str, result: dict) -> dict:
    """Turn the graph's result into JSON: either a finished answer or a pause."""
    if result.get("__interrupt__"):
        return {
            "status": "needs_approval",
            "thread_id": thread_id,
            "request": result["__interrupt__"][0].value,
        }
    return {
        "status": "done",
        "thread_id": thread_id,
        "intent": result["intent"],
        "answer": result["final_answer"],
        "problems": result.get("problems", []),
    }


# These endpoints are plain `def`, not `async def`. The graph is slow, blocking code, and
# FastAPI runs plain `def` endpoints in a worker thread. (The MCP tool loop uses asyncio.run,
# which would fail inside an `async def` endpoint that already has a running event loop.)
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(body: ChatRequest):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"question": body.message}, config)
    return to_response(thread_id, result)


@app.post("/approve")
def approve(body: ApprovalRequest):
    config = {"configurable": {"thread_id": body.thread_id}}
    if not graph.get_state(config).next:  # nothing is paused on this thread
        raise HTTPException(status_code=404, detail="No pending approval for this thread_id")
    decision = "approve" if body.approve else "reject"
    result = graph.invoke(Command(resume=decision), config)
    return to_response(body.thread_id, result)