import asyncio
import json
import sys
from pathlib import Path

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"

# Read-only tools run straight away. Tools that change data need a human's approval.
READ_ONLY_TOOLS = {"lookup_customer", "check_line_status"}
APPROVAL_TOOLS = {"open_ticket"}
ALLOWED_TOOLS = READ_ONLY_TOOLS | APPROVAL_TOOLS
MAX_ROUNDS = 4  # the loop stops after this many rounds, so it can never run forever

TOOL_PROMPT = (
    "You are a telecom support assistant. Use the tools to look up facts about "
    "the customer's account. If the customer has not given a customer ID "
    "(for example C001), do not call any tool. Only call a tool when you have "
    "the information it needs. If the customer asks for a support ticket and "
    "gives a customer ID, call open_ticket with that ID and a one-line summary "
    "of the problem."
)


def to_ollama_tool(tool) -> dict:
    """Describe an MCP tool in the format Ollama expects."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def _call_one(session, name: str, arguments: dict) -> str:
    """Run one tool and always return text. A tool failure becomes context, not a crash."""
    try:
        result = await session.call_tool(name, arguments)
        return result.content[0].text
    except Exception as error:
        return json.dumps({"error": str(error)})


def _server_params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=[str(SERVER)])


async def _run(question: str, model: str):
    results = []
    pending = None  # a write action waiting for a human decision

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            tools = [to_ollama_tool(t) for t in listed.tools
                     if t.name in ALLOWED_TOOLS]

            messages = [
                {"role": "system", "content": TOOL_PROMPT},
                {"role": "user", "content": question},
            ]
            for _ in range(MAX_ROUNDS):
                reply = ollama.chat(model=model, messages=messages, tools=tools,
                                    options={"temperature": 0})
                message = reply.message
                if not message.tool_calls:
                    break  # the model has no more tools to ask for

                messages.append(message)
                for call in message.tool_calls:
                    name = call.function.name
                    arguments = dict(call.function.arguments)

                    if name in APPROVAL_TOOLS:
                        # Do NOT run it. Record the request and wait for a human.
                        pending = {"tool": name, "arguments": arguments}
                        output = json.dumps({"status": "waiting for human approval"})
                    elif name in READ_ONLY_TOOLS:
                        output = await _call_one(session, name, arguments)
                        results.append({"source": f"tool:{name}", "text": output})
                    else:
                        output = json.dumps({"error": f"Tool {name} is not allowed"})

                    messages.append({"role": "tool", "content": output,
                                     "tool_name": name})
                if pending:
                    break  # stop here; the graph will pause for approval
    return results, pending


def run_tools(question: str, model: str):
    """Let the model use the MCP tools.
    Returns (context items, pending write action or None)."""
    return asyncio.run(_run(question, model))


async def _run_one(name: str, arguments: dict) -> str:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await _call_one(session, name, arguments)


def run_approved_tool(name: str, arguments: dict) -> dict:
    """Run a write tool AFTER a human approved it. Only approval tools are accepted."""
    if name not in APPROVAL_TOOLS:
        return {"error": f"Tool {name} is not an approval tool"}
    return json.loads(asyncio.run(_run_one(name, arguments)))