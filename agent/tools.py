import asyncio
import json
import sys
from pathlib import Path

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"

# The model may only use tools on this list. open_ticket joins in 6e, with approval.
READ_ONLY_TOOLS = {"lookup_customer", "check_line_status"}
MAX_ROUNDS = 4  # the loop stops after this many rounds, so it can never run forever

TOOL_PROMPT = (
    "You are a telecom support assistant. Use the tools to look up facts about "
    "the customer's account. If the customer has not given a customer ID "
    "(for example C001), do not call any tool. Only call a tool when you have "
    "the information it needs."
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


async def _run(question: str, model: str) -> list:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    results = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            tools = [to_ollama_tool(t) for t in listed.tools
                     if t.name in READ_ONLY_TOOLS]

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

                    if name not in READ_ONLY_TOOLS:
                        output = json.dumps({"error": f"Tool {name} is not allowed"})
                    else:
                        try:
                            result = await session.call_tool(name, arguments)
                            output = result.content[0].text
                        except Exception as error:  # a tool failure becomes context, not a crash
                            output = json.dumps({"error": str(error)})

                    results.append({"source": f"tool:{name}", "text": output})
                    messages.append({"role": "tool", "content": output,
                                     "tool_name": name})
    return results


def run_tools(question: str, model: str) -> list:
    """Let the model use the read-only MCP tools. Returns context items."""
    return asyncio.run(_run(question, model))