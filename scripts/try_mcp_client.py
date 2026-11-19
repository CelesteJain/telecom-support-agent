import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])


async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            for tool in tools.tools:
                first_line = (tool.description or "").split("\n")[0]
                print("TOOL:", tool.name, "-", first_line)
            print()

            calls = [
                ("lookup_customer", {"customer_id": "C001"}),
                ("check_line_status", {"customer_id": "C002"}),
                ("lookup_customer", {"customer_id": "C999"}),
                ("open_ticket", {"customer_id": "C002",
                                 "summary": "Roaming blocked while abroad"}),
            ]
            for name, args in calls:
                result = await session.call_tool(name, args)
                print(name, args, "->", result.content[0].text)


asyncio.run(main())