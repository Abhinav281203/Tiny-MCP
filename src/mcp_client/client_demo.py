import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client

from mcp_client.utils import ollama_tool_parser

"""
Make sure:
1. The server is running before running this script.
2. The server is configured to use SSE transport.
3. The server is listening on port 8050.

To run the server:
uv run server.py
"""

import json


async def main():
    # Connect to the server using SSE
    async with sse_client("http://localhost:8000/sse") as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            ollama_tools = ollama_tool_parser(tools_result.tools)

            for tool in ollama_tools:
                print(json.dumps(tool.model_dump(), indent=4))


if __name__ == "__main__":
    asyncio.run(main())
