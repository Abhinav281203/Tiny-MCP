import logging
import traceback
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, ListToolsResult
from ollama import ChatResponse

from agent.ollama_client import OllamaClient
from mcp_client.utils import ollama_tool_parser

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MCPClient:
    def __init__(self, model, server_url="http://0.0.0.0:8000/sse"):
        self.model = model
        self.server_url = server_url
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OllamaClient(model=model)
        self.tools = []
        self.messages = []
        self.headers = {}
        self.logger = logging.getLogger(__name__)

    # connect to the MCP server via SSE
    async def connect_to_server(self):
        try:
            params = {
                "url": self.server_url,
                "headers": self.headers or {},
                "timeout": 5.0,
                "sse_read_timeout": 300.0,
            }

            client_stream = await self.exit_stack.enter_async_context(
                sse_client(**params)
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(*client_stream)
            )

            await self.session.initialize()
            self.logger.info("Connected to MCP server via SSE")

            mcp_tools = await self.get_mcp_tools()
            self.tools = ollama_tool_parser(mcp_tools)

            self.logger.info(
                f"Available tools: {[tool.function.name for tool in self.tools]}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    # get MCP tool list
    async def get_mcp_tools(self):
        try:
            response: ListToolsResult = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    # process query
    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query}")
            self.messages.append({"role": "user", "content": query})

            while True:
                response = await self.call_llm()

                # simple text response
                if response.message.content and not response.message.tool_calls:
                    self.logger.info(f"LLM gave text response. Stopping!")
                    self.messages.append(
                        {"role": "assistant", "content": response.message.content}
                    )
                    break

                # tool call response
                assistant_message = {
                    "role": "assistant",
                    "content": response.message.content or "",
                }
                if response.message.tool_calls:
                    self.logger.info(f"LLM gave Tool call!")

                    assistant_message["tool_calls"] = [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for i, tool_call in enumerate(response.message.tool_calls)
                    ]

                    self.messages.append(assistant_message)

                    for i, tool_call in enumerate(response.message.tool_calls):
                        try:
                            self.logger.info(f"Calling tool {tool_call.function.name}")
                            result: CallToolResult = await self.session.call_tool(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            )
                            self.logger.info(
                                f"Tool {tool_call.function.name} - result: {result}..."
                            )

                            # Add tool result to messages
                            tool_content = result.structuredContent["result"]
                            # Ensure content is a string
                            if not isinstance(tool_content, str):
                                tool_content = str(tool_content)

                            self.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": f"call_{i}",
                                    "content": tool_content,
                                }
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Error calling tool {tool_call.function.name}: {e}"
                            )
                            raise

            return self.messages

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    # call LLM
    async def call_llm(self) -> ChatResponse:
        try:
            self.logger.info("Calling LLM")
            response: ChatResponse = self.llm.chat(
                messages=self.messages, tools=self.tools
            )
            return response
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    # cleanup
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise


async def main():
    c = MCPClient(model="llama3.2:latest")
    await c.connect_to_server()
    await c.process_query("Add 23 n 2345543")

    for m in c.messages:

        for k, v in m.items():
            print(f"{k} - {v}")

        print()

    await c.cleanup()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
