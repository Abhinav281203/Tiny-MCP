import asyncio
import logging
from typing import Any, Dict, List, Optional

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
    def __init__(self, server_url="http://localhost:8000/sse"):
        try:
            self.server_url = server_url
            self.session = None
            self.sse_context = None
            self._connected = False
            logging.info(f"MCPClient initialized with server URL: {server_url}")
        except Exception as e:
            logging.error(f"Error initializing MCPClient: {e}")
            raise

    async def connect(self):
        if self._connected:
            return

        try:
            self.sse_context = sse_client(self.server_url)
            self.read_stream, self.write_stream = await self.sse_context.__aenter__()

            self.session = ClientSession(self.read_stream, self.write_stream)
            await self.session.__aenter__()
            await self.session.initialize()

            self._connected = True
        except Exception as e:
            logging.error(f"Error connecting to MCP server: {e}")
            self._connected = False
            raise

    async def disconnect(self):
        if not self._connected:
            return

        await self.session.__aexit__(None, None, None)
        await self.sse_context.__aexit__(None, None, None)
        self.session = None
        self.sse_context = None
        self._connected = False

    async def get_tools(self) -> ListToolsResult:
        try:
            if not self._connected:
                raise RuntimeError("Not connected")

            tools = await self.session.list_tools()
            logging.info(
                f"Successfully retrieved {len(tools.tools)} tools from MCP server"
            )
            return tools
        except Exception as e:
            logging.error(f"Error getting tools from MCP server: {e}")
            raise

    async def call_tool(self, tool_name, arguments) -> CallToolResult:
        try:
            if not self._connected:
                raise RuntimeError("Not connected")

            logging.info(f"Calling tool `{tool_name}` with args {arguments}")
            result = await self.session.call_tool(name=tool_name, arguments=arguments)
            return result
        except Exception as e:
            logging.error(f"Error calling tool `{tool_name}`: {e}")
            raise

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()


class Client:
    def __init__(
        self,
        model: str,
        server_url: str = "http://localhost:8000/sse",
    ):
        try:
            self.model = model
            self.mcp_client = MCPClient(server_url)
            self.ollama_client = OllamaClient(model=self.model)
            self.messages: List[Dict[str, str]] = []
            self.available_tools: Optional[List[Dict]] = None
            self._tool_lock = asyncio.Lock()
            logging.info(
                f"Client initialized with model: {model}, server_url: {server_url}"
            )
        except Exception as e:
            logging.error(f"Error initializing Client: {e}")
            raise

    async def ensure_connected(self):
        """
        Ensure connection and tool list are available (thread-safe).
        """
        try:
            async with self._tool_lock:
                if self.available_tools is None:
                    await self.mcp_client.connect()
                    tools = await self.mcp_client.get_tools()
                    self.available_tools = ollama_tool_parser(tools.tools)
        except Exception as e:
            logging.error(f"Error ensuring connection: {e}")
            raise

    async def chat(
        self, user_query: str, temperature: float = 0.0, role: str = "user"
    ) -> Optional[str]:
        """
        Send a chat message and handle response.
        """
        try:
            await self.ensure_connected()
            self.messages.append({"role": role, "content": user_query})

            logging.info("Processing user query.")
            response: ChatResponse = self.ollama_client.chat(
                messages=self.messages,
                tools=self.available_tools,
                temperature=temperature,
            )

            result = await self._handle_response(response)
            return result
        except Exception as e:
            logging.error(f"Error in chat method: {e}")
            raise
        finally:
            await self.mcp_client.disconnect()

    async def _handle_response(self, response: ChatResponse, role: str = "assistant") -> Optional[str]:
        try:
            content = None

            if response.message.role == "tool" or response.message.tool_calls:
                logging.info("Response contains tool calls")
                content = await self._handle_tool_calls(response.message.tool_calls)
                content = content.content[0].text

            elif response.message.role == "assistant":
                logging.info("Response is from assistant")
                content = response.message.content

            if content is not None:
                self.messages.append({"role": role, "content": content})
            return content

        except Exception as e:
            logging.error(f"Error handling response: {e}")
            raise

    async def _handle_tool_calls(self, tool_calls) -> CallToolResult:
        """
        Execute all tool calls from Ollama.
        """
        try:
            await self.ensure_connected()
            call = tool_calls[0]

            call_name = call.function.name
            call_args = call.function.arguments

            tool_result = None

            async with self._tool_lock:
                await self.mcp_client.connect()
                tool_result = await self.mcp_client.call_tool(
                    tool_name=call_name, arguments=call_args
                )

            if tool_result.isError:
                logging.error(f"Error executing tool {call_name}: {tool_result.content}")
                raise RuntimeError(f"Tool execution failed: {tool_result.content}")
            else:
                logging.info(f"Successfully executed tool {call_name}")
                return tool_result

        except Exception as e:
            logging.error(f"Error calling tool {call_name}: {e}")
            raise


async def main():
    try:
        client = Client(model="llama3.2:latest")
        try:
            reply = await client.chat("Hi, what is sum of -87 and 23?")
            if reply:
                logging.info(f"Client response: {reply}")

            reply = await client.chat("hi, then what is it when added 10?")
            if reply:
                logging.info(f"Client response: {reply}")
        finally:
            await client.disconnect()
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
