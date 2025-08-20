import asyncio
from typing import AsyncGenerator
import logging

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseConnectionParams
from google.genai import types

load_dotenv()


class GoogleADKClient:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        session_id: str = "blahblah",
        app_name: str = "blaapp",
        user_id: str = "blah",
    ) -> None:
        self.model_name = model_name
        self.mcp_toolset = MCPToolset(
            connection_params=SseConnectionParams(
                url="http://0.0.0.0:8000/sse",
            )
        )
        self.agent = LlmAgent(
            model=model_name,
            name="assistant",
            instruction="Use the available MCP tools to assist the user.",
            tools=[self.mcp_toolset],
        )
        self.run_config = RunConfig(
            response_modalities=["TEXT"], streaming_mode=StreamingMode.SSE
        )

        self.app_name = app_name
        self.session_id = session_id
        self.user_id = user_id
        self.session_service = InMemorySessionService()

    async def create_session(self):
        self.session = await self.session_service.create_session(
            app_name=self.app_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service,
        )

    async def chat(self, query: str) -> AsyncGenerator[Event, None]:
        content = types.Content(role="user", parts=[types.Part(text=query)])
        events = self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content,
            run_config=self.run_config,
        )

        async for e in events:
            yield e

    async def cleanup(self):
        await self.mcp_toolset.close()
        await self.runner.close()


async def main():
    c = GoogleADKClient()
    await c.create_session()

    while 1:
        print("*" * 100)
        q = input()

        if q == "exit()":
            break

        async for e in c.chat(q):
            if e.is_final_response():
                print(e.content.parts[0].text)
            
            if e.content.parts[0].function_call:
                print()
                print("Function call: ")
                print(e.content.parts[0].function_call.name)
                print(e.content.parts[0].function_call.args)

            if e.content.parts[0].function_response:
                print()
                print("Function response: ")
                print(e.content.parts[0].function_response.response["result"].structuredContent["result"])
                print()

    await c.cleanup()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
