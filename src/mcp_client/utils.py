import logging
from typing import List

from mcp.types import Tool

from schemas import (
    FunctionDefinition,
    FunctionParameter,
    FunctionParameters,
    OllamaTool,
)


def ollama_tool_parser(tools: List[Tool]) -> List[OllamaTool]:
    """
    Converts MCP tools to Ollama tool format
    """
    ollama_tools = []

    for mcp_tool in tools:
        # Convert input schema properties to Ollama format
        properties = {}
        for prop_name, prop_schema in mcp_tool.inputSchema["properties"].items():
            properties[prop_name] = FunctionParameter(
                type=prop_schema["type"],
                description=prop_schema.get("title", prop_name),
            )

        # Create FunctionParameters
        function_parameters = FunctionParameters(
            required=mcp_tool.inputSchema["required"], properties=properties
        )

        # Create FunctionDefinition
        function_definition = FunctionDefinition(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=function_parameters,
        )

        # Create OllamaTool
        ollama_tool = OllamaTool(function=function_definition)
        ollama_tools.append(ollama_tool)

    return ollama_tools
