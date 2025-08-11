import logging
import os
from typing import Any, Dict, List

from pydantic import BaseModel

#  Ollama needs this format
# subtract_two_numbers_tool = {
#   'type': 'function',
#   'function': {
#     'name': 'subtract_two_numbers',
#     'description': 'Subtract two numbers',
#     'parameters': {
#       'type': 'object',
#       'required': ['a', 'b'],
#       'properties': {
#         'a': {'type': 'integer', 'description': 'The first number'},
#         'b': {'type': 'integer', 'description': 'The second number'},
#       },
#     },
#   },
# }


class FunctionParameter(BaseModel):
    type: str
    description: str


class FunctionParameters(BaseModel):
    type: str = "object"
    required: List[str]
    properties: Dict[str, FunctionParameter]


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


class OllamaTool(BaseModel):
    type: str = "function"
    function: FunctionDefinition
