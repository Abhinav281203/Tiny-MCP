import logging
import requests
from typing import Dict, List, Optional
import json

from schemas import OllamaTool
from ollama import ChatResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nemotron-mini:4b",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        logging.info(
            f"OllamaClient initialized successfully with server_url: {self.base_url}"
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[OllamaTool]] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> Dict:
        """
        Chat with Ollama using /api/chat.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "top_k": 40,
                "num_gpu": -1,
                "num_thread": -1,
            },
            "stream": False,
        }

        if tools:
            payload["tools"] = [tool.model_dump() for tool in tools]

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            logging.info("Successfully received response from Ollama")
            return ChatResponse.model_validate(data)
        except Exception as e:
            logging.error(f"Error communicating with Ollama: {e}")
            raise

    def list_models(self) -> List[str]:
        """
        List available models using /api/tags.
        """
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            model_list = [m["model"] for m in models]
            logging.info(f"Found {len(model_list)} available models")
            return model_list
        except Exception as e:
            logging.error(f"Error listing models: {e}")
            raise


# if __name__ == "__main__":
#     client = OllamaClient()
#     models = client.list_models()
#     for m in models:
#         print(f"* {m}")

#     query = "what will be the output if we replace 'e' with 'i' in beach?"
#     resp = client.chat(messages=[{"role": "user", "content": query}])
#     resp = ChatResponse.model_validate(resp)
#     print(resp)
