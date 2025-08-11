# MCP Implementation

A Python implementation of the Model Context Protocol (MCP) that enables AI models to interact with external tools and services through a standardized interface. This project includes both a server that exposes tools and a client that can communicate with AI models (specifically Ollama) to execute those tools.

## Project Structure

```
MCP try/
├── requirements.txt          # Python dependencies
├── src/
│   ├── agent/
│   │   └── ollama_client.py  # Ollama AI model client
│   ├── mcp_client/
│   │   ├── agent_client.py   # Main MCP client with AI integration
│   │   ├── client_demo.py    # Demo script for MCP client
│   │   └── utils.py          # Utility functions
│   ├── mcp_server.py         # MCP server with tool definitions
│   ├── schemas.py            # Pydantic schemas for data validation
│   └── ui/
│       └── app.py            # Streamlit UI (placeholder)
```

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- At least one Ollama model downloaded (e.g., `llama3.1:8b`)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "MCP try"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and start Ollama**:
   ```bash
   # Follow instructions at https://ollama.ai
   ollama pull llama3.1:8b  # or any other model
   ```

## Usage

### 1. Start the MCP Server

The server exposes mathematical tools and runs on port 8000:

```bash
cd src
python mcp_server.py
```

The server will start and listen for connections on `http://localhost:8000/sse`.

### 2. Use the AI-Integrated Client

Run the main client that integrates with Ollama:

```bash
cd src/mcp_client
python agent_client.py
```

This will:
- Connect to the MCP server
- Initialize an Ollama client
- Process user queries through the AI model
- Execute tools when the AI decides they're needed


## Available Tools

The MCP server currently exposes these mathematical tools:

- `add(a: int, b: int) -> int`: Add two numbers
- `subtract(a: int, b: int) -> int`: Subtract two numbers  
- `multiply(a: int, b: int) -> int`: Multiply two numbers
