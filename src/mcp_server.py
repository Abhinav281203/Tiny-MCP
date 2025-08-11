from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="mcp-server",
    host="0.0.0.0",
    port=8000,
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    try:
        result = a + b
        return result
    except Exception as e:
        raise


@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract two numbers"""
    try:
        result = a - b
        return result
    except Exception as e:
        raise


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    try:
        result = a * b
        return result
    except Exception as e:
        raise


if __name__ == "__main__":
    try:
        transport = "sse"
        mcp.run(transport=transport)
    except Exception as e:
        raise
