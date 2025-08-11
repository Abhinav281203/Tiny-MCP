import asyncio
import logging

import streamlit as st

from mcp_client.agent_client import Client

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

models = {
    "Llama3.2 3B": "llama3.2:latest",
    "Llama3.1 8B": "llama3.1:8b",
    "Nemotron-mini 4B": "nemotron-mini:4b",
}


def clear_conversation() -> None:
    if "client" in st.session_state:
        del st.session_state["client"]


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        return asyncio.create_task(coro)
    else:
        return loop.run_until_complete(coro)


def default_page():
    st.set_page_config(
        page_title="MCP try",
        page_icon="💾",
        layout="wide",
    )

    with st.sidebar:
        st.header("Config")

        model_label = st.selectbox(
            label="Model",
            options=list(models.keys()),
            index=0,
        )
        if "model" not in st.session_state or st.session_state.model != model_label:
            st.session_state.model = models[model_label]

        temperature = st.slider(
            label="Temperature", min_value=0.0, max_value=1.0, value=0.0
        )
        if (
            "temperature" not in st.session_state
            or st.session_state.temperature != temperature
        ):
            st.session_state.temperature = temperature

        if (
            "client" not in st.session_state
            or st.session_state.model != st.session_state.client.model
        ):
            st.session_state.client = Client(model=st.session_state.model)

    st.button("", on_click=clear_conversation, icon=":material/restart_alt:")

    WELCOME = "Hi! This is an Tiny MCP server"

    if "client" not in st.session_state or not getattr(
        st.session_state.client, "messages", []
    ):
        st.header(WELCOME)
    else:
        for msg in st.session_state.client.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if prompt := st.chat_input("Ask me something!"):
        run_async(
            st.session_state.client.chat(
                user_query=prompt, temperature=st.session_state.temperature
            )
        )
        st.rerun()


if __name__ == "__main__":
    default_page()
