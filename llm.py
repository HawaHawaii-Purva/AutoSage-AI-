"""LangChain model adapter for AutoSage.

The app uses LangChain's standard chat-model interface for both local Ollama
and hosted Hugging Face inference, so prompts, chains and model providers share
one interface throughout the application.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()


def _to_langchain_messages(messages: list[dict[str, str]]) -> list[Any]:
    converted: list[Any] = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = str(message.get("content", ""))
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


@lru_cache(maxsize=8)
def _build_chat_model(provider: str, model: str, base_url: str, temperature: float):
    """Build and cache a LangChain chat model by provider/config."""
    provider = provider.lower().strip()
    if provider == "hf":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN/HUGGINGFACEHUB_API_TOKEN is missing. Add it to Streamlit secrets or switch to local Ollama."
            )
        # LangChain's HF integration reads HUGGINGFACEHUB_API_TOKEN from the environment.
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
        endpoint = HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "900")),
            do_sample=temperature > 0,
            temperature=temperature,
            provider=os.getenv("HF_PROVIDER", "auto"),
        )
        return ChatHuggingFace(llm=endpoint)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def get_chat_model(temperature: float = 0.2):
    """Return the configured LangChain chat model."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    if provider == "hf":
        model = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        base_url = ""
    else:
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return _build_chat_model(provider, model, base_url, float(temperature))


def chat(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    """Invoke the configured LangChain chat model and return plain text."""
    response = get_chat_model(temperature).invoke(_to_langchain_messages(messages))
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")
