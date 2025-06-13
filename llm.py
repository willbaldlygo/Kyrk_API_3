"""OpenAI ChatCompletion wrapper."""
import os
from langchain.chat_models import ChatOpenAI

__all__ = ["get_llm"]

_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")
    return ChatOpenAI(model=_MODEL, temperature=0.1, max_tokens=2048)
