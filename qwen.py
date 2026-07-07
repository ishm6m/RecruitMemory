"""
qwen.py — talks to Qwen Cloud using the official OpenAI SDK.

Qwen Cloud exposes an "OpenAI-compatible" API, which means we can use the
familiar `openai` Python library and just point it at Qwen's base URL. This
file has exactly two jobs: `chat()` (get a reply) and `embed()` (turn text into
a vector of numbers we can compare with cosine similarity).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads the .env file so os.getenv() works

_client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v3")


def chat(messages, temperature=0.3):
    """Send a list of {role, content} messages, return the reply text."""
    resp = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def embed(text):
    """Turn a piece of text into an embedding vector (a list of floats)."""
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding
