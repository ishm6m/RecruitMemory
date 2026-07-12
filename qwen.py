"""
qwen.py, talks to Qwen Cloud using the official OpenAI SDK.

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
ASR_MODEL = os.getenv("QWEN_ASR_MODEL", "qwen3-asr-flash")

# Speech-to-text lives on the same endpoint for intl keys (verified); a second
# client is only built if QWEN_ASR_BASE_URL points ASR somewhere else.
_asr_client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"), base_url=os.getenv("QWEN_ASR_BASE_URL")
) if os.getenv("QWEN_ASR_BASE_URL") else _client


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


def transcribe(audio_b64_wav):
    """Turn a base64-encoded WAV clip (16 kHz mono 16-bit) into transcript text.

    Returns "" for silence. Payload shape verified against the live API:
    audio goes in as a data URI inside an `input_audio` content part.
    """
    resp = _asr_client.chat.completions.create(
        model=ASR_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "input_audio",
             "input_audio": {"data": "data:audio/wav;base64," + audio_b64_wav}}
        ]}],
        extra_body={"asr_options": {"enable_itn": True}},
    )
    return resp.choices[0].message.content or ""
