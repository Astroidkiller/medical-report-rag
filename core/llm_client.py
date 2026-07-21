"""
Unified LLM client for Community Health Intelligence Assistant.

Uses direct REST API calls to Google Gemini (no SDK dependency issues).
Supports:
  - Google Gemini API (simple API-key deploy) via REST
  - Groq / Llama (fallback, local dev)

Usage:
    from core.llm_client import generate, generate_stream

    answer = generate("Explain hemoglobin levels", system_prompt="You are a medical AI.")
    for chunk in generate_stream("Explain hemoglobin levels"):
        print(chunk, end="")
"""

import os
import sys
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    GEMINI_API_KEY,
    GCP_PROJECT_ID,
    GCP_LOCATION,
    GROQ_API_KEY,
)

# Timeout for LLM calls (seconds)
_LLM_TIMEOUT = 60

# ---------- Provider Clients (lazy singletons) ----------

_groq_client = None


def _get_groq_client():
    """Lazy-init Groq client (fallback for local dev)."""
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        _groq_client = Groq(api_key=api_key, timeout=_LLM_TIMEOUT)
    return _groq_client


# ---------- Public API ----------


def generate(
    prompt: str,
    system_prompt: str = None,
    model: str = None,
    provider: str = None,
) -> str:
    """
    Generate a response from the LLM.

    Args:
        prompt: The user message / prompt.
        system_prompt: Optional system instruction.
        model: Override the default model name.
        provider: Override the default provider ("gemini", "vertex_ai", or "groq").

    Returns:
        The generated text response.
    """
    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL

    if provider in ("gemini", "vertex_ai"):
        return _generate_gemini_rest(prompt, system_prompt, model)
    elif provider == "groq":
        return _generate_groq(prompt, system_prompt, model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def generate_stream(
    prompt: str,
    system_prompt: str = None,
    model: str = None,
    provider: str = None,
):
    """
    Stream a response from the LLM.

    Args:
        prompt: The user message / prompt.
        system_prompt: Optional system instruction.
        model: Override the default model name.
        provider: Override the default provider.

    Yields:
        Text chunks as they arrive.
    """
    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL

    if provider in ("gemini", "vertex_ai"):
        return _stream_gemini_rest(prompt, system_prompt, model)
    elif provider == "groq":
        return _stream_groq(prompt, system_prompt, model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ---------- Google Gemini REST Implementation ----------

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _mask_api_key(text: str) -> str:
    """Mask the API key in any error messages to prevent exposing it in the UI."""
    import re
    # Matches key= up to next parameter (&), space, quote, or end of line/string
    return re.sub(r"key=[^&\s'\"]+", "key=REDACTED", text)


def _generate_gemini_rest(prompt: str, system_prompt: str, model: str) -> str:
    """Generate with Google Gemini API using direct REST calls (no SDK needed)."""
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 4096,
        },
    }

    if system_prompt:
        body["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    try:
        resp = requests.post(url, json=body, timeout=_LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        # Extract text from response
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return ""

    except requests.exceptions.HTTPError as e:
        error_body = ""
        try:
            error_body = e.response.json().get("error", {}).get("message", str(e))
        except Exception:
            error_body = str(e)
        raise RuntimeError(_mask_api_key(f"Gemini API error: {error_body}")) from e
    except requests.exceptions.RequestException as e:
        raise RuntimeError(_mask_api_key(f"Gemini API request failed: {e}")) from e



def _stream_gemini_rest(prompt: str, system_prompt: str, model: str):
    """Stream with Google Gemini API using direct REST calls."""
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

    url = f"{GEMINI_API_BASE}/{model}:streamGenerateContent?key={api_key}&alt=sse"

    body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 4096,
        },
    }

    if system_prompt:
        body["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    def _chunks():
        try:
            resp = requests.post(url, json=body, timeout=_LLM_TIMEOUT, stream=True)
            resp.raise_for_status()
            resp.encoding = 'utf-8'

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                json_str = line[6:]  # Remove "data: " prefix
                if json_str.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(json_str)
                    candidates = chunk_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            text = parts[0].get("text", "")
                            if text:
                                yield text
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            yield f"\n\n[Streaming error: {_mask_api_key(str(e))}]"

    return _chunks()


# ---------- Gemini Embeddings via REST ----------


def embed_texts_gemini_rest(texts: list[str], model: str = "gemini-embedding-2") -> list[list[float]]:
    """Generate embeddings using Gemini REST API (batch endpoint for high performance)."""
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required for Gemini embeddings")

    if not texts:
        return []

    url = f"{GEMINI_API_BASE}/{model}:batchEmbedContents?key={api_key}"

    # Build batch requests
    requests_body = []
    for text in texts:
        requests_body.append({
            "model": f"models/{model}",
            "content": {
                "parts": [{"text": text}]
            }
        })

    body = {"requests": requests_body}

    try:
        resp = requests.post(url, json=body, timeout=_LLM_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        embeddings_data = data.get("embeddings", [])
        
        all_embeddings = []
        for emb in embeddings_data:
            values = emb.get("values", [])
            all_embeddings.append(values)
            
        # Fallback in case response is sparse
        if len(all_embeddings) < len(texts):
            diff = len(texts) - len(all_embeddings)
            all_embeddings.extend([[0.0] * 768] * diff)
            
        return all_embeddings
    except Exception as e:
        print(f"Batch embedding error ({type(e).__name__})")
        # Return zero vectors as fallback
        return [[0.0] * 768] * len(texts)


# ---------- Groq / Llama Fallback ----------


def _generate_groq(prompt: str, system_prompt: str, model: str) -> str:
    """Generate with Groq (local dev fallback)."""
    client = _get_groq_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def _stream_groq(prompt: str, system_prompt: str, model: str):
    """Stream with Groq (local dev fallback)."""
    client = _get_groq_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    )
    def _chunks():
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return _chunks()
