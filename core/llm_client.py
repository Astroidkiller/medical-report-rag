"""
Unified LLM client for Community Health Intelligence Assistant.

Abstracts the LLM provider so agents don't couple to a specific SDK.
Supports:
  - Google Gemini API (simple API-key deploy)
  - Vertex AI Gemini (GCP)
  - Groq / Llama (fallback, local dev)

Usage:
    from core.llm_client import generate, generate_stream

    answer = generate("Explain hemoglobin levels", system_prompt="You are a medical AI.")
    for chunk in generate_stream("Explain hemoglobin levels"):
        print(chunk, end="")
"""

import os
import sys

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
_LLM_TIMEOUT = 30

# ---------- Provider Clients (lazy singletons) ----------

_gemini_client = None
_vertex_model = None
_groq_client = None


def _get_gemini_client():
    """Lazy-init Google Gemini API client."""
    global _gemini_client
    if _gemini_client is None:
        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

        from google import genai
        from google.genai import types

        try:
            _gemini_client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=_LLM_TIMEOUT * 1000),
            )
        except TypeError:
            _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_vertex_model():
    """Lazy-init Vertex AI Gemini model."""
    global _vertex_model
    if _vertex_model is None:
        if not GCP_PROJECT_ID:
            raise ValueError("GCP_PROJECT_ID is required when LLM_PROVIDER=vertex_ai")

        from google import genai
        from google.genai import types

        try:
            client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT_ID,
                location=GCP_LOCATION,
                http_options=types.HttpOptions(timeout=_LLM_TIMEOUT * 1000),
            )
        except TypeError:
            client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT_ID,
                location=GCP_LOCATION,
            )
        _vertex_model = client
    return _vertex_model


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

    if provider == "gemini":
        return _generate_gemini(prompt, system_prompt, model)
    elif provider == "vertex_ai":
        return _generate_vertex(prompt, system_prompt, model)
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

    if provider == "gemini":
        return _stream_gemini(prompt, system_prompt, model)
    elif provider == "vertex_ai":
        return _stream_vertex(prompt, system_prompt, model)
    elif provider == "groq":
        return _stream_groq(prompt, system_prompt, model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ---------- Google Gemini Implementation ----------


def _generate_gemini(prompt: str, system_prompt: str, model: str) -> str:
    """Generate with Google Gemini API using google-genai SDK."""
    client = _get_gemini_client()
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        temperature=0.3,
        max_output_tokens=2048,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or ""


def _stream_gemini(prompt: str, system_prompt: str, model: str):
    """Stream with Google Gemini API."""
    client = _get_gemini_client()
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        temperature=0.3,
        max_output_tokens=2048,
    )

    response = client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=config,
    )

    def _chunks():
        for chunk in response:
            if chunk.text:
                yield chunk.text

    return _chunks()


# ---------- Vertex AI Gemini Implementation ----------


def _generate_vertex(prompt: str, system_prompt: str, model: str) -> str:
    """Generate with Vertex AI Gemini using google-genai SDK."""
    client = _get_vertex_model()
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        temperature=0.3,
        max_output_tokens=2048,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or ""


def _stream_vertex(prompt: str, system_prompt: str, model: str):
    """Stream with Vertex AI Gemini."""
    client = _get_vertex_model()
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system_prompt if system_prompt else None,
        temperature=0.3,
        max_output_tokens=2048,
    )

    response = client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=config,
    )

    def _chunks():
        for chunk in response:
            if chunk.text:
                yield chunk.text

    return _chunks()


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
