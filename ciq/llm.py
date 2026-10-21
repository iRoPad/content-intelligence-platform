"""Minimal, dependency-free client for the Anthropic Messages API."""
from __future__ import annotations

import json
import re
import urllib.request

from . import config

API_URL = "https://api.anthropic.com/v1/messages"


def available() -> bool:
    return bool(config.ANTHROPIC_API_KEY)


def complete(system: str, prompt: str, max_tokens: int = 2048, temperature: float = 0.0) -> str:
    body = json.dumps({
        "model": config.LLM_MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "x-api-key": config.ANTHROPIC_API_KEY or "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read())
    return "".join(block.get("text", "") for block in payload.get("content", []))


def extract_json(text: str):
    """Pull the first JSON object/array out of a model response."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = min([i for i in (text.find("["), text.find("{")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON found in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj
