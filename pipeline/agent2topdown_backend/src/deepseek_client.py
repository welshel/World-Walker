"""Minimal DeepSeek chat-completions client and tolerant JSON extraction."""

import json
import os
from typing import Optional

import requests

from .schema import SYSTEM_PROMPT


def extract_json_object(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("DeepSeek response did not contain a valid JSON object")


class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None, timeout: int = 90):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")

    def generate(self, user_prompt: str, repair_context: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if repair_context:
            messages.append({"role": "user", "content": user_prompt})
            messages.append({"role": "assistant", "content": repair_context.split("\n--- VALIDATION ERRORS ---\n", 1)[0]})
            error_text = repair_context.split("\n--- VALIDATION ERRORS ---\n", 1)[-1]
            messages.append({"role": "user", "content": "Your previous response failed local validation. Return a corrected complete JSON scene only. Validation errors:\n" + error_text})
        else:
            messages.append({"role": "user", "content": user_prompt})
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages, "temperature": 0.2, "response_format": {"type": "json_object"}},
            timeout=self.timeout,
        )
        if not response.ok:
            raise RuntimeError(f"DeepSeek API returned HTTP {response.status_code}: {response.text[:1000]}")
        try:
            body = response.json()
            return body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek API response had an unexpected shape") from exc
