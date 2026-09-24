"""
Universal LLM client for DateSATBench generation.

This is the canonical location for the LLM client in this repo.
"""

import json
import os
import re
from typing import Optional

try:
    from dotenv import load_dotenv
    import pathlib

    # Load .env from the repository root (best effort)
    script_path = pathlib.Path(__file__).resolve()
    repo_root = script_path.parents[2]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv()
except ImportError:
    pass

ENABLE_OPENAI = False
ENABLE_ANTHROPIC = True

PROVIDER_CONFIG = {
    "openai": {
        "enabled": ENABLE_OPENAI,
        "default_model": "gpt-5.1",
        "api_key_env": "OPENAI_API_KEY",
        "thinking_param": "reasoning_effort",
        "thinking_value": "high",
    },
    "anthropic": {
        "enabled": ENABLE_ANTHROPIC,
        "default_model": "claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "thinking_param": "thinking",
        "thinking_value": {"type": "enabled", "budget_tokens": 10000},
    },
}


def _strip_code_fences(s: str) -> str:
    if "```" not in s:
        return s.strip()
    m = re.search(r"```json\s*(.*?)```", s, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", s, flags=re.S)
    return m.group(1).strip() if m else s.strip()


def _normalize_llm_json(s: str) -> str:
    if not isinstance(s, str):
        return s
    txt = s.strip().lstrip("\ufeff")
    txt = txt.replace("```", "")
    smart_double = "\u201c\u201d\uFF02"
    smart_single = "\u2018\u2019\uFF07"
    for ch in smart_double:
        txt = txt.replace(ch, '"')
    for ch in smart_single:
        txt = txt.replace(ch, "'")
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    return txt


def _extract_json_array(s: str) -> str:
    start = s.find("[")
    end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    raise ValueError("No JSON array found in response.")


def _detect_provider_and_model() -> tuple[str, str]:
    available_providers = {}
    for provider_name, config in PROVIDER_CONFIG.items():
        if config["enabled"]:
            api_key = os.getenv(config["api_key_env"])
            if api_key:
                available_providers[provider_name] = config

    enabled_count = sum(1 for config in PROVIDER_CONFIG.values() if config["enabled"])
    if enabled_count == 0:
        raise ValueError(
            "No LLM providers enabled. Set ENABLE_OPENAI or ENABLE_ANTHROPIC to True in datesatbench_new/utils/llm.py."
        )

    if "anthropic" in available_providers:
        return "anthropic", available_providers["anthropic"]["default_model"]
    if "openai" in available_providers:
        return "openai", available_providers["openai"]["default_model"]

    enabled_keys = [
        config["api_key_env"] for config in PROVIDER_CONFIG.values() if config["enabled"]
    ]
    raise ValueError(
        f"No API key found for enabled providers. Please set: {', '.join(enabled_keys)}"
    )


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "auto",
        enable_thinking: bool = True,
    ):
        self.api_key = api_key
        self.provider = provider.lower()
        self.enable_thinking = enable_thinking

        if self.provider == "auto":
            self.provider, default_model = _detect_provider_and_model()
            self.model = model or default_model
        else:
            if self.provider not in PROVIDER_CONFIG:
                raise ValueError(
                    f"Unsupported provider: {provider}. Use 'openai', 'anthropic', or 'auto'."
                )

            config = PROVIDER_CONFIG[self.provider]
            if not config["enabled"]:
                raise ValueError(
                    f"{self.provider.capitalize()} provider is disabled. "
                    f"Set ENABLE_{self.provider.upper()} = True in datesatbench_new/utils/llm.py to enable it."
                )

            self.api_key = api_key or os.getenv(config["api_key_env"])
            if not self.api_key:
                raise ValueError(
                    f"{self.provider.capitalize()} API key required. "
                    f"Set {config['api_key_env']} or pass api_key."
                )

            self.model = model or config["default_model"]

        if self.provider == "openai":
            import openai

            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)

        self.max_tokens = 16000
        self.temperature = 0.4
        self.top_p = 0.95

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, temp, tokens)
        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, temp, tokens)
        raise RuntimeError(f"Unknown provider: {self.provider}")

    def _call_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking:
            config = PROVIDER_CONFIG["openai"]
            api_params[config["thinking_param"]] = config["thinking_value"]
        resp = self.client.chat.completions.create(**api_params)
        return resp.choices[0].message.content

    def _call_anthropic(
        self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int
    ) -> str:
        if self.enable_thinking:
            temperature = 1.0
        api_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": f"{system_prompt}\n",
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.enable_thinking:
            config = PROVIDER_CONFIG["anthropic"]
            api_params[config["thinking_param"]] = config["thinking_value"]

        resp = self.client.messages.create(**api_params)

        if self.enable_thinking:
            text_parts = []
            for block in resp.content:
                if block.type == "text":
                    text_parts.append(block.text)
            return "\n".join(text_parts)
        return resp.content[0].text

    def parse_json_response(self, response: str, extract_array: bool = False):
        txt = _strip_code_fences(response)
        normalized = _normalize_llm_json(txt)

        if extract_array:
            try:
                return json.loads(normalized)
            except json.JSONDecodeError:
                extracted = _extract_json_array(normalized)
                return json.loads(_normalize_llm_json(extracted))

        return json.loads(normalized)
