"""Adaptador `AIClient` para la API de DeepSeek (compatible con OpenAI)."""

from __future__ import annotations

import httpx

from oraculo.infrastructure.openai_compat import OpenAICompatClient

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekClient(OpenAICompatClient):
    """Cliente HTTP async que implementa el puerto `AIClient` para DeepSeek.

    Añade al payload los campos propios de DeepSeek (`thinking` y
    `reasoning_effort`); el resto del comportamiento vive en `OpenAICompatClient`.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float,
        max_tokens: int,
        system_prompt: str,
        thinking: bool,
        reasoning_effort: str,
        http_client: httpx.AsyncClient,
        base_url: str = DEEPSEEK_BASE_URL,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            http_client=http_client,
            base_url=base_url,
        )
        self._thinking = thinking
        self._reasoning_effort = reasoning_effort

    def _extra_payload(self) -> dict[str, object]:
        extra: dict[str, object] = {
            "thinking": {"type": "enabled" if self._thinking else "disabled"},
        }
        if self._thinking:
            extra["reasoning_effort"] = self._reasoning_effort
        return extra
