"""Adaptador `AIClient` para la API de Xiaomi MiMo (compatible con OpenAI)."""

from __future__ import annotations

import base64

import httpx

from oraculo.domain.attachments import AttachmentPayload
from oraculo.domain.errors import AIError
from oraculo.infrastructure.openai_compat import OpenAICompatClient

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


class MimoClient(OpenAICompatClient):
    """Cliente HTTP async que implementa el puerto `AIClient` para MiMo.

    La API de MiMo no admite los campos `thinking` ni `reasoning_effort` de
    DeepSeek, por lo que el payload se envía sin extensiones de proveedor.
    Tampoco expone Files API: los binarios admitidos (imágenes) viajan en línea
    como `image_url` con data URL.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float,
        max_tokens: int,
        system_prompt: str,
        http_client: httpx.AsyncClient,
        base_url: str = MIMO_BASE_URL,
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

    def _auth_headers(self) -> dict[str, str]:
        """Documentación oficial: se aceptan `api-key` o `Authorization: Bearer`.

        Se envían las dos para cubrir ambos formatos del gateway.
        """
        return {
            "api-key": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
        }

    async def _binary_block(self, attachment: AttachmentPayload) -> dict[str, object]:
        """Representa un binario como `image_url` con data URL.

        Solo las imágenes se admiten en binario; el resto llega aquí únicamente
        si la clasificación falló y se responde con un error tipado.
        """
        if not attachment.mime_type.startswith("image/"):
            raise AIError("la API de MiMo no admite este archivo en binario")
        encoded = base64.b64encode(attachment.data or b"").decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{attachment.mime_type};base64,{encoded}"},
        }
