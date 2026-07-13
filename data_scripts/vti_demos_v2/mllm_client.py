"""Vision-capable MLLM client for demos_v2 stages.

Mirrors ``evaluation/classifiers/judges.py`` spec strings
(``anthropic[:model]`` / ``openai[:model]`` / ``mock``) and reuses the
base64 image encoding approach from ``data_scripts/generate_captions.py``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from PIL import Image

from src.paths import project_root

load_dotenv(project_root() / ".env")

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
}


@dataclass
class CallResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None


def parse_provider_spec(spec: str) -> tuple[str, str]:
    spec = (spec or "mock").strip()
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    if provider == "mock":
        return "mock", "mock"
    if not model:
        model = _DEFAULT_MODELS.get(provider, "")
    if provider not in ("anthropic", "openai", "mock"):
        raise ValueError(
            f"Unknown provider spec '{spec}'. Use mock, anthropic[:model], openai[:model]."
        )
    return provider, model


def _encode_image(image: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def extract_json_object(text: str) -> dict:
    """Parse a JSON object from a model reply (tolerates markdown fences)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model reply")
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            raise ValueError(f"no JSON object in reply: {text[:200]!r}")
        text = text[start : end + 1]
    return json.loads(text)


class MLLMClient:
    """One-image-per-call (or text-only) client with mandatory mock support."""

    def __init__(
        self,
        spec: str,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ):
        self.spec = spec
        self.provider, self.model = parse_provider_spec(spec)
        self.max_tokens = max_tokens
        # Newer Anthropic models (e.g. opus-4-8) reject ``temperature`` as
        # deprecated; omit by default. OpenAI still accepts 0.0 when set.
        self.temperature = temperature
        self.name = f"{self.provider}:{self.model}" if self.provider != "mock" else "mock"
        self._client = None
        if self.provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "Anthropic requested but ANTHROPIC_API_KEY is not set "
                    "(expected in repo-root .env or the environment)."
                )
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
        elif self.provider == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OpenAI requested but OPENAI_API_KEY is not set "
                    "(expected in repo-root .env or the environment)."
                )
            from openai import OpenAI
            self._client = OpenAI(api_key=key)

    def complete(
        self,
        *,
        system: str,
        user_text: str,
        image: Optional[Union[Image.Image, Path]] = None,
        mock_json: Optional[Dict[str, Any]] = None,
    ) -> CallResult:
        if self.provider == "mock":
            payload = mock_json if mock_json is not None else {"ok": True}
            return CallResult(
                text=json.dumps(payload),
                model="mock",
                input_tokens=0,
                output_tokens=0,
            )

        img: Optional[Image.Image] = None
        if image is not None:
            img = image if isinstance(image, Image.Image) else load_image(Path(image))

        if self.provider == "anthropic":
            return self._complete_anthropic(system, user_text, img)
        return self._complete_openai(system, user_text, img)

    def _complete_anthropic(
        self, system: str, user_text: str, image: Optional[Image.Image]
    ) -> CallResult:
        content: list = []
        if image is not None:
            b64, media_type = _encode_image(image)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64},
            })
        content.append({"type": "text", "text": user_text})
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self._client.messages.create(**kwargs)
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        usage = getattr(resp, "usage", None)
        return CallResult(
            text="".join(parts),
            model=self.model,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            raw=resp,
        )

    def _complete_openai(
        self, system: str, user_text: str, image: Optional[Image.Image]
    ) -> CallResult:
        user_content: list = [{"type": "text", "text": user_text}]
        if image is not None:
            b64, media_type = _encode_image(image)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            })
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        # OpenAI: default to 0 when unspecified for deterministic JSON.
        kwargs["temperature"] = 0.0 if self.temperature is None else self.temperature
        resp = self._client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = getattr(resp, "usage", None)
        return CallResult(
            text=text,
            model=self.model,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            raw=resp,
        )
