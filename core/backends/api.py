"""Closed-API backend (OpenAI, DeepSeek, Gemini).

Generation only — no logits, no hooks. Its job in this repo is the CoT
measure: every provider surfaces reasoning traces differently, and
`_extract_reasoning` normalizes them into `GenerationResult.reasoning` so the
downstream reasoning-length analyses treat open and closed models alike.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from core.backends.base import GenerationResult, LMBackend, ScoredTokens

SUPPORTED_PROVIDERS = ("openai", "deepseek", "gemini")


class APIBackend(LMBackend):

    supports_logits = False
    supports_interp = False

    def __init__(
        self,
        name: str,
        cfg: Dict[str, Any],
        api_key: str,
        base_url: Optional[str] = None,
    ):
        super().__init__(name, cfg)
        self.provider = cfg["type"].lower().strip()
        self.model_name = cfg["path"]
        self.base_url = base_url

        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"provider must be one of {SUPPORTED_PROVIDERS}")

        if self.provider in ("openai", "deepseek"):
            from openai import OpenAI

            default_url = "https://api.deepseek.com" if self.provider == "deepseek" else None
            url = base_url or default_url
            self.client = OpenAI(api_key=api_key, base_url=url) if url else OpenAI(api_key=api_key)
        else:
            from google import genai

            self.client = genai.Client(api_key=api_key)

    # ---- CoT extraction ----

    @staticmethod
    def _openai_reasoning(resp) -> Optional[str]:
        """Concatenate reasoning summary parts, when the model emits them."""
        parts: List[str] = []
        for item in getattr(resp, "output", None) or []:
            if getattr(item, "type", None) == "reasoning":
                for summary in getattr(item, "summary", None) or []:
                    text = getattr(summary, "text", None)
                    if text:
                        parts.append(text)
        return "\n".join(parts) or None

    @staticmethod
    def _gemini_reasoning(resp) -> Optional[str]:
        """Gemini marks thought parts with `part.thought is True`."""
        parts: List[str] = []
        for cand in getattr(resp, "candidates", None) or []:
            for part in getattr(cand.content, "parts", None) or []:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    parts.append(part.text)
        return "\n".join(parts) or None

    # ---- generation ----

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        if self.provider == "openai":
            resp = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                max_output_tokens=int(max_new_tokens),
            )
            return GenerationResult(
                text=resp.output_text or "",
                reasoning=self._openai_reasoning(resp),
                raw=resp,
                usage=getattr(resp, "usage", None),
                provider="openai",
                model=self.model_name,
                response_id=getattr(resp, "id", None),
            )

        if self.provider == "deepseek":
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=int(max_new_tokens),
            )
            msg = resp.choices[0].message
            return GenerationResult(
                text=msg.content or "",
                reasoning=getattr(msg, "reasoning_content", None),
                raw=resp,
                usage=getattr(resp, "usage", None),
                provider="deepseek",
                model=self.model_name,
                response_id=getattr(resp, "id", None),
            )

        from google.genai import types

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=int(max_new_tokens),
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
        return GenerationResult(
            text=resp.text or "",
            reasoning=self._gemini_reasoning(resp),
            raw=resp,
            usage=getattr(resp, "usage_metadata", None),
            provider="gemini",
            model=self.model_name,
            response_id=getattr(resp, "id", None),
        )

    def score_options(self, prompt: str, option_a: str, option_b: str) -> ScoredTokens:
        raise NotImplementedError(
            f"{self.name} is an API model; option logits are not available. "
            "Use correctness from the generated text instead."
        )
