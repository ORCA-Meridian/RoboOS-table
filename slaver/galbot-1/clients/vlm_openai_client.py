import logging
from typing import Any

from openai import OpenAI


class OpenAICompatibleVLMClient:
    """Calls OpenAI-compatible vision APIs, including Qwen via DashScope."""

    def __init__(self, cfg: dict, log: logging.Logger):
        self.provider = cfg.get("provider", "openai_compatible")
        if self.provider not in {"openai_compatible", "openai"}:
            raise ValueError(
                "Unsupported vlm.provider={!r}; use 'openai_compatible' or 'openai'.".format(
                    self.provider
                )
            )
        self.model = cfg["model"]
        self.log = log
        timeout = float(cfg.get("timeout", 120))
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["api_base"], timeout=timeout)
        self.extra_body = self._build_extra_body(cfg)

    def _build_extra_body(self, cfg: dict) -> dict[str, Any] | None:
        if self.provider != "openai_compatible" or "enable_thinking" not in cfg:
            return None
        return {"enable_thinking": bool(cfg["enable_thinking"])}

    def judge_completion(self, content: list) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=16,
            temperature=0.0,
            extra_body=self.extra_body,
        )
        return (response.choices[0].message.content or "").strip().lower()

    def decide(self, system_prompt: str, content: list, tools: list) -> tuple[str | None, str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            tools=tools,
            tool_choice="required",
            max_tokens=64,
            temperature=0.0,
            extra_body=self.extra_body,
        )
        message = response.choices[0].message
        if not message.tool_calls:
            return None, message.content or ""

        tool_call = message.tool_calls[0]
        arguments = tool_call.function.arguments or "{}"
        self.log.info(
            "[vlm-driven] tool_call: id=%s name=%s arguments=%s",
            getattr(tool_call, "id", ""),
            tool_call.function.name,
            arguments,
        )
        return tool_call.function.name, arguments
