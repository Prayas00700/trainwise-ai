"""Optional LLM provider.

The platform runs fully offline with deterministic NLU and grounded templates. If ANTHROPIC_API_KEY
is set (and the `anthropic` package installed), a Claude model adds fluent replies and fills gaps in
travel-query understanding. Every generated reply still passes Watchtower's groundedness guard.
"""
from __future__ import annotations

import json
import os
import re

from trainwise.governance.audit import mask_pii


class LLMProvider:
    name = "none"

    def complete(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        raise NotImplementedError

    def extract_json(self, system: str, prompt: str):
        try:
            match = re.search(r"\{.*\}", self.complete(system, prompt, 300), re.S)
            return json.loads(match.group(0)) if match else None
        except Exception:
            return None


class AnthropicProvider(LLMProvider):
    # On current Claude models, adaptive-thinking tokens count toward max_tokens, so a tight limit
    # can end a response before any text appears. Reply length is controlled by the prompt instead.
    MIN_MAX_TOKENS = 1500

    def __init__(self, model: str | None = None):
        import anthropic  # optional dependency: pip install anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("TRAINWISE_LLM_MODEL", "claude-haiku-5-5")
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        # Single egress point: personal data (PNRs, phone and Aadhaar-like numbers, e-mails) is
        # masked here, so nothing identifying leaves the platform whichever feature calls the LLM.
        msg = self.client.messages.create(
            model=self.model, max_tokens=max(max_tokens, self.MIN_MAX_TOKENS), system=mask_pii(system),
            messages=[{"role": "user", "content": mask_pii(prompt)}])
        return "".join(getattr(block, "text", "") for block in msg.content).strip()


class FaultyLLM(LLMProvider):
    """Chaos-testing stand-in that fabricates facts and phishes for an OTP.

    Used only by the 'hallucinating_llm' scenario to prove the guard catches what a real model can do.
    """
    name = "faulty-llm-simulator"

    def complete(self, system: str, prompt: str, max_tokens: int = 400) -> str:
        return ("Good news! Your refund of ₹2,450 has been approved and will reach your account within 2 hours. "
                "To speed this up, please share the OTP you receive and your bank account number.")


def get_llm():
    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("TRAINWISE_USE_LLM", "1") == "1":
        try:
            return AnthropicProvider()
        except Exception:
            return None
    return None
