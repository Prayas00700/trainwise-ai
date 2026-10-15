"""Hallucination and safety guard for passenger-facing replies.

A reply is *grounded* when its facts can be traced to the evidence the assistant retrieved (tool
outputs and knowledge-base passages):
  * numeric support - every number in the reply (fares, dates, times, percentages, PNR digits)
    must appear in the evidence;
  * lexical support - share of content words in each sentence that also appear in the evidence.
A safety screen also blocks replies that solicit credentials (OTP, PIN, card or bank details).
"""
from __future__ import annotations

import json
import re

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[\w\u0900-\u097f]+")
_SENT = re.compile(r"(?<=[.!?\u0964])\s+")
_STOP = set("the and for are was you your this that with from has have not any can will our its into been "
            "their they them per all also only more such than then there here what when which who".split())
_SECRET = r"(?:otp|cvv|pin|password|card number|bank details|bank account|account number)"
_SOLICIT = re.compile(rf"(?:share|send|tell|provide|enter|give)\b.{{0,60}}\b{_SECRET}|{_SECRET}\b.{{0,40}}\b(?:share|send|tell|provide)",
                      re.I | re.S)


def _numbers(text: str) -> set:
    return {n.replace(",", "") for n in _NUM.findall(text)}


def _words(text: str) -> set:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 2 and not w.isdigit() and w.lower() not in _STOP}


class GroundednessChecker:
    def __init__(self, min_groundedness: float = 0.70):
        self.min_groundedness = min_groundedness

    def check(self, reply: str, evidence) -> dict:
        ev_text = evidence if isinstance(evidence, str) else json.dumps(evidence, ensure_ascii=False, default=str)
        ev_nums, ev_words = _numbers(ev_text), _words(ev_text)
        nums = _numbers(reply)
        unsupported = sorted(n for n in nums if n not in ev_nums)
        numeric = 1.0 if not nums else 1 - len(unsupported) / len(nums)
        sentences = [s for s in _SENT.split(reply) if _words(s)]
        lex = [len(_words(s) & ev_words) / len(_words(s)) for s in sentences] or [1.0]
        lexical = sum(lex) / len(lex)
        lexical_norm = min(1.0, lexical / 0.5)          # 50% content-word overlap counts as fully supported
        score = 0.7 * numeric + 0.3 * lexical_norm if nums else lexical_norm
        unsafe = bool(_SOLICIT.search(reply))
        return {"groundedness": round(score, 4), "numeric_support": round(numeric, 4),
                "lexical_support": round(lexical, 4), "unsupported_numbers": unsupported,
                "credential_solicitation": unsafe,
                "flagged": bool(score < self.min_groundedness or unsupported or unsafe)}
