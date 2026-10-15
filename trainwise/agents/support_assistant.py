"""Module 6 - Multilingual (English / Hindi / Hinglish) customer-support assistant.

Pipeline: language detection -> intent classification (governed model) -> tool / knowledge-base
retrieval -> reply (LLM if configured, otherwise approved templates) -> groundedness + safety guard
-> monitoring and a PII-masked audit record of every turn.

Fee and policy answers are always served verbatim from the verified knowledge base, even when an LLM
is configured: a paraphrase can silently drop a condition (such as a minimum charge), and a grounding
check cannot catch an omission because nothing was invented.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import date
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from trainwise.agents.nlu import detect_language, extract_entities
from trainwise.governance.audit import mask_pii
from trainwise.governance.hallucination import GroundednessChecker

COPY = {
    "pnr_en": "PNR {pnr}: train {train}, journey date {date}, class {cls}, departs {dep}. Current status: {status}.",
    "pnr_hi": "पीएनआर {pnr}: ट्रेन {train}, यात्रा तिथि {date}, क्लास {cls}, प्रस्थान {dep}। वर्तमान स्थिति: {status}।",
    "pred_en": " Estimated chance of confirmation: {pct}% (chance of at least RAC: {travel}%). Likely status when the chart is prepared: {final}.",
    "pred_hi": " कन्फर्म होने की अनुमानित संभावना: {pct}% (कम से कम आरएसी मिलने की संभावना: {travel}%)। चार्ट बनने पर संभावित स्थिति: {final}।",
    "refund_en": "Refund for PNR {pnr}: status {status}, ₹{amount} initiated on {on} to your {mode}.",
    "refund_hi": "पीएनआर {pnr} का रिफंड: स्थिति {status}, ₹{amount} की राशि {on} को आपके {mode} में भेजी गई।",
    "norefund_en": "I could not find a refund for PNR {pnr}.",
    "norefund_hi": "पीएनआर {pnr} के लिए कोई रिफंड नहीं मिला।",
    "notfound_en": "I could not find PNR {pnr}. Please check the 10-digit number on your ticket.",
    "notfound_hi": "पीएनआर {pnr} नहीं मिला। कृपया टिकट पर लिखा 10 अंकों का नंबर जाँचें।",
    "askpnr_en": "Please share your 10-digit PNR number so I can check it.",
    "askpnr_hi": "कृपया अपना 10 अंकों का पीएनआर नंबर बताइए, मैं जाँच कर देता हूँ।",
    "greet_en": "Namaste! I can help with PNR status, waitlist confirmation chances, refunds, cancellation rules, train suggestions and journey planning. What do you need?",
    "greet_hi": "नमस्ते! मैं पीएनआर स्थिति, वेटलिस्ट कन्फर्म होने की संभावना, रिफंड, कैंसिलेशन नियम, ट्रेन सुझाव और यात्रा योजना में मदद कर सकता हूँ। बताइए, क्या चाहिए?",
    "complaint_en": "I'm sorry about this. I have logged complaint {ticket} and handed it to the on-board services team; a human agent will follow up with you.",
    "complaint_hi": "असुविधा के लिए खेद है। मैंने शिकायत {ticket} दर्ज कर ली है और ऑन-बोर्ड सेवा टीम को भेज दी है; एक मानव एजेंट आपसे संपर्क करेगा।",
    "clarify_en": "I want to help with the right thing. Is this about a PNR status, a refund, cancellation rules, or planning a journey?",
    "clarify_hi": "सही मदद के लिए बताइए: क्या यह पीएनआर स्थिति, रिफंड, कैंसिलेशन नियम या यात्रा योजना के बारे में है?",
    "plan_en": "Best option: {train} in {cls}, departs {dep}, arrives {arr}, fare ₹{fare}, seat status {avail} ({travel}% chance of travelling). Open the journey planner to compare alternatives and approve the booking.",
    "plan_hi": "सबसे अच्छा विकल्प: {train}, {cls}, प्रस्थान {dep}, आगमन {arr}, किराया ₹{fare}, सीट स्थिति {avail} (यात्रा की संभावना {travel}%)। विकल्पों की तुलना और बुकिंग की मंज़ूरी के लिए जर्नी प्लानर खोलें।",
    "source_en": "Source: {source} (verified {verified}).",
    "source_hi": "स्रोत: {source} (सत्यापित {verified})।",
}
_STATUS_HI = {"initiated": "प्रक्रिया में", "processed": "पूरा", "credited": "जमा"}
VERBATIM_INTENTS = ("cancellation_policy", "booking_help")   # exact verified text, never paraphrased


def mask_pnr(pnr: str) -> str:
    return mask_pii(pnr)


class KnowledgeBase:
    """Source-attributed, bilingual policy passages with character n-gram TF-IDF retrieval."""

    def __init__(self, path: Path):
        self.entries = json.loads(Path(path).read_text(encoding="utf-8"))
        self.index = {}
        for lang in ("en", "hi"):   # one index per language; titles and keywords weighted twice
            docs = [f"{e['title']} {e['title']} {e['keywords']} {e['keywords']} {e['text_' + lang]}".lower()
                    for e in self.entries]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(docs)
            self.index[lang] = (vec, vec.transform(docs))

    def search(self, query: str, lang: str = "en", top_k: int = 2) -> list:
        vec, matrix = self.index["hi" if lang == "hi" else "en"]
        sims = (matrix @ vec.transform([query.lower()]).T).toarray().ravel()
        order = sims.argsort()[::-1]
        best = sims[order[0]]
        return [{**self.entries[i], "score": round(float(sims[i]), 4)} for i in order[:top_k]
                if sims[i] > 0.05 and sims[i] >= 0.85 * best]

    def get(self, entry_id: str) -> dict:
        return next(e for e in self.entries if e["id"] == entry_id)


class SupportAssistant:
    def __init__(self, platform):
        self.p = platform
        self.checker = GroundednessChecker(platform.policies["genai"]["min_groundedness"])
        self.llm = platform.llm
        self.mode = "generative" if self.llm else "grounded_templates"

    @property
    def llm_name(self):
        return getattr(self.llm, "name", None) if self.mode == "generative" and self.llm else None

    def set_mode(self, mode: str, actor: str = "system") -> None:
        self.mode = mode
        self.p.audit.record(actor, "support_mode_changed", "support_assistant", {"mode": mode})

    # ------------------------------------------------------------------ main entry
    def chat(self, message: str, session_id: str = "default", user_id: str | None = None,
             synthetic: bool = False) -> dict:
        t0 = time.perf_counter()
        lang = detect_language(message)
        out_lang = "hi" if lang == "hi" else "en"
        cls = self.p.services["intent_classifier"].predict({"text": message})
        if cls["status"] != "ok":
            intent, conf = "unknown", 0.0
        else:
            intent, conf = cls["intent"], cls["confidence"]
        ent = extract_entities(message)
        handler = {"pnr_status": self._pnr, "refund_status": self._refund, "cancellation_policy": self._kb,
                   "booking_help": self._kb, "travel_recommendation": self._plan, "journey_planning": self._plan,
                   "greeting": self._greet, "complaint": self._complaint}.get(intent)
        if handler is None or conf < 0.35:
            draft, evidence, escalate = self._clarify(out_lang)
            verbatim = False
        else:
            draft, evidence, escalate = handler(message, ent, out_lang, user_id)
            verbatim = intent in VERBATIM_INTENTS
        reply, guard, used_llm, source = self._finalise(message, draft, evidence, out_lang, verbatim)
        turn = {"session_id": session_id, "lang": lang, "intent": intent, "confidence": conf,
                "groundedness": guard["groundedness"], "flagged": guard["flagged"],
                "unsupported": guard["unsupported_numbers"], "credential_solicitation": guard["credential_solicitation"],
                "escalated": escalate, "mode": self.mode, "llm": used_llm, "reply_source": source,
                "synthetic": synthetic, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        self.p.monitor.log("support_turns", turn)
        self.p.audit.record("support_assistant", "chat_turn", session_id,
                            {"message": message[:300], "reply": reply[:400],
                             **{k: turn[k] for k in ("lang", "intent", "confidence", "groundedness", "flagged",
                                                     "escalated", "reply_source")}})
        sources = [e.get("source") for e in evidence.get("kb", [])] if isinstance(evidence, dict) else []
        return {"reply": reply, "language": lang, "intent": intent, "confidence": conf, "guard": guard,
                "escalated": escalate, "sources": sources, "mode": self.mode, "reply_source": source}

    def feedback(self, session_id: str, rating: int) -> None:
        self.p.monitor.log("csat", {"session_id": session_id, "rating": int(rating)})
        self.p.audit.record("passenger", "csat_submitted", session_id, {"rating": int(rating)})

    def _finalise(self, message: str, draft: str, evidence: dict, lang: str, verbatim: bool = False):
        """Return (reply, guard result, whether the LLM was called, where the reply came from)."""
        if self.mode == "generative" and self.llm is not None and not verbatim:
            system = ("You are TrainWise, an Indian Railways passenger-support assistant. Answer ONLY from the EVIDENCE. "
                      "Never invent numbers, dates, train names, refund amounts or timelines, and keep every "
                      "condition, limit and exception the evidence states. Never ask for OTPs, passwords, card or "
                      "bank details. If the evidence does not answer the question, say a human agent will follow "
                      "up. Write plain text only: no Markdown, bullet symbols or asterisks. "
                      f"Reply in {'Hindi' if lang == 'hi' else 'English'} in under 90 words.")
            prompt = f"EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False, default=str)}\n\nPASSENGER: {message}"
            try:
                generated = self.llm.complete(system, prompt, 350)
            except Exception:
                generated = None
            if generated:
                guard = self.checker.check(generated, evidence)
                if not guard["flagged"]:
                    return generated, guard, True, "llm"
                served = self.checker.check(draft, evidence)   # blocked: serve the grounded template instead
                return draft, {**guard, "blocked_reply": generated[:300], "served": "grounded_template",
                               "served_groundedness": served["groundedness"]}, True, "template_after_guard_block"
        return draft, self.checker.check(draft, evidence), False, "knowledge_base_verbatim" if verbatim else "template"

    # ------------------------------------------------------------------ handlers -> (draft, evidence, escalate)
    def _pnr(self, msg, ent, lang, user_id):
        pnr = ent.get("pnr")
        if not pnr:
            return COPY[f"askpnr_{lang}"], {"approved_copy": COPY[f"askpnr_{lang}"]}, False
        masked, rec = mask_pnr(pnr), self.p.pnr_db.get(pnr)
        if not rec:
            text = COPY[f"notfound_{lang}"].format(pnr=masked)
            return text, {"pnr_masked": masked, "found": False, "approved_copy": COPY[f"notfound_{lang}"]}, False
        status = "Cancelled" if rec["cancelled"] else rec["current_status"]
        ev = {"pnr_masked": masked, "train": f"{rec['train_number']} {rec['train_name']}", "class": rec["class"],
              "journey_date": rec["journey_date"], "departure": rec["departure"], "current_status": status,
              "approved_copy": COPY[f"pnr_{lang}"] + COPY[f"pred_{lang}"]}
        text = COPY[f"pnr_{lang}"].format(pnr=masked, train=ev["train"], date=rec["journey_date"], cls=rec["class"],
                                          dep=rec["departure"], status=status)
        kind = "RAC" if status.startswith("RAC") else "WL" if "WL" in status else None
        if kind and not rec["cancelled"]:
            journey = date.fromisoformat(rec["journey_date"])
            snap = {"kind": kind, "position": int(status.split()[-1]),
                    "days_before": max((journey - self.p.today()).days, 0)}
            pred = self.p.tools.confirmation_for(self.p.network.get(rec["train_number"]), rec["class"], journey, snap)
            if pred.get("status") == "ok":
                pct, travel = round(100 * pred["confirmation_probability"]), round(100 * pred["travel_probability"])
                ev.update(confirmation_probability_pct=pct, travel_probability_pct=travel,
                          predicted_final_status=pred["predicted_final_status"], model_version=pred["version"])
                text += COPY[f"pred_{lang}"].format(pct=pct, travel=travel, final=pred["predicted_final_status"])
        return text, ev, False

    def _refund(self, msg, ent, lang, user_id):
        kb = self.p.kb.get("refund_timeline")
        pnr = ent.get("pnr")
        if not pnr:
            text = COPY[f"askpnr_{lang}"] + " " + kb[f"text_{lang}"]
            return text, {"kb": [kb], "approved_copy": COPY[f"askpnr_{lang}"]}, False
        masked, rec = mask_pnr(pnr), self.p.pnr_db.get(pnr)
        if not rec or not rec.get("refund"):
            text = COPY[f"norefund_{lang}"].format(pnr=masked) + " " + kb[f"text_{lang}"]
            return text, {"pnr_masked": masked, "kb": [kb], "approved_copy": COPY[f"norefund_{lang}"]}, False
        r = rec["refund"]
        status = _STATUS_HI.get(r["status"], r["status"]) if lang == "hi" else r["status"]
        text = COPY[f"refund_{lang}"].format(pnr=masked, status=status, amount=r["amount"], on=r["initiated_on"],
                                             mode=r["mode"]) + " " + kb[f"text_{lang}"]
        return text, {"pnr_masked": masked, "refund": {**r, "status_label": status}, "kb": [kb],
                      "approved_copy": COPY[f"refund_{lang}"]}, False

    def _kb(self, msg, ent, lang, user_id):
        hits = self.p.kb.search(msg, lang)
        if not hits:
            return self._clarify(lang)
        parts = [h[f"text_{lang}"] for h in hits]
        parts.append(COPY[f"source_{lang}"].format(source="; ".join(h["source"] for h in hits),
                                                   verified=hits[0]["verified_on"]))
        return " ".join(parts), {"kb": hits, "approved_copy": COPY[f"source_{lang}"]}, False

    def _plan(self, msg, ent, lang, user_id):
        plan = self.p.planner.plan(msg, user_id=user_id)
        if plan["status"] != "plan_ready":
            text = plan.get("question") or plan.get("message") or COPY[f"clarify_{lang}"]
            return text, {"planner": {k: plan.get(k) for k in ("status", "question", "message")}}, False
        b = plan["recommendation"]
        ev = {"train": f"{b['train_number']} {b['train_name']}", "class": b["class_label"], "departure": b["departure"],
              "arrival": b["arrival"], "fare": b["fare"], "availability": b["availability"],
              "travel_probability_pct": round(100 * b["travel_probability"]), "plan_id": plan["plan_id"],
              "approved_copy": COPY[f"plan_{lang}"]}
        fmt = "%d %b %H:%M"
        from datetime import datetime
        text = COPY[f"plan_{lang}"].format(train=ev["train"], cls=ev["class"],
                                           dep=datetime.fromisoformat(b["departure"]).strftime(fmt),
                                           arr=datetime.fromisoformat(b["arrival"]).strftime(fmt), fare=b["fare"],
                                           avail=b["availability"], travel=ev["travel_probability_pct"])
        return text, ev, False

    def _greet(self, msg, ent, lang, user_id):
        return COPY[f"greet_{lang}"], {"approved_copy": COPY[f"greet_{lang}"]}, False

    def _complaint(self, msg, ent, lang, user_id):
        ticket = f"CMP-{uuid.uuid4().hex[:6].upper()}"
        self.p.audit.record("support_assistant", "complaint_logged", ticket, {"message": msg[:300]})
        return COPY[f"complaint_{lang}"].format(ticket=ticket), {"ticket": ticket, "approved_copy": COPY[f"complaint_{lang}"]}, True

    def _clarify(self, lang):
        return COPY[f"clarify_{lang}"], {"approved_copy": COPY[f"clarify_{lang}"]}, False
