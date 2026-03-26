from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import TypedDict

from groq import Groq

from config import settings


class ScoredPhonemeResult(TypedDict):
    expected_phoneme: str
    predicted_phoneme: str
    confidence_score: float
    severity: str
    start_time: float
    end_time: float
    word: str
    source: str
    error_type: str
    severity_score: float
    importance_weight: float
    phoneme_score: float


class WordScore(TypedDict):
    word: str
    weighted_score: float
    average_confidence: float
    phoneme_count: int
    performance_band: str


class ScoringPayload(TypedDict):
    phoneme_results: list[ScoredPhonemeResult]
    word_scores: list[WordScore]
    overall_score: float
    performance_band: str


class FeedbackPayload(TypedDict):
    summary: str
    action_items: list[str]
    encouragement: str
    feedback_provider: str
    feedback_model: str


SYSTEM_PROMPT = """
You are a pronunciation coach for South Asian English learners.

Follow these rules:
- Address the learner directly using "you".
- Focus only on the most important pronunciation issues.
- Name the specific phoneme or sound that needs attention when possible.
- Give simple articulation advice that the learner can act on immediately.
- Match the tone to the performance band:
  - Excellent/Strong: encouraging and light.
  - On Track: constructive and supportive.
  - Needs More Practice/Needs Careful Practice: direct, specific, and encouraging.
- Return valid JSON only.
- Use this exact schema:
  {"summary":"...","action_items":["...","..."],"encouragement":"..."}
- Keep the summary to one short sentence.
- Return exactly 2 action items.
- Keep the encouragement to one short sentence.
""".strip()


def build_payload_description(target_text: str, scoring_payload: ScoringPayload) -> str:
    weakest_phoneme_results = sorted(
        scoring_payload["phoneme_results"],
        key=lambda result: result["phoneme_score"],
    )[:6]

    weakest_word_scores = sorted(
        scoring_payload["word_scores"],
        key=lambda word_score: word_score["weighted_score"],
    )[:4]

    assessment_context = {
        "target_text": target_text,
        "overall_score": round(float(scoring_payload["overall_score"]), 2),
        "performance_band": scoring_payload["performance_band"],
        "weak_words": [
            {
                "word": word_score["word"],
                "score": round(float(word_score["weighted_score"]), 2),
                "average_confidence": round(float(word_score["average_confidence"]), 2),
                "phoneme_count": int(word_score["phoneme_count"]),
                "band": word_score["performance_band"],
            }
            for word_score in weakest_word_scores
        ],
        "weak_phonemes": [
            {
                "word": result["word"],
                "expected": result["expected_phoneme"],
                "predicted": result["predicted_phoneme"],
                "error": result["error_type"],
                "severity": result["severity"],
                "confidence": round(float(result["confidence_score"]), 2),
                "score": round(float(result["phoneme_score"]), 2),
                "importance_weight": round(float(result["importance_weight"]), 2),
            }
            for result in weakest_phoneme_results
        ],
    }

    return json.dumps(
        assessment_context,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_user_prompt(target_text: str, scoring_payload: ScoringPayload) -> str:
    payload_description = build_payload_description(target_text, scoring_payload)
    return (
        "Use the pronunciation assessment JSON below to coach the learner.\n"
        "Ground every point in the provided scores and phoneme errors.\n"
        "Mention only real weaknesses supported by the data.\n"
        "Prioritize the lowest-scoring phonemes and weakest words.\n"
        "Do not describe a phoneme as incorrect when the expected and predicted values match or the error is 'none'.\n"
        "Make the advice specific, practical, and natural-sounding.\n"
        "Return only valid JSON in the required schema.\n"
        f"{payload_description}"
    )


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def build_fallback_feedback(target_text: str, scoring_payload: ScoringPayload) -> FeedbackPayload:
    performance_band = scoring_payload["performance_band"]
    weakest_phonemes = sorted(
        scoring_payload["phoneme_results"],
        key=lambda result: result["phoneme_score"],
    )[:3]

    if weakest_phonemes:
        sound_summary = ", ".join(
            f"{result['expected_phoneme']} in '{result['word']}'" for result in weakest_phonemes
        )
    else:
        sound_summary = "your target sounds"

    if performance_band in {"Excellent", "Strong"}:
        return {
            "summary": f"You pronounced '{target_text}' clearly overall.",
            "action_items": [
                f"Keep refining {sound_summary} so the sounds stay crisp.",
                "Repeat the phrase once more at a steady pace to keep it consistent.",
            ],
            "encouragement": "You are building strong pronunciation control.",
            "feedback_provider": "fallback",
            "feedback_model": "rule-based",
        }

    if performance_band == "On Track":
        return {
            "summary": f"You are getting close on '{target_text}', but {sound_summary} need more precision.",
            "action_items": [
                "Slow down slightly and shape each sound more clearly.",
                "Say the phrase again after isolating the weakest sound once first.",
            ],
            "encouragement": "You are making steady progress.",
            "feedback_provider": "fallback",
            "feedback_model": "rule-based",
        }

    return {
        "summary": f"You need to focus more carefully on {sound_summary} in '{target_text}'.",
        "action_items": [
            "Produce the weakest sound slowly before repeating the whole phrase.",
            "Use clearer mouth placement and avoid rushing the word ending.",
        ],
        "encouragement": "Keep practicing, because you can improve this with repetition.",
        "feedback_provider": "fallback",
        "feedback_model": "rule-based",
    }


def _call_groq(target_text: str, scoring_payload: ScoringPayload) -> FeedbackPayload:
    response = get_groq_client().chat.completions.create(
        model=settings.groq_model,
        temperature=0.15,
        max_completion_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(target_text, scoring_payload)},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Groq returned an empty feedback response.")

    parsed = json.loads(content)
    summary = str(parsed["summary"]).strip()
    action_items = [str(item).strip() for item in parsed["action_items"]][:2]
    encouragement = str(parsed["encouragement"]).strip()

    if len(action_items) < 2:
        raise ValueError("Groq returned fewer than 2 action items.")

    return {
        "summary": summary,
        "action_items": action_items,
        "encouragement": encouragement,
        "feedback_provider": "groq",
        "feedback_model": settings.groq_model,
    }


async def generate_feedback(target_text: str, scoring_payload: ScoringPayload) -> FeedbackPayload:
    if not settings.use_groq or not settings.groq_api_key:
        return build_fallback_feedback(target_text, scoring_payload)

    try:
        return await asyncio.to_thread(_call_groq, target_text, scoring_payload)
    except Exception:
        return build_fallback_feedback(target_text, scoring_payload)
