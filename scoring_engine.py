from __future__ import annotations

from collections import defaultdict
from typing import Literal, TypedDict


ErrorType = Literal["substitution", "deletion", "insertion", "distortion", "none"]
Severity = Literal["none", "mild", "moderate", "severe"]
PerformanceBand = Literal[
    "Excellent",
    "Strong",
    "On Track",
    "Needs More Practice",
    "Needs Careful Practice",
]


class PhonemeResult(TypedDict):
    expected_phoneme: str
    predicted_phoneme: str
    confidence_score: float
    severity: str
    start_time: float
    end_time: float
    word: str
    source: str


class ScoredPhonemeResult(TypedDict):
    expected_phoneme: str
    predicted_phoneme: str
    confidence_score: float
    severity: Severity
    start_time: float
    end_time: float
    word: str
    source: str
    error_type: ErrorType
    severity_score: float
    importance_weight: float
    phoneme_score: float


class WordScore(TypedDict):
    word: str
    weighted_score: float
    average_confidence: float
    phoneme_count: int
    performance_band: PerformanceBand


class ScoringPayload(TypedDict):
    phoneme_results: list[ScoredPhonemeResult]
    word_scores: list[WordScore]
    overall_score: float
    performance_band: PerformanceBand


ERROR_TYPE_WEIGHTS: dict[ErrorType, float] = {
    "none": 0.0,
    "insertion": 0.2,
    "distortion": 0.35,
    "deletion": 0.55,
    "substitution": 0.7,
}

HIGH_IMPORTANCE_PHONEMES = {
    "TH",
    "DH",
    "F",
    "V",
    "P",
    "B",
    "S",
    "Z",
    "SH",
    "CH",
    "R",
    "L",
    "W",
}


def classify_error_type(
    expected_phoneme: str,
    predicted_phoneme: str,
    confidence_score: float,
) -> ErrorType:
    if confidence_score <= 0.2:
        return "deletion"

    if predicted_phoneme != expected_phoneme:
        return "substitution"

    if confidence_score < 0.7:
        return "distortion"

    return "none"


def get_importance_weight(phoneme: str) -> float:
    if phoneme in HIGH_IMPORTANCE_PHONEMES:
        return 1.35

    return 1.0


def compute_severity_score(error_type: ErrorType, confidence_score: float) -> float:
    confidence_penalty = 1.0 - max(0.0, min(1.0, confidence_score))
    return min(1.0, confidence_penalty + ERROR_TYPE_WEIGHTS[error_type])


def severity_from_score(severity_score: float) -> Severity:
    if severity_score < 0.2:
        return "none"
    if severity_score < 0.45:
        return "mild"
    if severity_score < 0.7:
        return "moderate"
    return "severe"


def compute_phoneme_score(
    confidence_score: float,
    severity_score: float,
    importance_weight: float,
) -> float:
    base_score = max(0.0, confidence_score * (1.0 - 0.6 * severity_score))
    weighted_score = base_score * importance_weight
    return min(1.0, weighted_score)


def performance_band_from_score(score: float) -> PerformanceBand:
    if score >= 0.9:
        return "Excellent"
    if score >= 0.78:
        return "Strong"
    if score >= 0.62:
        return "On Track"
    if score >= 0.45:
        return "Needs More Practice"
    return "Needs Careful Practice"


def score_pronunciation(phoneme_results: list[PhonemeResult]) -> ScoringPayload:
    scored_phoneme_results: list[ScoredPhonemeResult] = []

    for result in phoneme_results:
        error_type = classify_error_type(
            expected_phoneme=result["expected_phoneme"],
            predicted_phoneme=result["predicted_phoneme"],
            confidence_score=result["confidence_score"],
        )
        importance_weight = get_importance_weight(result["expected_phoneme"])
        severity_score = compute_severity_score(error_type, result["confidence_score"])
        severity = severity_from_score(severity_score)
        phoneme_score = compute_phoneme_score(
            confidence_score=result["confidence_score"],
            severity_score=severity_score,
            importance_weight=importance_weight,
        )

        scored_phoneme_results.append(
            {
                **result,
                "severity": severity,
                "error_type": error_type,
                "severity_score": round(severity_score, 4),
                "importance_weight": round(importance_weight, 4),
                "phoneme_score": round(phoneme_score, 4),
            }
        )

    phonemes_by_word: dict[str, list[ScoredPhonemeResult]] = defaultdict(list)
    for result in scored_phoneme_results:
        phonemes_by_word[result["word"]].append(result)

    word_scores: list[WordScore] = []
    for word, phonemes in phonemes_by_word.items():
        weighted_sum = sum(
            phoneme["phoneme_score"] * phoneme["importance_weight"] for phoneme in phonemes
        )
        weight_sum = sum(phoneme["importance_weight"] for phoneme in phonemes) or 1.0
        weighted_score = weighted_sum / weight_sum
        average_confidence = sum(phoneme["confidence_score"] for phoneme in phonemes) / len(phonemes)

        word_scores.append(
            {
                "word": word,
                "weighted_score": round(weighted_score, 4),
                "average_confidence": round(average_confidence, 4),
                "phoneme_count": len(phonemes),
                "performance_band": performance_band_from_score(weighted_score),
            }
        )

    if word_scores:
        overall_score = sum(word["weighted_score"] for word in word_scores) / len(word_scores)
    else:
        overall_score = 0.0

    return {
        "phoneme_results": scored_phoneme_results,
        "word_scores": word_scores,
        "overall_score": round(overall_score, 4),
        "performance_band": performance_band_from_score(overall_score),
    }
