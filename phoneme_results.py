from __future__ import annotations

from typing import TypedDict


class PhonemeSegment(TypedDict):
    word: str
    phoneme: str
    start_time: float
    end_time: float
    source: str


class WordSegment(TypedDict):
    text: str
    start_time: float
    end_time: float
    confidence: float


class PhonemeResult(TypedDict):
    expected_phoneme: str
    predicted_phoneme: str
    confidence_score: float
    severity: str
    start_time: float
    end_time: float
    word: str
    source: str


PHONEME_CONFUSIONS: dict[str, list[str]] = {
    "P": ["B", "F"],
    "B": ["P", "V"],
    "F": ["P", "V"],
    "V": ["B", "F", "W"],
    "TH": ["S", "T", "D"],
    "DH": ["D", "Z", "TH"],
    "S": ["SH", "TH"],
    "Z": ["S", "DH"],
    "SH": ["S", "CH"],
    "CH": ["SH", "JH"],
    "R": ["L", "W"],
    "L": ["R"],
    "T": ["D", "TH"],
    "D": ["T", "DH"],
    "K": ["G"],
    "G": ["K"],
    "W": ["V", "B"],
}


def find_matching_word_segment(
    phoneme_segment: PhonemeSegment,
    word_segments: list[WordSegment],
) -> WordSegment | None:
    matches = [segment for segment in word_segments if segment["text"] == phoneme_segment["word"]]

    if not matches:
        return None

    phoneme_midpoint = (phoneme_segment["start_time"] + phoneme_segment["end_time"]) / 2

    def midpoint_distance(segment: WordSegment) -> float:
        segment_midpoint = (segment["start_time"] + segment["end_time"]) / 2
        return abs(segment_midpoint - phoneme_midpoint)

    return min(matches, key=midpoint_distance)


def derive_predicted_phoneme(expected_phoneme: str, confidence: float) -> str:
    if confidence >= 0.8:
        return expected_phoneme

    likely_confusions = PHONEME_CONFUSIONS.get(expected_phoneme, [])

    if confidence >= 0.55:
        return likely_confusions[0] if likely_confusions else expected_phoneme

    if confidence >= 0.35:
        return likely_confusions[1] if len(likely_confusions) > 1 else (
            likely_confusions[0] if likely_confusions else expected_phoneme
        )

    return likely_confusions[-1] if likely_confusions else expected_phoneme


def derive_severity(expected_phoneme: str, predicted_phoneme: str, confidence: float) -> str:
    if predicted_phoneme == expected_phoneme:
        if confidence >= 0.85:
            return "none"
        if confidence >= 0.65:
            return "minor"
        if confidence >= 0.45:
            return "moderate"
        return "severe"

    if confidence >= 0.7:
        return "minor"
    if confidence >= 0.5:
        return "moderate"
    return "severe"


def build_phoneme_results(
    phoneme_segments: list[PhonemeSegment],
    word_segments: list[WordSegment],
) -> list[PhonemeResult]:
    results: list[PhonemeResult] = []

    for phoneme_segment in phoneme_segments:
        matched_word = find_matching_word_segment(phoneme_segment, word_segments)
        confidence = float(matched_word["confidence"]) if matched_word is not None else 0.0

        expected_phoneme = phoneme_segment["phoneme"]
        predicted_phoneme = derive_predicted_phoneme(expected_phoneme, confidence)
        severity = derive_severity(expected_phoneme, predicted_phoneme, confidence)

        results.append(
            {
                "expected_phoneme": expected_phoneme,
                "predicted_phoneme": predicted_phoneme,
                "confidence_score": round(confidence, 4),
                "severity": severity,
                "start_time": round(float(phoneme_segment["start_time"]), 4),
                "end_time": round(float(phoneme_segment["end_time"]), 4),
                "word": phoneme_segment["word"],
                "source": phoneme_segment["source"],
            }
        )

    return results
