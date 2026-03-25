from __future__ import annotations

import pytest

import scoring_engine


@pytest.mark.parametrize(
    ("score", "expected_band"),
    [
        (0.95, "Excellent"),
        (0.84, "Strong"),
        (0.70, "On Track"),
        (0.50, "Needs More Practice"),
        (0.20, "Needs Careful Practice"),
    ],
)
def test_performance_band_from_score_thresholds(score: float, expected_band: str) -> None:
    assert scoring_engine.performance_band_from_score(score) == expected_band


def test_classify_error_type_prioritizes_deletion_for_very_low_confidence() -> None:
    assert scoring_engine.classify_error_type("TH", "T", 0.15) == "deletion"
    assert scoring_engine.classify_error_type("TH", "T", 0.85) == "substitution"
    assert scoring_engine.classify_error_type("TH", "TH", 0.6) == "distortion"
    assert scoring_engine.classify_error_type("TH", "TH", 0.95) == "none"


def test_score_pronunciation_aggregates_word_and_overall_scores() -> None:
    payload = scoring_engine.score_pronunciation(
        [
            {
                "expected_phoneme": "TH",
                "predicted_phoneme": "TH",
                "confidence_score": 0.95,
                "severity": "none",
                "start_time": 0.0,
                "end_time": 0.2,
                "word": "think",
                "source": "estimated_from_word_alignment",
            },
            {
                "expected_phoneme": "AE",
                "predicted_phoneme": "EH",
                "confidence_score": 0.85,
                "severity": "none",
                "start_time": 0.2,
                "end_time": 0.4,
                "word": "cat",
                "source": "estimated_from_word_alignment",
            },
        ]
    )

    assert payload["phoneme_results"][0]["importance_weight"] == pytest.approx(1.35)
    assert payload["phoneme_results"][0]["phoneme_score"] == pytest.approx(1.0)
    assert payload["phoneme_results"][1]["error_type"] == "substitution"
    assert payload["word_scores"] == [
        {
            "word": "think",
            "weighted_score": pytest.approx(1.0),
            "average_confidence": pytest.approx(0.95),
            "phoneme_count": 1,
            "performance_band": "Excellent",
        },
        {
            "word": "cat",
            "weighted_score": pytest.approx(0.4165),
            "average_confidence": pytest.approx(0.85),
            "phoneme_count": 1,
            "performance_band": "Needs Careful Practice",
        },
    ]
    assert payload["overall_score"] == pytest.approx(0.7083, abs=1e-4)
    assert payload["performance_band"] == "On Track"
