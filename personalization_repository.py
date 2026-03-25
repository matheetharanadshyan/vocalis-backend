from __future__ import annotations

import json
from collections import Counter, defaultdict
from math import sqrt

from database import execute, execute_many, fetch_all, fetch_one


def save_attempt(
    user_id: int,
    *,
    target_text: str,
    target_difficulty: str,
    normalized_target_text: str,
    overall_score: float,
    performance_band: str,
    feedback_summary: str,
    feedback_action_items: list[str],
    feedback_encouragement: str,
    word_scores: list[dict],
    phoneme_results: list[dict],
    phoneme_model_transcript: str,
) -> int:
    cursor = execute(
        """
        INSERT INTO attempts (
            user_id,
            target_text,
            target_difficulty,
            normalized_target_text,
            overall_score,
            performance_band,
            feedback_summary,
            feedback_action_items_json,
            feedback_encouragement,
            word_scores_json,
            phoneme_results_json,
            phoneme_model_transcript
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            target_text,
            target_difficulty,
            normalized_target_text,
            overall_score,
            performance_band,
            feedback_summary,
            json.dumps(feedback_action_items),
            feedback_encouragement,
            json.dumps(word_scores),
            json.dumps(phoneme_results),
            phoneme_model_transcript,
        ),
    )
    return int(cursor.lastrowid)


def get_attempt_summary(user_id: int):
    return fetch_one(
        """
        SELECT
            COUNT(*),
            AVG(overall_score),
            MAX(created_at)
        FROM attempts
        WHERE user_id = ?
        """,
        (user_id,),
    )


def get_recent_attempts(user_id: int, limit: int = 5):
    return fetch_all(
        """
        SELECT
            id,
            target_text,
            target_difficulty,
            overall_score,
            performance_band,
            created_at
        FROM attempts
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )


def get_recent_attempt_phoneme_results(user_id: int, limit: int = 12):
    return fetch_all(
        """
        SELECT
            phoneme_results_json,
            created_at
        FROM attempts
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )


def save_attempt_phoneme_summaries(
    user_id: int,
    attempt_id: int,
    phoneme_results: list[dict],
) -> None:
    grouped_results: dict[str, list[dict]] = defaultdict(list)
    for result in phoneme_results:
        phoneme = str(result.get("expected_phoneme", "")).strip()
        if not phoneme:
            continue

        grouped_results[phoneme].append(result)

    if not grouped_results:
        return

    rows: list[tuple] = []
    for phoneme, grouped in grouped_results.items():
        average_phoneme_score = sum(float(result.get("phoneme_score", 0.0)) for result in grouped) / len(grouped)
        average_severity_score = sum(float(result.get("severity_score", 0.0)) for result in grouped) / len(grouped)
        weak_occurrence_count = sum(
            1
            for result in grouped
            if str(result.get("error_type", "none")) != "none" or float(result.get("phoneme_score", 0.0)) < 0.75
        )
        rows.append(
            (
                attempt_id,
                user_id,
                phoneme,
                round(average_phoneme_score, 4),
                round(average_severity_score, 4),
                len(grouped),
                weak_occurrence_count,
            )
        )

    execute_many(
        """
        INSERT INTO phoneme_attempt_summaries (
            attempt_id,
            user_id,
            phoneme,
            average_phoneme_score,
            average_severity_score,
            occurrence_count,
            weak_occurrence_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def get_recent_phoneme_attempt_summaries(user_id: int, phonemes: list[str], limit: int = 16):
    normalized_phonemes = [phoneme for phoneme in phonemes if phoneme]
    if not normalized_phonemes:
        return []

    placeholders = ", ".join(["?"] * len(normalized_phonemes))
    return fetch_all(
        f"""
        SELECT
            phoneme,
            average_phoneme_score,
            average_severity_score,
            occurrence_count,
            weak_occurrence_count,
            created_at
        FROM phoneme_attempt_summaries
        WHERE user_id = ? AND phoneme IN ({placeholders})
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, *normalized_phonemes, limit),
    )


def get_attempt_history(user_id: int, limit: int = 50):
    return fetch_all(
        """
        SELECT
            id,
            target_text,
            target_difficulty,
            overall_score,
            performance_band,
            feedback_summary,
            created_at
        FROM attempts
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )


def get_focus_phonemes(user_id: int, limit: int = 3):
    return fetch_all(
        """
        SELECT
            phoneme,
            total_occurrences,
            weak_occurrences,
            average_score,
            average_severity_score,
            recent_weighted_score,
            common_error_types_json,
            last_seen_at
        FROM phoneme_memory
        WHERE user_id = ?
        ORDER BY recent_weighted_score DESC, weak_occurrences DESC, total_occurrences DESC
        LIMIT ?
        """,
        (user_id, limit),
    )


def get_phoneme_memory_entry(user_id: int, phoneme: str):
    return fetch_one(
        """
        SELECT
            id,
            total_occurrences,
            weak_occurrences,
            average_score,
            average_severity_score,
            recent_weighted_score,
            common_error_types_json
        FROM phoneme_memory
        WHERE user_id = ? AND phoneme = ?
        """,
        (user_id, phoneme),
    )


def get_phoneme_memory_entries(user_id: int, phonemes: list[str]):
    normalized_phonemes = [phoneme for phoneme in phonemes if phoneme]
    if not normalized_phonemes:
        return []

    placeholders = ", ".join(["?"] * len(normalized_phonemes))
    return fetch_all(
        f"""
        SELECT
            id,
            phoneme,
            total_occurrences,
            weak_occurrences,
            average_score,
            average_severity_score,
            recent_weighted_score,
            common_error_types_json
        FROM phoneme_memory
        WHERE user_id = ? AND phoneme IN ({placeholders})
        """,
        (user_id, *normalized_phonemes),
    )


def upsert_phoneme_memory(user_id: int, phoneme_results: list[dict]) -> None:
    results_by_phoneme: dict[str, list[dict]] = defaultdict(list)
    for result in phoneme_results:
        phoneme = str(result.get("expected_phoneme", "")).strip()
        if not phoneme:
            continue

        results_by_phoneme[phoneme].append(result)

    if not results_by_phoneme:
        return

    existing_rows = get_phoneme_memory_entries(user_id, list(results_by_phoneme.keys()))
    existing_by_phoneme = {
        str(row[1]): {
            "id": int(row[0]),
            "total_occurrences": int(row[2]),
            "weak_occurrences": int(row[3]),
            "average_score": float(row[4]),
            "average_severity_score": float(row[5]),
            "recent_weighted_score": float(row[6]),
            "common_error_types": Counter(json.loads(str(row[7] or "[]"))),
        }
        for row in existing_rows
    }

    inserts: list[tuple] = []
    updates: list[tuple] = []

    for phoneme, grouped_results in results_by_phoneme.items():
        existing = existing_by_phoneme.get(phoneme)

        if existing is None:
            total_occurrences = 0
            weak_occurrences = 0
            average_score = 0.0
            average_severity_score = 0.0
            recent_weighted_score = 0.0
            common_error_types: Counter[str] = Counter()
            entry_id = None
        else:
            total_occurrences = existing["total_occurrences"]
            weak_occurrences = existing["weak_occurrences"]
            average_score = existing["average_score"]
            average_severity_score = existing["average_severity_score"]
            recent_weighted_score = existing["recent_weighted_score"]
            common_error_types = existing["common_error_types"]
            entry_id = existing["id"]

        for result in grouped_results:
            phoneme_score = float(result.get("phoneme_score", 0.0))
            severity_score = float(result.get("severity_score", 0.0))
            error_type = str(result.get("error_type", "none"))
            is_weak = error_type != "none" or phoneme_score < 0.75

            updated_total_occurrences = total_occurrences + 1
            updated_weak_occurrences = weak_occurrences + (1 if is_weak else 0)
            average_score = (
                ((average_score * total_occurrences) + phoneme_score) / updated_total_occurrences
            )
            average_severity_score = (
                ((average_severity_score * total_occurrences) + severity_score) / updated_total_occurrences
            )
            recent_weighted_score = (
                recent_weighted_score * 0.72
                + (1.0 - phoneme_score)
                + severity_score
                + (0.35 if is_weak else 0.0)
            )
            total_occurrences = updated_total_occurrences
            weak_occurrences = updated_weak_occurrences

            if error_type != "none":
                common_error_types[error_type] += 1

        flattened_error_types: list[str] = []
        for name, count in common_error_types.most_common(4):
            flattened_error_types.extend([name] * count)

        row_payload = (
            total_occurrences,
            weak_occurrences,
            average_score,
            average_severity_score,
            recent_weighted_score,
            json.dumps(flattened_error_types),
        )
        if entry_id is None:
            inserts.append((user_id, phoneme, *row_payload))
            continue

        updates.append((*row_payload, entry_id))

    if inserts:
        execute_many(
            """
            INSERT INTO phoneme_memory (
                user_id,
                phoneme,
                total_occurrences,
                weak_occurrences,
                average_score,
                average_severity_score,
                recent_weighted_score,
                common_error_types_json,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            inserts,
        )

    if updates:
        execute_many(
            """
            UPDATE phoneme_memory
            SET
                total_occurrences = ?,
                weak_occurrences = ?,
                average_score = ?,
                average_severity_score = ?,
                recent_weighted_score = ?,
                common_error_types_json = ?,
                last_seen_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            updates,
        )


def build_user_personalization_summary(user_id: int) -> dict:
    summary_row = get_attempt_summary(user_id)
    recent_attempt_rows = get_recent_attempts(user_id, limit=4)
    focus_rows = get_focus_phonemes(user_id, limit=3)

    attempt_count = int(summary_row[0]) if summary_row and summary_row[0] is not None else 0
    average_score = float(summary_row[1]) if summary_row and summary_row[1] is not None else 0.0
    last_attempt_at = str(summary_row[2]) if summary_row and summary_row[2] else None

    recent_attempts = [
        {
            "id": int(row[0]),
            "target_text": str(row[1]),
            "target_difficulty": str(row[2]),
            "overall_score": float(row[3]),
            "performance_band": str(row[4]),
            "created_at": str(row[5]),
        }
        for row in recent_attempt_rows
    ]

    phoneme_trend_map = build_focus_phoneme_trends(
        user_id,
        [str(row[0]) for row in focus_rows],
    )
    focus_phonemes = []
    for row in focus_rows:
        phoneme = str(row[0])
        trend_payload = phoneme_trend_map.get(
            phoneme,
            {
                "recent_scores": [],
                "weighted_trend_delta": 0.0,
                "trend_direction": "stable",
                "trend_summary": "Not enough recent data yet.",
                "consistency_score": 0.0,
                "consistency_direction": "stable",
                "consistency_summary": "Consistency needs a few more saved attempts.",
                "trend_confidence": 0.0,
                "trend_confidence_label": "low",
            },
        )
        focus_phonemes.append(
            {
                "phoneme": phoneme,
                "total_occurrences": int(row[1]),
                "weak_occurrences": int(row[2]),
                "average_score": float(row[3]),
                "average_severity_score": float(row[4]),
                "recent_weighted_score": float(row[5]),
                "common_error_types": json.loads(str(row[6] or "[]")),
                "last_seen_at": str(row[7]),
                "recent_scores": trend_payload["recent_scores"],
                "weighted_trend_delta": trend_payload["weighted_trend_delta"],
                "trend_direction": trend_payload["trend_direction"],
                "trend_summary": trend_payload["trend_summary"],
                "consistency_score": trend_payload["consistency_score"],
                "consistency_direction": trend_payload["consistency_direction"],
                "consistency_summary": trend_payload["consistency_summary"],
                "trend_confidence": trend_payload["trend_confidence"],
                "trend_confidence_label": trend_payload["trend_confidence_label"],
            }
        )

    current_focus = focus_phonemes[0]["phoneme"] if focus_phonemes else None
    recurring_sound_note = None
    if focus_phonemes:
        recurring_sound_note = f"You often struggle with /{focus_phonemes[0]['phoneme']}/."

    improvement_note = None
    consistency_note = None

    if len(recent_attempts) >= 2:
        latest_score = recent_attempts[0]["overall_score"]
        previous_score = recent_attempts[1]["overall_score"]
        score_delta = latest_score - previous_score

        if score_delta >= 0.08:
            improvement_note = "Your scores are improving noticeably."
        elif score_delta >= 0.03:
            improvement_note = "Your latest attempt was a step forward."
        elif score_delta <= -0.08:
            improvement_note = "Today’s attempts are a little less stable, so slower repetition may help."

        recent_average = sum(attempt["overall_score"] for attempt in recent_attempts[:3]) / min(
            len(recent_attempts), 3
        )
        if abs(latest_score - recent_average) <= 0.04:
            consistency_note = "Your recent pronunciation is becoming more consistent."

    if focus_phonemes and focus_phonemes[0]["average_score"] >= 0.72:
        improvement_note = f"You are starting to improve on /{focus_phonemes[0]['phoneme']}/."

    return {
        "attempt_count": attempt_count,
        "average_score": average_score,
        "last_attempt_at": last_attempt_at,
        "recent_attempts": recent_attempts,
        "focus_phonemes": focus_phonemes,
        "current_focus": current_focus,
        "recurring_sound_note": recurring_sound_note,
        "improvement_note": improvement_note,
        "consistency_note": consistency_note,
    }


def build_focus_phoneme_trends(user_id: int, focus_phonemes: list[str], recent_attempt_limit: int = 16) -> dict[str, dict]:
    tracked_phonemes = [phoneme for phoneme in focus_phonemes if phoneme]
    if not tracked_phonemes:
        return {}

    recent_rows = get_recent_phoneme_attempt_summaries(
        user_id,
        tracked_phonemes,
        limit=recent_attempt_limit,
    )
    score_history: dict[str, list[float]] = {phoneme: [] for phoneme in tracked_phonemes}

    for row in reversed(recent_rows):
        phoneme = str(row[0])
        if phoneme not in score_history:
            continue

        score_history[phoneme].append(float(row[1]))

    trend_map: dict[str, dict] = {}
    for phoneme, scores in score_history.items():
        recent_scores = scores[-4:]
        weighted_trend_delta = compute_weighted_trend_delta(recent_scores)
        consistency_score = compute_consistency_score(recent_scores)
        trend_confidence = compute_trend_confidence(recent_scores, consistency_score)
        trend_confidence_label = confidence_label_from_score(trend_confidence)

        if len(recent_scores) < 2:
            trend_direction = "stable"
            consistency_direction = "stable"
            trend_summary = "Not enough recent data yet."
            consistency_summary = "Consistency needs a few more saved attempts."
        else:
            if weighted_trend_delta >= 0.035:
                trend_direction = "improving"
                trend_summary = (
                    f"Recent weighted trend is up {abs(weighted_trend_delta) * 100:.0f} points."
                )
            elif weighted_trend_delta <= -0.035:
                trend_direction = "declining"
                trend_summary = (
                    f"Recent weighted trend is down {abs(weighted_trend_delta) * 100:.0f} points."
                )
            else:
                trend_direction = "stable"
                trend_summary = "Recent improvement is mostly steady."

            if consistency_score >= 0.82:
                consistency_direction = "improving"
                consistency_summary = "Recent attempts on this sound are becoming more consistent."
            elif consistency_score <= 0.58:
                consistency_direction = "declining"
                consistency_summary = "Recent attempts on this sound are still quite uneven."
            else:
                consistency_direction = "stable"
                consistency_summary = "Recent attempts on this sound show moderate consistency."

            if trend_confidence < 0.45:
                trend_summary = f"{trend_summary} Confidence is still limited with this amount of data."

        trend_map[phoneme] = {
            "recent_scores": recent_scores,
            "weighted_trend_delta": weighted_trend_delta,
            "trend_direction": trend_direction,
            "trend_summary": trend_summary,
            "consistency_score": consistency_score,
            "consistency_direction": consistency_direction,
            "consistency_summary": consistency_summary,
            "trend_confidence": trend_confidence,
            "trend_confidence_label": trend_confidence_label,
        }

    return trend_map


def compute_weighted_trend_delta(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0

    weights = list(range(1, len(scores) + 1))
    midpoint = len(scores) // 2
    earlier_scores = scores[:midpoint]
    later_scores = scores[midpoint:]
    earlier_weights = weights[:midpoint]
    later_weights = weights[midpoint:]

    if not earlier_scores or not later_scores:
        return round(scores[-1] - scores[0], 4)

    earlier_average = sum(score * weight for score, weight in zip(earlier_scores, earlier_weights)) / sum(
        earlier_weights
    )
    later_average = sum(score * weight for score, weight in zip(later_scores, later_weights)) / sum(
        later_weights
    )
    return round(later_average - earlier_average, 4)


def compute_consistency_score(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    normalized_spread = min(1.0, sqrt(variance) / 0.18)
    return round(max(0.0, 1.0 - normalized_spread), 4)


def compute_trend_confidence(scores: list[float], consistency_score: float) -> float:
    if not scores:
        return 0.0

    sample_confidence = min(1.0, len(scores) / 4)
    confidence = sample_confidence * 0.65 + consistency_score * 0.35
    return round(confidence, 4)


def confidence_label_from_score(confidence_score: float) -> str:
    if confidence_score >= 0.8:
        return "high"
    if confidence_score >= 0.55:
        return "medium"
    return "low"
