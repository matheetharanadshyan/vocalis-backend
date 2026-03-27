from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from personalization_repository import (
    build_user_personalization_summary,
    save_attempt,
    save_attempt_phoneme_summaries,
    upsert_phoneme_memory,
)
from target_texts import TargetTextManager


def derive_attempt_focus_phonemes(phoneme_results: list[dict], limit: int = 3) -> list[str]:
    ranked_results = sorted(
        (
            result
            for result in phoneme_results
            if str(result.get("error_type", "none")) != "none" or float(result.get("phoneme_score", 1.0)) < 0.75
        ),
        key=lambda result: (
            float(result.get("phoneme_score", 0.0)),
            -float(result.get("severity_score", 0.0)),
            -float(result.get("importance_weight", 0.0)),
        ),
    )

    focus_phonemes: list[str] = []
    for result in ranked_results:
        phoneme = str(result.get("expected_phoneme", "")).strip().upper()
        if not phoneme or phoneme in focus_phonemes:
            continue
        focus_phonemes.append(phoneme)
        if len(focus_phonemes) >= limit:
            break

    return focus_phonemes


async def resolve_practice_context(
    *,
    user_id: int | None,
    target_text_manager: TargetTextManager,
    run_timed_blocking,
    io_executor: ThreadPoolExecutor,
) -> tuple[dict | None, list[str], dict, list[str]]:
    personalization_summary = (
        await run_timed_blocking(
            "personalization.initial_summary",
            build_user_personalization_summary,
            user_id,
            executor=io_executor,
        )
        if user_id is not None
        else None
    )
    focus_phonemes = (
        [entry["phoneme"] for entry in personalization_summary["focus_phonemes"]]
        if personalization_summary
        else []
    )
    current_target_text = target_text_manager.next_target_for_focus(focus_phonemes)
    current_target_focus_matches = target_text_manager.match_focus_phonemes(
        current_target_text["text"],
        focus_phonemes,
    )
    return personalization_summary, focus_phonemes, current_target_text, current_target_focus_matches


async def persist_personalization_state(
    *,
    user_id: int | None,
    current_target_text: dict,
    alignment: dict,
    scoring_payload: dict,
    feedback_payload: dict,
    phoneme_model_payload: dict,
    target_text_manager: TargetTextManager,
    run_timed_blocking,
    io_executor: ThreadPoolExecutor,
) -> tuple[dict | None, list[str], dict]:
    if user_id is not None:
        attempt_id = await run_timed_blocking(
            "attempt.save",
            save_attempt,
            user_id,
            target_text=current_target_text["text"],
            target_difficulty=current_target_text["difficulty"],
            normalized_target_text=alignment["normalized_target_text"],
            overall_score=scoring_payload["overall_score"],
            performance_band=scoring_payload["performance_band"],
            feedback_summary=feedback_payload["summary"],
            feedback_action_items=feedback_payload["action_items"],
            feedback_encouragement=feedback_payload["encouragement"],
            word_scores=scoring_payload["word_scores"],
            phoneme_results=scoring_payload["phoneme_results"],
            phoneme_model_transcript=phoneme_model_payload["phoneme_model_transcript"],
            executor=io_executor,
        )
        await run_timed_blocking(
            "attempt.save_phoneme_summaries",
            save_attempt_phoneme_summaries,
            user_id,
            attempt_id,
            scoring_payload["phoneme_results"],
            executor=io_executor,
        )
        await run_timed_blocking(
            "phoneme_memory.upsert",
            upsert_phoneme_memory,
            user_id,
            scoring_payload["phoneme_results"],
            executor=io_executor,
        )
        personalization_summary = await run_timed_blocking(
            "personalization.summary",
            build_user_personalization_summary,
            user_id,
            executor=io_executor,
        )
        summary_focus_phonemes = [entry["phoneme"] for entry in personalization_summary["focus_phonemes"]]
        attempt_focus_phonemes = derive_attempt_focus_phonemes(scoring_payload["phoneme_results"])
        focus_phonemes = attempt_focus_phonemes or summary_focus_phonemes
        next_target_text = target_text_manager.next_target_for_focus(focus_phonemes)
        return personalization_summary, focus_phonemes, next_target_text

    next_target_text = target_text_manager.next_target()
    return None, [], next_target_text
