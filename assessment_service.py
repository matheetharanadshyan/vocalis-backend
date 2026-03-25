from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from alignment import align_target_text
from feedback_generation import generate_feedback
from phoneme_estimates import estimate_phoneme_segments, get_g2p
from phoneme_modeling import load_phoneme_model_bundle, run_phoneme_model
from phoneme_results import build_phoneme_results
from scoring_engine import score_pronunciation


async def preload_assessment_dependencies(
    *,
    run_timed_blocking,
    model_executor: ThreadPoolExecutor,
    load_alignment_bundle,
    logger,
    on_dependency_status=None,
    use_phoneme_model: bool,
) -> None:
    try:
        await run_timed_blocking("alignment.preload", load_alignment_bundle, executor=model_executor)
        if on_dependency_status is not None:
            on_dependency_status("alignment_bundle", ready=True)
    except Exception as error:
        logger.exception("Failed to preload alignment bundle.")
        if on_dependency_status is not None:
            on_dependency_status("alignment_bundle", ready=False, error=str(error))

    try:
        await run_timed_blocking("g2p.preload", get_g2p, executor=model_executor)
        if on_dependency_status is not None:
            on_dependency_status("g2p", ready=True)
    except Exception as error:
        logger.exception("Failed to preload grapheme-to-phoneme resources.")
        if on_dependency_status is not None:
            on_dependency_status("g2p", ready=False, error=str(error))

    if use_phoneme_model:
        try:
            await run_timed_blocking(
                "phoneme_model.preload",
                load_phoneme_model_bundle,
                executor=model_executor,
            )
            if on_dependency_status is not None:
                on_dependency_status("phoneme_model", ready=True)
        except Exception as error:
            logger.exception("Failed to preload phoneme model bundle.")
            if on_dependency_status is not None:
                on_dependency_status("phoneme_model", ready=False, error=str(error))


async def assess_pronunciation(
    *,
    audio_bytes: bytes,
    current_target_text: dict,
    preprocess_audio_bytes,
    run_timed_blocking,
    run_timed_async,
    io_executor: ThreadPoolExecutor,
    model_executor: ThreadPoolExecutor,
    alignment_semaphore: asyncio.Semaphore,
    phoneme_model_semaphore: asyncio.Semaphore,
    feedback_semaphore: asyncio.Semaphore,
) -> dict:
    processed = await run_timed_blocking(
        "audio.preprocess",
        preprocess_audio_bytes,
        audio_bytes,
        executor=io_executor,
    )
    alignment_task = asyncio.create_task(
        run_timed_blocking(
            "audio.alignment",
            align_target_text,
            processed["processed_audio"],
            current_target_text["text"],
            executor=model_executor,
            semaphore=alignment_semaphore,
        )
    )
    phoneme_model_task = asyncio.create_task(
        run_timed_blocking(
            "audio.phoneme_model",
            run_phoneme_model,
            processed["processed_audio"],
            executor=model_executor,
            semaphore=phoneme_model_semaphore,
        )
    )
    alignment, phoneme_model_payload = await asyncio.gather(
        alignment_task,
        phoneme_model_task,
    )
    phoneme_segments = await run_timed_blocking(
        "audio.estimate_phonemes",
        estimate_phoneme_segments,
        target_text=current_target_text["text"],
        word_segments=alignment["word_segments"],
        executor=io_executor,
    )
    phoneme_results = await run_timed_blocking(
        "audio.build_phoneme_results",
        build_phoneme_results,
        phoneme_segments,
        alignment["word_segments"],
        executor=io_executor,
    )
    scoring_payload = await run_timed_blocking(
        "audio.score",
        score_pronunciation,
        phoneme_results,
        executor=io_executor,
    )
    feedback_payload = await run_timed_async(
        "audio.feedback",
        generate_feedback(
            target_text=current_target_text["text"],
            scoring_payload=scoring_payload,
        ),
        semaphore=feedback_semaphore,
    )

    return {
        "processed": processed,
        "alignment": alignment,
        "phoneme_segments": phoneme_segments,
        "phoneme_model_payload": phoneme_model_payload,
        "scoring_payload": scoring_payload,
        "feedback_payload": feedback_payload,
    }
