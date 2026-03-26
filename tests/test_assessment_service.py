from __future__ import annotations

import asyncio
from types import ModuleType


def build_assessment_service_module(load_backend_module):
    alignment_module = ModuleType("alignment")
    alignment_module.align_target_text = lambda processed_audio, target_text: {
        "normalized_target_text": target_text,
        "character_segments": [],
        "word_segments": [
            {
                "text": target_text,
                "start_time": 0.0,
                "end_time": 1.0,
                "confidence": 0.95,
            }
        ],
        "model_quantized": True,
        "model_device": "cpu",
    }

    feedback_generation_module = ModuleType("feedback_generation")

    async def generate_feedback(*, target_text: str, scoring_payload: dict) -> dict:
        return {
            "summary": f"Feedback for {target_text}",
            "action_items": ["Keep going", "Slow down a little"],
            "encouragement": "Nice work",
            "feedback_provider": "fallback",
            "feedback_model": "rule-based",
        }

    feedback_generation_module.generate_feedback = generate_feedback

    phoneme_estimates_module = ModuleType("phoneme_estimates")
    phoneme_estimates_module.get_g2p = lambda: object()
    phoneme_estimates_module.estimate_phoneme_segments = lambda target_text, word_segments: [
        {
            "word": target_text,
            "phoneme": "HH",
            "start_time": 0.0,
            "end_time": 0.5,
            "source": "estimated_from_word_alignment",
        }
    ]

    phoneme_modeling_module = ModuleType("phoneme_modeling")
    phoneme_modeling_module.load_phoneme_model_bundle = lambda: object()
    phoneme_modeling_module.run_phoneme_model = lambda processed_audio: {
        "phoneme_model_used": True,
        "phoneme_model_id": "test-model",
        "phoneme_model_segments": [],
        "phoneme_model_transcript": "hello",
        "phoneme_model_error": "",
        "phoneme_model_quantized": True,
    }

    phoneme_results_module = ModuleType("phoneme_results")
    phoneme_results_module.build_phoneme_results = lambda phoneme_segments, word_segments: [
        {
            "expected_phoneme": "HH",
            "predicted_phoneme": "HH",
            "confidence_score": 0.95,
            "severity": "low",
            "start_time": 0.0,
            "end_time": 0.5,
            "word": "hello",
            "source": "estimated_from_word_alignment",
            "error_type": "match",
            "severity_score": 0.1,
            "importance_weight": 1.0,
            "phoneme_score": 0.95,
        }
    ]

    scoring_engine_module = ModuleType("scoring_engine")
    scoring_engine_module.score_pronunciation = lambda phoneme_results: {
        "phoneme_results": phoneme_results,
        "word_scores": [],
        "overall_score": 95.0,
        "performance_band": "Strong",
    }

    return load_backend_module(
        "assessment_service",
        {
            "alignment": alignment_module,
            "feedback_generation": feedback_generation_module,
            "phoneme_estimates": phoneme_estimates_module,
            "phoneme_modeling": phoneme_modeling_module,
            "phoneme_results": phoneme_results_module,
            "scoring_engine": scoring_engine_module,
        },
    )


def test_assess_pronunciation_reports_progress_updates_and_timings(load_backend_module) -> None:
    assessment_service = build_assessment_service_module(load_backend_module)
    progress_payloads: list[dict] = []

    async def progress_notifier(payload: dict) -> None:
        progress_payloads.append(payload)

    async def run_timed_blocking(stage_name: str, func, *args, **kwargs):
        kwargs = {key: value for key, value in kwargs.items() if key not in {"executor", "semaphore"}}
        return func(*args, **kwargs)

    async def run_timed_async(stage_name: str, awaitable, **kwargs):
        return await awaitable

    def preprocess_audio_bytes(audio_bytes: bytes) -> dict:
        return {
            "processed_audio": [0.1, 0.2, 0.3],
            "original_sample_rate": 48000,
            "processed_sample_rate": 16000,
            "original_channels": 1,
            "processed_channels": 1,
            "num_samples": 3,
            "duration_seconds": 0.1,
        }

    result = asyncio.run(
        assessment_service.assess_pronunciation(
            audio_bytes=b"audio",
            current_target_text={"text": "hello", "difficulty": "easy"},
            preprocess_audio_bytes=preprocess_audio_bytes,
            run_timed_blocking=run_timed_blocking,
            run_timed_async=run_timed_async,
            io_executor=object(),
            model_executor=object(),
            alignment_semaphore=asyncio.Semaphore(1),
            phoneme_model_semaphore=asyncio.Semaphore(1),
            feedback_semaphore=asyncio.Semaphore(1),
            progress_notifier=progress_notifier,
        )
    )

    assert [payload["type"] for payload in progress_payloads] == [
        "audio.aligning",
        "audio.scoring",
        "audio.feedback",
    ]
    assert set(result["timings"]) >= {
        "preprocess",
        "alignment",
        "phoneme_model",
        "estimate_phonemes",
        "build_phoneme_results",
        "score",
        "feedback",
        "total",
    }
    assert result["timings"]["total"] >= 0
