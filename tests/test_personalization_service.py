from __future__ import annotations

import asyncio
from types import ModuleType


def build_personalization_service_module(load_backend_module):
    personalization_repository_module = ModuleType("personalization_repository")
    personalization_repository_module.build_user_personalization_summary = lambda user_id: {
        "focus_phonemes": [{"phoneme": "R"}],
    }
    personalization_repository_module.save_attempt = lambda *args, **kwargs: 1
    personalization_repository_module.save_attempt_phoneme_summaries = lambda *args, **kwargs: None
    personalization_repository_module.upsert_phoneme_memory = lambda *args, **kwargs: None

    target_texts_module = ModuleType("target_texts")
    target_texts_module.TargetTextManager = object

    return load_backend_module(
        "personalization_service",
        {
            "personalization_repository": personalization_repository_module,
            "target_texts": target_texts_module,
        },
    )


def test_derive_attempt_focus_phonemes_prefers_latest_weakest_unique_sounds(load_backend_module) -> None:
    personalization_service = build_personalization_service_module(load_backend_module)

    focus_phonemes = personalization_service.derive_attempt_focus_phonemes(
        [
            {
                "expected_phoneme": "R",
                "error_type": "substitution",
                "phoneme_score": 0.42,
                "severity_score": 0.55,
                "importance_weight": 1.0,
            },
            {
                "expected_phoneme": "TH",
                "error_type": "substitution",
                "phoneme_score": 0.18,
                "severity_score": 0.82,
                "importance_weight": 1.35,
            },
            {
                "expected_phoneme": "TH",
                "error_type": "substitution",
                "phoneme_score": 0.21,
                "severity_score": 0.8,
                "importance_weight": 1.35,
            },
            {
                "expected_phoneme": "IH",
                "error_type": "none",
                "phoneme_score": 0.91,
                "severity_score": 0.08,
                "importance_weight": 1.0,
            },
        ]
    )

    assert focus_phonemes == ["TH", "R"]


def test_persist_personalization_state_uses_latest_attempt_focus_for_next_target(load_backend_module) -> None:
    personalization_service = build_personalization_service_module(load_backend_module)

    class FakeTargetTextManager:
        def __init__(self) -> None:
            self.focus_phonemes: list[str] | None = None

        def next_target_for_focus(self, focus_phonemes: list[str] | None = None) -> dict:
            self.focus_phonemes = list(focus_phonemes or [])
            return {"text": "think this through", "difficulty": "medium"}

    async def run_timed_blocking(stage_name: str, func, *args, **kwargs):
        kwargs = {key: value for key, value in kwargs.items() if key != "executor"}
        return func(*args, **kwargs)

    target_text_manager = FakeTargetTextManager()
    personalization_summary, focus_phonemes, next_target_text = asyncio.run(
        personalization_service.persist_personalization_state(
            user_id=7,
            current_target_text={"text": "river road", "difficulty": "easy"},
            alignment={"normalized_target_text": "river|road"},
            scoring_payload={
                "overall_score": 0.51,
                "performance_band": "Needs More Practice",
                "word_scores": [],
                "phoneme_results": [
                    {
                        "expected_phoneme": "TH",
                        "predicted_phoneme": "T",
                        "error_type": "substitution",
                        "phoneme_score": 0.19,
                        "severity_score": 0.84,
                        "importance_weight": 1.35,
                        "word": "think",
                    },
                    {
                        "expected_phoneme": "R",
                        "predicted_phoneme": "R",
                        "error_type": "none",
                        "phoneme_score": 0.92,
                        "severity_score": 0.07,
                        "importance_weight": 1.0,
                        "word": "road",
                    },
                ],
            },
            feedback_payload={
                "summary": "Focus on /TH/.",
                "action_items": ["Practice /TH/", "Slow down"],
                "encouragement": "Keep going",
            },
            phoneme_model_payload={"phoneme_model_transcript": "think"},
            target_text_manager=target_text_manager,
            run_timed_blocking=run_timed_blocking,
            io_executor=object(),
        )
    )

    assert personalization_summary == {"focus_phonemes": [{"phoneme": "R"}]}
    assert focus_phonemes == ["TH"]
    assert target_text_manager.focus_phonemes == ["TH"]
    assert next_target_text == {"text": "think this through", "difficulty": "medium"}
