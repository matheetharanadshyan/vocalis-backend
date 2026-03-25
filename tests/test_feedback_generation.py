from __future__ import annotations

import asyncio
from types import ModuleType, SimpleNamespace


def sample_scoring_payload() -> dict:
    return {
        "phoneme_results": [
            {
                "expected_phoneme": "TH",
                "predicted_phoneme": "T",
                "confidence_score": 0.45,
                "severity": "moderate",
                "start_time": 0.0,
                "end_time": 0.2,
                "word": "think",
                "source": "estimated_from_word_alignment",
                "error_type": "substitution",
                "severity_score": 0.8,
                "importance_weight": 1.35,
                "phoneme_score": 0.21,
            },
            {
                "expected_phoneme": "IH",
                "predicted_phoneme": "IH",
                "confidence_score": 0.92,
                "severity": "none",
                "start_time": 0.2,
                "end_time": 0.4,
                "word": "think",
                "source": "estimated_from_word_alignment",
                "error_type": "none",
                "severity_score": 0.08,
                "importance_weight": 1.0,
                "phoneme_score": 0.88,
            },
        ],
        "word_scores": [
            {
                "word": "think",
                "weighted_score": 0.54,
                "average_confidence": 0.685,
                "phoneme_count": 2,
                "performance_band": "Needs More Practice",
            }
        ],
        "overall_score": 0.54,
        "performance_band": "Needs More Practice",
    }


def build_feedback_module(
    load_backend_module,
    *,
    use_groq: bool = False,
    groq_api_key: str = "",
    response_content: str = '{"summary":"Nice work.","action_items":["A","B"],"encouragement":"Keep going."}',
):
    config_module = ModuleType("config")
    config_module.settings = SimpleNamespace(
        use_groq=use_groq,
        groq_api_key=groq_api_key,
        groq_model="test-model",
    )

    groq_module = ModuleType("groq")

    class FakeGroq:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            del kwargs
            message = SimpleNamespace(content=response_content)
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    groq_module.Groq = FakeGroq

    return load_backend_module(
        "feedback_generation",
        {
            "config": config_module,
            "groq": groq_module,
        },
    )


def test_build_fallback_feedback_returns_rule_based_payload(load_backend_module) -> None:
    feedback_generation = build_feedback_module(load_backend_module)

    payload = feedback_generation.build_fallback_feedback("think", sample_scoring_payload())

    assert payload["feedback_provider"] == "fallback"
    assert payload["feedback_model"] == "rule-based"
    assert len(payload["action_items"]) == 2
    assert "think" in payload["summary"]


def test_call_groq_parses_structured_json_response(load_backend_module) -> None:
    feedback_generation = build_feedback_module(
        load_backend_module,
        use_groq=True,
        groq_api_key="secret-key",
        response_content='{"summary":"Nice work.","action_items":["Slow down.","Repeat once more."],"encouragement":"Keep going."}',
    )

    payload = feedback_generation._call_groq("think", sample_scoring_payload())

    assert payload == {
        "summary": "Nice work.",
        "action_items": ["Slow down.", "Repeat once more."],
        "encouragement": "Keep going.",
        "feedback_provider": "groq",
        "feedback_model": "test-model",
    }


def test_generate_feedback_falls_back_when_groq_is_disabled(load_backend_module) -> None:
    feedback_generation = build_feedback_module(load_backend_module, use_groq=False)

    payload = asyncio.run(
        feedback_generation.generate_feedback("think", sample_scoring_payload())
    )

    assert payload["feedback_provider"] == "fallback"


def test_generate_feedback_falls_back_when_groq_call_fails(load_backend_module) -> None:
    feedback_generation = build_feedback_module(
        load_backend_module,
        use_groq=True,
        groq_api_key="secret-key",
    )
    feedback_generation._call_groq = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom"))

    payload = asyncio.run(
        feedback_generation.generate_feedback("think", sample_scoring_payload())
    )

    assert payload["feedback_provider"] == "fallback"
