from __future__ import annotations

import json
from types import ModuleType


def build_target_texts_module(load_backend_module, phoneme_map: dict[str, list[list[str]]]):
    phoneme_estimates_module = ModuleType("phoneme_estimates")
    phoneme_estimates_module.phonemize_text = lambda text: phoneme_map[text]

    return load_backend_module(
        "target_texts",
        {
            "phoneme_estimates": phoneme_estimates_module,
        },
    )


def test_load_target_texts_filters_invalid_entries(load_backend_module, tmp_path) -> None:
    target_texts = build_target_texts_module(load_backend_module, {"hello": [["HH", "AH", "L", "OW"]]})
    target_texts_path = tmp_path / "target_texts.json"
    target_texts_path.write_text(
        json.dumps(
            [
                {"text": "hello", "difficulty": "EASY"},
                {"text": " ", "difficulty": "medium"},
                {"text": "world", "difficulty": "invalid"},
                "not-an-object",
            ]
        ),
        encoding="utf-8",
    )
    target_texts.TARGET_TEXTS_PATH = target_texts_path

    assert target_texts.load_target_texts() == [{"text": "hello", "difficulty": "easy"}]


def test_flatten_target_text_phonemes_uppercases_stubbed_phonemes(load_backend_module) -> None:
    target_texts = build_target_texts_module(
        load_backend_module,
        {"think": [["th", "ih", "ng", "k"]]},
    )

    assert target_texts.flatten_target_text_phonemes("think") == {"TH", "IH", "NG", "K"}


def test_target_text_manager_prefers_targets_that_match_focus_phonemes(load_backend_module) -> None:
    target_texts = build_target_texts_module(
        load_backend_module,
        {
            "cat": [["K", "AE", "T"]],
            "think": [["TH", "IH", "NG", "K"]],
        },
    )
    target_texts.random.shuffle = lambda items: None

    manager = target_texts.TargetTextManager(
        [
            {"text": "cat", "difficulty": "easy"},
            {"text": "think", "difficulty": "easy"},
        ]
    )

    assert manager.next_target_for_focus(["th"]) == {"text": "think", "difficulty": "easy"}
    assert manager.match_focus_phonemes("think", ["th", "ae"]) == ["TH"]
