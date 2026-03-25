import json
import random
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from phoneme_estimates import phonemize_text

TARGET_TEXTS_PATH = Path(__file__).with_name("target_texts.json")


class TargetText(TypedDict):
    text: str
    difficulty: str


@lru_cache(maxsize=512)
def get_target_text_phonemes(target_text: str) -> tuple[tuple[str, ...], ...]:
    phoneme_words = phonemize_text(target_text)
    return tuple(tuple(phoneme for phoneme in word) for word in phoneme_words)


def flatten_target_text_phonemes(target_text: str) -> set[str]:
    return {
        phoneme.upper()
        for word in get_target_text_phonemes(target_text)
        for phoneme in word
    }


def load_target_texts() -> list[TargetText]:
    with TARGET_TEXTS_PATH.open("r", encoding="utf-8") as target_text_file:
        target_texts = json.load(target_text_file)

    if not isinstance(target_texts, list) or not target_texts:
        raise ValueError("target_texts.json must contain a non-empty list of target text objects.")

    cleaned_target_texts: list[TargetText] = []

    for target_text in target_texts:
        if not isinstance(target_text, dict):
            continue

        text = str(target_text.get("text", "")).strip()
        difficulty = str(target_text.get("difficulty", "")).strip().lower()

        if not text or difficulty not in {"easy", "medium", "hard"}:
            continue

        cleaned_target_texts.append(
            {
                "text": text,
                "difficulty": difficulty,
            }
        )

    if not cleaned_target_texts:
        raise ValueError("target_texts.json does not contain any valid target text objects.")

    return cleaned_target_texts


class TargetTextManager:
    def __init__(self, target_texts: list[TargetText] | None = None) -> None:
        self._all_target_texts = list(target_texts or load_target_texts())
        self._difficulty_order = ["easy", "medium", "hard"]
        self._remaining_by_difficulty: dict[str, list[TargetText]] = {
            difficulty: [] for difficulty in self._difficulty_order
        }
        self._current_difficulty_index = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        for difficulty in self._difficulty_order:
            difficulty_targets = [
                target_text
                for target_text in self._all_target_texts
                if target_text["difficulty"] == difficulty
            ]
            random.shuffle(difficulty_targets)
            self._remaining_by_difficulty[difficulty] = difficulty_targets

        self._current_difficulty_index = 0

    def next_target(self) -> TargetText:
        if all(not self._remaining_by_difficulty[difficulty] for difficulty in self._difficulty_order):
            self._reshuffle()

        while self._current_difficulty_index < len(self._difficulty_order):
            current_difficulty = self._difficulty_order[self._current_difficulty_index]
            remaining_targets = self._remaining_by_difficulty[current_difficulty]

            if remaining_targets:
                return remaining_targets.pop()

            self._current_difficulty_index += 1

        self._reshuffle()
        return self.next_target()

    def next_target_for_focus(self, focus_phonemes: list[str] | None = None) -> TargetText:
        normalized_focus = [phoneme.strip().upper() for phoneme in (focus_phonemes or []) if phoneme]
        if not normalized_focus:
            return self.next_target()

        if all(not self._remaining_by_difficulty[difficulty] for difficulty in self._difficulty_order):
            self._reshuffle()

        while self._current_difficulty_index < len(self._difficulty_order):
            current_difficulty = self._difficulty_order[self._current_difficulty_index]
            remaining_targets = self._remaining_by_difficulty[current_difficulty]

            if not remaining_targets:
                self._current_difficulty_index += 1
                continue

            best_index = self._find_best_target_index(remaining_targets, normalized_focus)
            if best_index is not None:
                return remaining_targets.pop(best_index)

            return remaining_targets.pop()

        self._reshuffle()
        return self.next_target_for_focus(normalized_focus)

    def match_focus_phonemes(self, target_text: str, focus_phonemes: list[str] | None = None) -> list[str]:
        normalized_focus = [phoneme.strip().upper() for phoneme in (focus_phonemes or []) if phoneme]
        if not normalized_focus:
            return []

        flattened_phonemes = flatten_target_text_phonemes(target_text)
        return [phoneme for phoneme in normalized_focus if phoneme in flattened_phonemes]

    def _find_best_target_index(
        self,
        targets: list[TargetText],
        focus_phonemes: list[str],
    ) -> int | None:
        best_index: int | None = None
        best_score = 0

        for index, target in enumerate(targets):
            flattened_phonemes = flatten_target_text_phonemes(target["text"])
            match_score = sum(1 for phoneme in focus_phonemes if phoneme in flattened_phonemes)

            if match_score > best_score:
                best_index = index
                best_score = match_score

        return best_index if best_score > 0 else None
