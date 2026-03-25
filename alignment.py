from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

from config import settings


@dataclass
class AlignmentBundle:
    processor: AutoProcessor
    model: AutoModelForCTC
    blank_token_id: int
    vocabulary: dict[str, int]
    inference_device: str
    is_quantized: bool


_alignment_bundle: AlignmentBundle | None = None


def get_supported_quantization_engine() -> str | None:
    supported_engines = set(torch.backends.quantized.supported_engines)

    if "qnnpack" in supported_engines:
        return "qnnpack"

    if "fbgemm" in supported_engines:
        return "fbgemm"

    return None


def load_alignment_bundle() -> AlignmentBundle:
    global _alignment_bundle

    if _alignment_bundle is not None:
        return _alignment_bundle

    processor = AutoProcessor.from_pretrained(settings.huggingface_model_id)
    model = AutoModelForCTC.from_pretrained(settings.huggingface_model_id)
    inference_device = settings.device
    is_quantized = False

    if settings.use_quantized_ctc_model and settings.device == "cpu":
        quantization_engine = get_supported_quantization_engine()

        if quantization_engine is not None:
            torch.backends.quantized.engine = quantization_engine
            try:
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                inference_device = "cpu"
                is_quantized = True
            except Exception:
                is_quantized = False

    if not is_quantized:
        model.to(inference_device)

    model.eval()

    tokenizer = processor.tokenizer
    blank_token_id = tokenizer.pad_token_id
    if blank_token_id is None:
        raise ValueError("The tokenizer does not define a pad token for CTC blank alignment.")

    _alignment_bundle = AlignmentBundle(
        processor=processor,
        model=model,
        blank_token_id=blank_token_id,
        vocabulary=tokenizer.get_vocab(),
        inference_device=inference_device,
        is_quantized=is_quantized,
    )
    return _alignment_bundle


def normalize_target_text(target_text: str, vocabulary: dict[str, int]) -> str:
    normalized_text = target_text.strip().lower().replace(" ", "|")

    unsupported_tokens = sorted({character for character in normalized_text if character not in vocabulary})
    if unsupported_tokens:
        unsupported_list = ", ".join(unsupported_tokens)
        raise ValueError(f"Unsupported target text tokens for alignment: {unsupported_list}")

    return normalized_text


def build_extended_sequence(target_ids: list[int], blank_token_id: int) -> list[int]:
    extended_sequence = [blank_token_id]

    for token_id in target_ids:
        extended_sequence.append(token_id)
        extended_sequence.append(blank_token_id)

    return extended_sequence


def compute_viterbi_path(
    log_probs: np.ndarray,
    extended_sequence: list[int],
    blank_token_id: int,
) -> list[tuple[int, int]]:
    num_frames = log_probs.shape[0]
    num_states = len(extended_sequence)

    trellis = np.full((num_frames, num_states), -inf, dtype=np.float32)
    backpointers = np.full((num_frames, num_states), -1, dtype=np.int16)

    trellis[0, 0] = log_probs[0, blank_token_id]
    if num_states > 1:
        trellis[0, 1] = log_probs[0, extended_sequence[1]]

    for frame_index in range(1, num_frames):
        for state_index, token_id in enumerate(extended_sequence):
            candidates: list[tuple[float, int]] = [(trellis[frame_index - 1, state_index], state_index)]

            if state_index - 1 >= 0:
                candidates.append((trellis[frame_index - 1, state_index - 1], state_index - 1))

            if (
                state_index - 2 >= 0
                and token_id != blank_token_id
                and token_id != extended_sequence[state_index - 2]
            ):
                candidates.append((trellis[frame_index - 1, state_index - 2], state_index - 2))

            best_score, best_previous_state = max(candidates, key=lambda candidate: candidate[0])
            trellis[frame_index, state_index] = best_score + log_probs[frame_index, token_id]
            backpointers[frame_index, state_index] = best_previous_state

    last_state = num_states - 1
    if num_states > 1 and trellis[num_frames - 1, num_states - 2] > trellis[num_frames - 1, last_state]:
        last_state = num_states - 2

    path: list[tuple[int, int]] = []
    state_index = last_state

    for frame_index in range(num_frames - 1, -1, -1):
        path.append((frame_index, state_index))
        previous_state = backpointers[frame_index, state_index]
        if previous_state < 0:
            break
        state_index = int(previous_state)

    path.reverse()
    return path


def extract_character_segments(
    path: list[tuple[int, int]],
    extended_sequence: list[int],
    normalized_target_text: str,
    log_probs: np.ndarray,
    frame_duration_seconds: float,
    blank_token_id: int,
) -> list[dict]:
    character_segments: list[dict] = []
    current_segment: dict | None = None

    for frame_index, state_index in path:
        token_id = extended_sequence[state_index]

        if token_id == blank_token_id or state_index % 2 == 0:
            current_segment = None
            continue

        character_index = (state_index - 1) // 2
        character = normalized_target_text[character_index]
        confidence = float(np.exp(log_probs[frame_index, token_id]))

        if current_segment and current_segment["index"] == character_index:
            current_segment["end_time"] = (frame_index + 1) * frame_duration_seconds
            current_segment["confidence_values"].append(confidence)
            continue

        current_segment = {
            "index": character_index,
            "character": character,
            "start_time": frame_index * frame_duration_seconds,
            "end_time": (frame_index + 1) * frame_duration_seconds,
            "confidence_values": [confidence],
        }
        character_segments.append(current_segment)

    for segment in character_segments:
        confidence_values = segment.pop("confidence_values")
        segment["confidence"] = float(sum(confidence_values) / len(confidence_values))

    return character_segments


def extract_word_segments(character_segments: list[dict]) -> list[dict]:
    word_segments: list[dict] = []
    current_word_characters: list[str] = []
    current_word_start_time: float | None = None
    current_word_end_time: float | None = None
    current_word_confidences: list[float] = []

    for segment in character_segments:
        if segment["character"] == "|":
            if current_word_characters:
                word_segments.append(
                    {
                        "text": "".join(current_word_characters),
                        "start_time": current_word_start_time,
                        "end_time": current_word_end_time,
                        "confidence": float(sum(current_word_confidences) / len(current_word_confidences)),
                    }
                )
            current_word_characters = []
            current_word_start_time = None
            current_word_end_time = None
            current_word_confidences = []
            continue

        if current_word_start_time is None:
            current_word_start_time = segment["start_time"]

        current_word_characters.append(segment["character"])
        current_word_end_time = segment["end_time"]
        current_word_confidences.append(segment["confidence"])

    if current_word_characters:
        word_segments.append(
            {
                "text": "".join(current_word_characters),
                "start_time": current_word_start_time,
                "end_time": current_word_end_time,
                "confidence": float(sum(current_word_confidences) / len(current_word_confidences)),
            }
        )

    return word_segments


def align_target_text(processed_audio: np.ndarray, target_text: str) -> dict:
    alignment_bundle = load_alignment_bundle()
    normalized_target_text = normalize_target_text(target_text, alignment_bundle.vocabulary)
    target_ids = [alignment_bundle.vocabulary[character] for character in normalized_target_text]

    if not target_ids:
        raise ValueError("Target text is empty after normalization.")

    inputs = alignment_bundle.processor(
        processed_audio,
        sampling_rate=settings.sample_rate,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(alignment_bundle.inference_device)

    with torch.no_grad():
        logits = alignment_bundle.model(input_values).logits[0]

    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    extended_sequence = build_extended_sequence(target_ids, alignment_bundle.blank_token_id)
    path = compute_viterbi_path(log_probs, extended_sequence, alignment_bundle.blank_token_id)

    frame_duration_seconds = float(processed_audio.shape[0] / settings.sample_rate / max(1, log_probs.shape[0]))

    character_segments = extract_character_segments(
        path=path,
        extended_sequence=extended_sequence,
        normalized_target_text=normalized_target_text,
        log_probs=log_probs,
        frame_duration_seconds=frame_duration_seconds,
        blank_token_id=alignment_bundle.blank_token_id,
    )
    word_segments = extract_word_segments(character_segments)

    return {
        "normalized_target_text": normalized_target_text,
        "character_segments": character_segments,
        "word_segments": word_segments,
        "model_quantized": alignment_bundle.is_quantized,
        "model_device": alignment_bundle.inference_device,
    }
