from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

from config import settings


@dataclass
class PhonemeModelBundle:
    processor: AutoProcessor
    model: AutoModelForCTC
    inference_device: str
    is_quantized: bool


_phoneme_model_bundle: PhonemeModelBundle | None = None


def configure_phonemizer_environment() -> None:
    candidate_library_paths = [
        os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"),
        "/opt/homebrew/lib/libespeak.dylib",
        "/usr/local/lib/libespeak.dylib",
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/x86_64-linux-gnu/libespeak.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak.so.1",
        "/usr/lib64/libespeak-ng.so.1",
        "/usr/lib64/libespeak.so.1",
        "/usr/lib/libespeak-ng.so.1",
        "/usr/lib/libespeak.so.1",
    ]

    for candidate_path in candidate_library_paths:
        if candidate_path and Path(candidate_path).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = candidate_path
            return

    espeak_prefix = shutil.which("espeak")
    if espeak_prefix:
        sibling_library_path = str(Path(espeak_prefix).resolve().parent.parent / "lib" / "libespeak.dylib")
        if Path(sibling_library_path).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = sibling_library_path
            return

    espeak_ng_prefix = shutil.which("espeak-ng")
    if espeak_ng_prefix:
        sibling_candidates = [
            Path(espeak_ng_prefix).resolve().parent.parent / "lib" / "libespeak-ng.so.1",
            Path(espeak_ng_prefix).resolve().parent.parent / "lib" / "libespeak.so.1",
        ]

        for candidate_path in sibling_candidates:
            if candidate_path.exists():
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(candidate_path)
                return


def load_phoneme_model_bundle() -> PhonemeModelBundle:
    global _phoneme_model_bundle

    if _phoneme_model_bundle is not None:
        return _phoneme_model_bundle

    configure_phonemizer_environment()
    processor = AutoProcessor.from_pretrained(settings.phoneme_model_id)
    model = AutoModelForCTC.from_pretrained(settings.phoneme_model_id)
    inference_device = settings.device
    is_quantized = False

    if settings.use_quantized_phoneme_model and settings.device == "cpu":
        try:
            supported_engines = torch.backends.quantized.supported_engines

            if "qnnpack" in supported_engines:
                torch.backends.quantized.engine = "qnnpack"
            elif "fbgemm" in supported_engines:
                torch.backends.quantized.engine = "fbgemm"

            if torch.backends.quantized.engine in {"qnnpack", "fbgemm"}:
                model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
                inference_device = "cpu"
                is_quantized = True
        except Exception:
            inference_device = "cpu"

    if not is_quantized:
        model.to(inference_device)
    model.eval()

    _phoneme_model_bundle = PhonemeModelBundle(
        processor=processor,
        model=model,
        inference_device=inference_device,
        is_quantized=is_quantized,
    )
    return _phoneme_model_bundle


def is_special_token(token: str) -> bool:
    return token in {"<pad>", "<s>", "</s>", "<unk>"} or not token.strip()


def extract_predicted_phoneme_segments(
    predicted_ids: np.ndarray,
    log_probs: np.ndarray,
    processor: AutoProcessor,
    frame_duration_seconds: float,
) -> list[dict]:
    segments: list[dict] = []
    current_segment: dict | None = None

    for frame_index, token_id in enumerate(predicted_ids):
        token = processor.tokenizer.convert_ids_to_tokens(int(token_id))

        if token is None or is_special_token(token):
            current_segment = None
            continue

        confidence = float(np.exp(log_probs[frame_index, int(token_id)]))

        if current_segment and current_segment["phoneme"] == token:
            current_segment["end_time"] = (frame_index + 1) * frame_duration_seconds
            current_segment["confidence_values"].append(confidence)
            continue

        current_segment = {
            "phoneme": token,
            "start_time": frame_index * frame_duration_seconds,
            "end_time": (frame_index + 1) * frame_duration_seconds,
            "confidence_values": [confidence],
        }
        segments.append(current_segment)

    for segment in segments:
        confidence_values = segment.pop("confidence_values")
        segment["confidence"] = float(sum(confidence_values) / len(confidence_values))

    return segments


def run_phoneme_model(processed_audio: np.ndarray) -> dict:
    if not settings.use_phoneme_model:
        return {
            "phoneme_model_used": False,
            "phoneme_model_id": settings.phoneme_model_id,
            "phoneme_model_segments": [],
            "phoneme_model_transcript": "",
            "phoneme_model_error": "Phoneme model disabled in configuration.",
            "phoneme_model_quantized": False,
        }

    try:
        bundle = load_phoneme_model_bundle()
        inputs = bundle.processor(
            processed_audio,
            sampling_rate=settings.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(bundle.inference_device)

        with torch.no_grad():
            logits = bundle.model(input_values).logits[0]

        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        predicted_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        frame_duration_seconds = float(
            processed_audio.shape[0] / settings.sample_rate / max(1, log_probs.shape[0])
        )

        phoneme_model_segments = extract_predicted_phoneme_segments(
            predicted_ids=predicted_ids,
            log_probs=log_probs,
            processor=bundle.processor,
            frame_duration_seconds=frame_duration_seconds,
        )
        phoneme_model_transcript = bundle.processor.batch_decode(
            [predicted_ids],
            skip_special_tokens=True,
        )[0].strip()

        return {
            "phoneme_model_used": True,
            "phoneme_model_id": settings.phoneme_model_id,
            "phoneme_model_segments": phoneme_model_segments,
            "phoneme_model_transcript": phoneme_model_transcript,
            "phoneme_model_error": "",
            "phoneme_model_quantized": bundle.is_quantized,
        }
    except Exception as error:
        return {
            "phoneme_model_used": False,
            "phoneme_model_id": settings.phoneme_model_id,
            "phoneme_model_segments": [],
            "phoneme_model_transcript": "",
            "phoneme_model_error": str(error),
            "phoneme_model_quantized": False,
        }
