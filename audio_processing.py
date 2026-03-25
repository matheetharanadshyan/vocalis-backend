from io import BytesIO

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from config import settings


class AudioValidationError(ValueError):
    pass


def load_wav_audio(audio_bytes: bytes) -> tuple[int, np.ndarray]:
    try:
        sample_rate, audio = wavfile.read(BytesIO(audio_bytes))
    except Exception as error:
        raise AudioValidationError("Audio must be a valid WAV recording.") from error
    return sample_rate, audio


def validate_audio_bytes(audio_bytes: bytes) -> tuple[int, np.ndarray]:
    if not audio_bytes:
        raise AudioValidationError("Audio recording is empty.")

    max_payload_bytes = settings.max_websocket_buffer_mb * 1024 * 1024
    if len(audio_bytes) > max_payload_bytes:
        raise AudioValidationError(
            f"Audio recording exceeds the {settings.max_websocket_buffer_mb}MB upload limit."
        )

    sample_rate, audio = load_wav_audio(audio_bytes)
    if sample_rate <= 0:
        raise AudioValidationError("Audio sample rate is invalid.")

    if audio.size == 0:
        raise AudioValidationError("Audio recording does not contain any samples.")

    if audio.ndim > 2:
        raise AudioValidationError("Audio recording uses an unsupported channel layout.")

    num_samples = int(audio.shape[0]) if audio.ndim > 0 else 0
    if num_samples <= 0:
        raise AudioValidationError("Audio recording does not contain any samples.")

    duration_seconds = num_samples / sample_rate
    if duration_seconds < settings.min_audio_duration_seconds:
        raise AudioValidationError(
            f"Audio recording is too short. Please record at least {settings.min_audio_duration_seconds:.1f} seconds."
        )

    if duration_seconds > settings.max_audio_duration_seconds:
        raise AudioValidationError(
            f"Audio recording is too long. Please keep recordings under {settings.max_audio_duration_seconds:.1f} seconds."
        )

    return sample_rate, audio


def convert_to_float32(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        return audio.astype(np.float32) / max_value

    return audio.astype(np.float32)


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio

    return np.mean(audio, axis=1, dtype=np.float32)


def downsample_audio(audio: np.ndarray, original_sample_rate: int) -> np.ndarray:
    target_sample_rate = settings.sample_rate

    if original_sample_rate == target_sample_rate:
        return audio.astype(np.float32)

    return resample_poly(audio, target_sample_rate, original_sample_rate).astype(np.float32)


def preprocess_audio_bytes(audio_bytes: bytes) -> dict:
    original_sample_rate, audio = validate_audio_bytes(audio_bytes)
    original_channels = 1 if audio.ndim == 1 else audio.shape[1]

    audio = convert_to_float32(audio)
    mono_audio = convert_to_mono(audio)
    processed_audio = downsample_audio(mono_audio, original_sample_rate)

    return {
        "original_sample_rate": int(original_sample_rate),
        "processed_sample_rate": int(settings.sample_rate),
        "original_channels": int(original_channels),
        "processed_channels": 1,
        "processed_audio": processed_audio,
        "num_samples": int(processed_audio.shape[0]),
        "duration_seconds": float(processed_audio.shape[0] / settings.sample_rate),
    }
