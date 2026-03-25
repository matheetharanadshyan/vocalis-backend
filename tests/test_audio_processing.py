from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest


class FakeArray:
    def __init__(
        self,
        num_samples: int,
        *,
        ndim: int = 1,
        channels: int = 1,
        dtype: str = "float",
    ) -> None:
        self.ndim = ndim
        self.dtype = dtype
        self.shape = (num_samples,) if ndim == 1 else (num_samples, channels)
        self.size = num_samples if ndim == 1 else num_samples * channels

    def astype(self, dtype) -> "FakeArray":
        return FakeArray(
            self.shape[0],
            ndim=self.ndim,
            channels=self.shape[1] if self.ndim > 1 else 1,
            dtype=str(dtype),
        )


def build_numpy_stub() -> ModuleType:
    numpy_module = ModuleType("numpy")
    numpy_module.ndarray = FakeArray
    numpy_module.float32 = "float32"
    numpy_module.integer = "integer"
    numpy_module.issubdtype = lambda dtype, kind: dtype == kind
    numpy_module.iinfo = lambda dtype: SimpleNamespace(max=32767)
    numpy_module.mean = lambda audio, axis=1, dtype=None: FakeArray(audio.shape[0])
    return numpy_module


def build_scipy_stubs() -> dict[str, ModuleType]:
    scipy_module = ModuleType("scipy")
    scipy_io_module = ModuleType("scipy.io")
    wavfile_module = ModuleType("scipy.io.wavfile")
    signal_module = ModuleType("scipy.signal")

    wavfile_module.read = lambda *_args, **_kwargs: (_args, _kwargs)
    signal_module.resample_poly = lambda audio, _target, _original: audio

    scipy_io_module.wavfile = wavfile_module
    scipy_module.io = scipy_io_module
    scipy_module.signal = signal_module

    return {
        "scipy": scipy_module,
        "scipy.io": scipy_io_module,
        "scipy.io.wavfile": wavfile_module,
        "scipy.signal": signal_module,
    }


def build_audio_processing_module(load_backend_module):
    config_module = ModuleType("config")
    config_module.settings = SimpleNamespace(
        max_websocket_buffer_mb=1,
        min_audio_duration_seconds=0.5,
        max_audio_duration_seconds=10.0,
        sample_rate=16000,
    )

    return load_backend_module(
        "audio_processing",
        {
            "config": config_module,
            "numpy": build_numpy_stub(),
            **build_scipy_stubs(),
        },
    )


def test_validate_audio_bytes_rejects_empty_payload(load_backend_module) -> None:
    audio_processing = build_audio_processing_module(load_backend_module)

    with pytest.raises(audio_processing.AudioValidationError, match="Audio recording is empty"):
        audio_processing.validate_audio_bytes(b"")


def test_validate_audio_bytes_rejects_short_recordings(load_backend_module) -> None:
    audio_processing = build_audio_processing_module(load_backend_module)
    audio_processing.load_wav_audio = lambda _audio_bytes: (16000, FakeArray(1000))

    with pytest.raises(audio_processing.AudioValidationError, match="too short"):
        audio_processing.validate_audio_bytes(b"wav")


def test_preprocess_audio_bytes_returns_expected_metadata(load_backend_module) -> None:
    audio_processing = build_audio_processing_module(load_backend_module)
    raw_audio = FakeArray(96000, ndim=2, channels=2)
    processed_audio = FakeArray(32000)

    audio_processing.validate_audio_bytes = lambda _audio_bytes: (48000, raw_audio)
    audio_processing.convert_to_float32 = lambda audio: audio
    audio_processing.convert_to_mono = lambda audio: FakeArray(audio.shape[0])
    audio_processing.downsample_audio = lambda audio, sample_rate: processed_audio

    payload = audio_processing.preprocess_audio_bytes(b"wav")

    assert payload["original_sample_rate"] == 48000
    assert payload["processed_sample_rate"] == 16000
    assert payload["original_channels"] == 2
    assert payload["processed_channels"] == 1
    assert payload["num_samples"] == 32000
    assert payload["duration_seconds"] == 2.0
    assert payload["processed_audio"] is processed_audio
